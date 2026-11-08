"""
============================================================
E_KoopProp — 多数据集 × 多预测长度（自回归滚动 + 独立checkpoint版）
============================================================
修改说明（相比上一版，仅改动2处）：
  ① 新增 CHECKPOINT_META：每个数据集对应自己的预训练 checkpoint
  ② DynamicConfig.__init__ 中 self.checkpoint_path 从 CHECKPOINT_META 读取
     （原来是类变量固定指向 weather，现在按 dataset_name 动态选择）
  其余所有逻辑零修改。

前置条件：
  需要先为每个数据集训练好 checkpoint，命令见文件末尾注释。
  checkpoint 路径格式与 weather 完全一致，只是目录名不同。
============================================================
"""

import os, sys, math, copy, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from scipy.fft import fft, fftfreq

sys.path.insert(0, "/data_p6/zjx/TS_JEPA")
from src.models.encoder import Encoder as OfficialEncoder
from src.models.decoder import LinearDecoder
from src.data_loaders.data_loader import get_evaluation_loaders


# ============================================================
# 0. 多数据集配置表
# ============================================================

# PRED_LENS = [96, 192, 336, 720]
PRED_LENS = [96]

# ── 数据路径 ──────────────────────────────────────────────
DATASET_META = {
    "Weather": {
        "data_path": "/data_p6/zjx/TS_JEPA/data/weather/weather.csv",
    },
    "ETTh1": {
        "data_path": "/data_p6/zjx/TS_JEPA/data/ETT-small/ETTh1.csv",
    },
    "ETTh2": {
        "data_path": "/data_p6/zjx/TS_JEPA/data/ETT-small/ETTh2.csv",
    },
    "ETTm1": {
        "data_path": "/data_p6/zjx/TS_JEPA/data/ETT-small/ETTm1.csv",
    },
    "ETTm2": {
        "data_path": "/data_p6/zjx/TS_JEPA/data/ETT-small/ETTm2.csv",
    },
}

# ── ★ 修改①：每个数据集独立的预训练 checkpoint 路径 ──────
# checkpoint 文件名格式与 weather 完全一致，只有目录名不同
_CKPT_TEMPLATE = (
    "/data_p6/zjx/TS_JEPA/logs/output_model/{dataset}/"
    "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_ratio_patches_10_"
    "encoder_128_2_1_predictor_128_2_1_epoch_5000.pt"
)

CHECKPOINT_META = {
    name: _CKPT_TEMPLATE.format(dataset=name.lower()
                                if name == "Weather" else name)
    for name in DATASET_META
}
# 展开后：
#   Weather → .../output_model/weather/lr_...epoch_5000.pt
#   ETTh1   → .../output_model/ETTh1/lr_...epoch_5000.pt
#   ETTh2   → .../output_model/ETTh2/lr_...epoch_5000.pt
#   ETTm1   → .../output_model/ETTm1/lr_...epoch_5000.pt
#   ETTm2   → .../output_model/ETTm2/lr_...epoch_5000.pt


# ============================================================
# 1. 动态配置
# ============================================================

class DynamicConfig:
    """
    关键设计：
      pred_len 只影响【测试时的自回归步数】，
      训练目标始终是预测下一个 patch（32点），
      与原始 E_KoopProp 完全一致。

    ★ 修改②：checkpoint_path 从 CHECKPOINT_META 按 dataset_name 读取，
      不再是固定的 weather checkpoint。
    """
    # ── 编码器固定参数（与预训练严格对齐）────────────────
    context        = 32
    num_patches    = 10
    embed_dim      = 128
    encoder_nhead  = 2
    encoder_layers = 1
    kernel_size    = 3

    # ── 训练参数（与原始完全一致）──────────────────────────
    batch_size  = 32
    lr          = 1e-5
    num_epochs  = 500
    patience    = 50
    min_delta   = 1e-6

    # ── 频率解耦参数 ────────────────────────────────────────
    spectral_dim = 16

    # ── 频率模块微调参数 ────────────────────────────────────
    finetune_lr        = 1e-4
    finetune_epochs    = 200
    finetune_patience  = 30
    finetune_min_delta = 1e-6
    finetune_warmup    = 10

    # ── Koopman 参数 ────────────────────────────────────────
    koop_spectral_radius = 0.99
    koop_lyap_lambda     = 1e-3
    koop_lyap_epsilon    = 0.1
    koop_pred_steps      = 1      # 训练时始终1步

    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    seed   = 42

    def __init__(self, dataset_name: str, pred_len: int):
        assert dataset_name in DATASET_META, \
            f"未知数据集: {dataset_name}，可选: {list(DATASET_META.keys())}"
        self.dataset_name = dataset_name
        self.pred_len     = pred_len
        self.data_path    = DATASET_META[dataset_name]["data_path"]

        # ★ 修改②：按 dataset_name 动态选择 checkpoint
        self.checkpoint_path = CHECKPOINT_META[dataset_name]

        # 启动时检查 checkpoint 是否存在，提前报错
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"\n[错误] checkpoint 不存在: {self.checkpoint_path}\n"
                f"  请先为 {dataset_name} 运行预训练，命令见文件末尾注释。"
            )

    @property
    def inv_dim(self):
        return self.embed_dim - self.spectral_dim   # 112

    @property
    def autoregressive_steps(self):
        """
        测试时自回归步数 = ceil(pred_len / context)
        实际预测点数 = steps × context，评估时截断到 pred_len。
        """
        return math.ceil(self.pred_len / self.context)

    @property
    def actual_pred_points(self):
        return self.autoregressive_steps * self.context

    def __repr__(self):
        return (f"Config(dataset={self.dataset_name}, "
                f"pred_len={self.pred_len}, "
                f"ar_steps={self.autoregressive_steps}, "
                f"ckpt={os.path.basename(self.checkpoint_path)})")


# ============================================================
# 2. 工具函数（零修改）
# ============================================================

def mse(pred, true):
    return np.mean((pred - true) ** 2)

def mae(pred, true):
    return np.mean(np.abs(pred - true))

def warmup_cosine_lr(opt, warmup, total):
    def fn(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        p = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)

def compute_dominant_freq(patch_np: np.ndarray) -> float:
    spec  = np.abs(fft(patch_np))
    freqs = fftfreq(len(patch_np))
    pos   = freqs > 0
    return float(freqs[pos][np.argmax(spec[pos])])


# ============================================================
# 3. 数据加载（零修改）
# ============================================================

def load_official_data(cfg: DynamicConfig):
    print(f"\n[数据] dataset={cfg.dataset_name} | pred_len={cfg.pred_len} "
          f"(ar_steps={cfg.autoregressive_steps})")
    from torch.utils.data import DataLoader as DL
    import copy as _copy

    full_loader = get_evaluation_loaders(
        cfg.data_path,
        cfg.batch_size,
        ratio_patches=cfg.num_patches,
        mask_ratio=0.7,
    )
    dataset  = full_loader.dataset
    train_df = dataset.train_df
    test_df  = dataset.test_df

    n_val        = int(len(train_df) * 0.1)
    n_tr         = len(train_df) - n_val
    dataset.train_df = train_df[:n_tr]
    val_df           = train_df[n_tr:]

    train_loader = DL(dataset, batch_size=cfg.batch_size,
                      shuffle=True, drop_last=True, num_workers=4)

    val_dataset          = _copy.copy(dataset)
    val_dataset.train_df = val_df
    val_loader = DL(val_dataset, batch_size=cfg.batch_size,
                    shuffle=False, num_workers=4)

    n_te = (len(test_df) - cfg.context * cfg.num_patches) // cfg.context
    print(f"  train={n_tr:,} | val={n_val:,} | "
          f"test可用步数={n_te:,} (每步预测{cfg.context}点)")
    return train_loader, val_loader, test_df


# ============================================================
# 4. 加载官方 Encoder（零修改，自动使用 cfg.checkpoint_path）
# ============================================================

def load_official_encoder(cfg: DynamicConfig) -> OfficialEncoder:
    print(f"\n[加载] {cfg.dataset_name} checkpoint: "
          f"{os.path.basename(cfg.checkpoint_path)}")
    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)

    enc = OfficialEncoder(
        num_patches = cfg.num_patches,
        dim_in      = cfg.context,
        kernel_size = cfg.kernel_size,
        embed_dim   = cfg.embed_dim,
        embed_bias  = True,
        nhead       = cfg.encoder_nhead,
        num_layers  = cfg.encoder_layers,
        jepa        = True,
    ).to(cfg.device)

    enc.load_state_dict(ckpt["encoder"], strict=True)
    n = sum(p.numel() for p in enc.parameters())
    print(f"  ✅ 加载成功 | epoch={ckpt.get('epoch','?')} | 参数量={n:,}")

    x_t = torch.randn(2, cfg.num_patches, cfg.context).to(cfg.device)
    with torch.no_grad():
        z_t = enc(x_t)
        z_s = torch.sum(z_t, dim=1)
    print(f"  ✅ encoder输出: patch_repr={tuple(z_t.shape)}, "
          f"mean={z_s.mean():.4f}, std={z_s.std():.4f}")
    return enc


# ============================================================
# 5. 模型模块（零修改）
# ============================================================

class SpectralHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)


class DecoupledProjector(nn.Module):
    def __init__(self, embed_dim, spectral_dim):
        super().__init__()
        self.proj_inv  = nn.Linear(embed_dim, embed_dim - spectral_dim)
        self.proj_spec = nn.Linear(embed_dim, spectral_dim)
    def forward(self, z):
        return self.proj_inv(z), self.proj_spec(z)


def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    n      = A.shape[-1]
    I      = torch.eye(n, device=A.device, dtype=A.dtype)
    A_skew = (A - A.transpose(-2, -1)) / 2.0
    return torch.linalg.solve(I + A_skew, I - A_skew)


class ODOKoopmanPropagator(nn.Module):
    def __init__(self, spectral_dim, spectral_radius_bound=0.99,
                 lyapunov_lambda=1e-3, lyapunov_epsilon=0.1):
        super().__init__()
        self.d   = spectral_dim
        self.rho = spectral_radius_bound
        self.lam = lyapunov_lambda
        self.eps = lyapunov_epsilon
        self.A_U       = nn.Parameter(
            torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.A_V       = nn.Parameter(
            torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.zeros(spectral_dim))

    def get_K(self):
        U     = cayley_transform(self.A_U)
        V     = cayley_transform(self.A_V)
        sigma = torch.clamp(F.softplus(self.log_sigma), max=self.rho)
        return U @ torch.diag(sigma) @ V.t()

    def forward(self, z_osc, steps=1):
        K = self.get_K()
        if steps == 1:
            return z_osc @ K.t()
        return z_osc @ torch.linalg.matrix_power(K, steps).t()

    def lyapunov_loss(self):
        K = self.get_K()
        I = torch.eye(self.d, device=K.device, dtype=K.dtype)
        return self.lam * F.relu(
            torch.norm(K.t() @ K - I, p='fro') - self.eps)

    @torch.no_grad()
    def spectral_radius(self):
        return torch.linalg.eigvals(self.get_K()).abs().max().item()

    @torch.no_grad()
    def singular_values(self):
        return self.get_K().svd().S.cpu()


class EKoopPropEncoder(nn.Module):
    def __init__(self, official_enc, decoupler, propagator, pred_steps=1):
        super().__init__()
        self.official_enc = official_enc
        self.decoupler    = decoupler
        self.propagator   = propagator
        self.pred_steps   = pred_steps
        for p in self.official_enc.parameters(): p.requires_grad_(False)
        for p in self.decoupler.parameters():    p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            z            = torch.sum(self.official_enc(x), dim=1)
            z_inv, z_osc = self.decoupler(z)
        z_osc_prop = self.propagator(z_osc, self.pred_steps)
        return torch.cat([z_inv, z_osc_prop], dim=-1)


# ============================================================
# 6. 训练函数（零修改）
# ============================================================

def train_forecast_head(encoder_fn, decoder, train_loader, val_loader,
                        cfg, tag):
    print(f"\n  ── 训练预测头: {tag} ──")
    decoder = decoder.to(cfg.device)
    opt = torch.optim.AdamW(decoder.parameters(),
                            lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=1e-7)
    best, pat, best_st = float('inf'), 0, None

    for ep in range(1, cfg.num_epochs + 1):
        decoder.train()
        for context_patches, target_patch in train_loader:
            context_patches = context_patches.to(cfg.device)
            target_patch    = target_patch.to(cfg.device)
            with torch.no_grad():
                z = encoder_fn(context_patches)
            pred = decoder(z)
            loss = F.mse_loss(pred, target_patch)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
        sch.step()

        decoder.eval(); vl = 0.0
        with torch.no_grad():
            for context_patches, target_patch in val_loader:
                z    = encoder_fn(context_patches.to(cfg.device))
                pred = decoder(z)
                vl  += F.mse_loss(pred,
                                  target_patch.to(cfg.device)).item()
        vl /= max(1, len(val_loader))

        if ep % 50 == 0 or ep <= 3:
            print(f"    ep {ep:3d} | val_mse={vl:.6f}")
        if vl < best - cfg.min_delta:
            best, pat, best_st = vl, 0, copy.deepcopy(decoder.state_dict())
        else:
            pat += 1
        if pat >= cfg.patience:
            print(f"    [早停] ep={ep} | best_val_mse={best:.6f}"); break

    if best_st: decoder.load_state_dict(best_st)
    return decoder


def finetune_freq_module(encoder_fn, freq_module, train_loader,
                         val_loader, cfg, tag):
    print(f"\n  [频率模块微调] {tag}")
    freq_module = freq_module.to(cfg.device)

    train_freqs = []
    for cp, _ in train_loader:
        last = cp[:, -1, :].numpy()
        for i in range(len(last)):
            train_freqs.append(compute_dominant_freq(last[i]))
    freq_arr  = np.array(train_freqs)
    freq_mean = float(freq_arr.mean())
    freq_std  = float(freq_arr.std()) + 1e-8

    def get_freq_label(cp_np):
        last   = cp_np[:, -1, :]
        labels = np.array([compute_dominant_freq(last[i])
                           for i in range(len(last))])
        return torch.tensor((labels - freq_mean) / freq_std,
                            dtype=torch.float32)

    opt = torch.optim.AdamW(freq_module.parameters(),
                            lr=cfg.finetune_lr, weight_decay=1e-4)
    sch = warmup_cosine_lr(opt, cfg.finetune_warmup, cfg.finetune_epochs)
    best, pat, best_st = float('inf'), 0, None

    for ep in range(1, cfg.finetune_epochs + 1):
        freq_module.train()
        for cp, _ in train_loader:
            fl = get_freq_label(cp.numpy()).to(cfg.device)
            with torch.no_grad():
                z = encoder_fn(cp.to(cfg.device))
            loss = F.mse_loss(freq_module(z), fl)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(freq_module.parameters(), 1.0)
            opt.step()
        sch.step()

        freq_module.eval(); vl = 0.0
        with torch.no_grad():
            for cp, _ in val_loader:
                fl = get_freq_label(cp.numpy()).to(cfg.device)
                z  = encoder_fn(cp.to(cfg.device))
                vl += F.mse_loss(freq_module(z), fl).item()
        vl /= max(1, len(val_loader))

        if ep % 30 == 0 or ep <= 3:
            print(f"    ep {ep:3d} | val_freq_mse={vl:.6f}")
        if vl < best - cfg.finetune_min_delta:
            best, pat, best_st = vl, 0, copy.deepcopy(freq_module.state_dict())
        else:
            pat += 1
        if pat >= cfg.finetune_patience:
            print(f"    [早停] ep={ep} | best={best:.6f}"); break

    if best_st: freq_module.load_state_dict(best_st)
    return freq_module


def train_koopprop_head(ekoop_enc, decoder, train_loader, val_loader,
                        cfg, tag="E_KoopProp"):
    print(f"\n  ── 训练 {tag} ──")
    ekoop_enc = ekoop_enc.to(cfg.device)
    decoder   = decoder.to(cfg.device)
    trainable = (list(ekoop_enc.propagator.parameters()) +
                 list(decoder.parameters()))

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=1e-7)
    best, pat = float('inf'), 0
    best_prop_st = best_dec_st = None

    for ep in range(1, cfg.num_epochs + 1):
        ekoop_enc.train(); decoder.train()
        ep_pred = ep_lyap = nb = 0
        for cp, tp in train_loader:
            cp = cp.to(cfg.device); tp = tp.to(cfg.device)
            z      = ekoop_enc(cp)
            pred   = decoder(z)
            L_pred = F.mse_loss(pred, tp)
            L_lyap = ekoop_enc.propagator.lyapunov_loss()
            loss   = L_pred + L_lyap
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            ep_pred += L_pred.item(); ep_lyap += L_lyap.item(); nb += 1
        sch.step()

        ekoop_enc.eval(); decoder.eval(); vl = 0.0
        with torch.no_grad():
            for cp, tp in val_loader:
                z    = ekoop_enc(cp.to(cfg.device))
                pred = decoder(z)
                vl  += F.mse_loss(pred, tp.to(cfg.device)).item()
        vl /= max(1, len(val_loader))

        if ep % 50 == 0 or ep <= 3:
            rho = ekoop_enc.propagator.spectral_radius()
            print(f"    ep {ep:3d} | pred={ep_pred/nb:.5f} | "
                  f"lyap={ep_lyap/nb:.5f} | val={vl:.5f} | ρ(K)={rho:.4f}")
        if vl < best - cfg.min_delta:
            best = vl; pat = 0
            best_prop_st = copy.deepcopy(ekoop_enc.propagator.state_dict())
            best_dec_st  = copy.deepcopy(decoder.state_dict())
        else:
            pat += 1
        if pat >= cfg.patience:
            print(f"    [早停] ep={ep} | best_val_mse={best:.6f}"); break

    if best_prop_st: ekoop_enc.propagator.load_state_dict(best_prop_st)
    if best_dec_st:  decoder.load_state_dict(best_dec_st)
    rho_f = ekoop_enc.propagator.spectral_radius()
    sv    = ekoop_enc.propagator.singular_values()
    print(f"    ✅ 完成 | best_val={best:.6f} | ρ(K)={rho_f:.6f} | "
          f"σ∈[{sv.min():.4f},{sv.max():.4f}]")
    return ekoop_enc, decoder


# ============================================================
# 7. 自回归滚动测试（零修改）
# ============================================================

def rolling_test_autoregressive(encoder_fn, decoder, test_df,
                                cfg, tag):
    print(f"\n  ── 自回归滚动测试: {tag} ──")
    context     = cfg.context
    num_patches = cfg.num_patches
    n_steps     = cfg.autoregressive_steps
    pred_len    = cfg.pred_len

    window_size = num_patches * context + pred_len
    stride      = max(pred_len, context)
    num_windows = max(1, (len(test_df) - window_size) // stride + 1)

    print(f"    test_df长度={len(test_df):,} | pred_len={pred_len} | "
          f"ar_steps={n_steps} | 评估窗口数={num_windows:,}")

    decoder.eval()
    all_mse, all_mae = [], []

    for w in range(num_windows):
        start     = w * stride
        input_buf = test_df[start : start + num_patches * context].clone()

        if start + num_patches * context + pred_len > len(test_df):
            break

        preds = []
        for step in range(n_steps):
            ctx = (input_buf[-num_patches * context:]
                   .reshape(num_patches, context)
                   .unsqueeze(0)
                   .to(cfg.device))
            with torch.no_grad():
                pred_patch = decoder(encoder_fn(ctx)).flatten().cpu()
            preds.append(pred_patch)
            input_buf = torch.cat([input_buf, pred_patch], dim=0)

        pred_full = torch.cat(preds, dim=0)[:pred_len].numpy()
        tgt_start = start + num_patches * context
        target    = test_df[tgt_start : tgt_start + pred_len].numpy()

        if len(target) < pred_len:
            break

        all_mse.append(mse(pred_full, target))
        all_mae.append(mae(pred_full, target))

    result = {"MSE": float(np.mean(all_mse)), "MAE": float(np.mean(all_mae))}
    print(f"    ✅ MSE={result['MSE']:.6f} | MAE={result['MAE']:.6f} "
          f"(基于{len(all_mse)}个窗口)")
    return result


# ============================================================
# 8. 单数据集单预测长度完整消融链（零修改）
# ============================================================

def run_single(dataset_name: str, pred_len: int) -> dict:
    cfg = DynamicConfig(dataset_name, pred_len)   # ← 自动选 checkpoint

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print("\n" + "█" * 70)
    print(f"█  dataset={dataset_name} | pred_len={pred_len} "
          f"| ar_steps={cfg.autoregressive_steps}")
    print(f"█  checkpoint: {os.path.basename(cfg.checkpoint_path)}")
    print("█" * 70)

    train_loader, val_loader, test_df = load_official_data(cfg)
    official_enc = load_official_encoder(cfg)
    for p in official_enc.parameters(): p.requires_grad = False
    official_enc.eval()

    def base_enc_fn(x):
        return torch.sum(official_enc(x.to(cfg.device)), dim=1)

    results = {}

    # ── A_Scratch ─────────────────────────────────────────
    print(f"\n{'='*65}\n【A_Scratch】")
    scratch_enc = OfficialEncoder(
        num_patches=cfg.num_patches, dim_in=cfg.context,
        kernel_size=cfg.kernel_size, embed_dim=cfg.embed_dim,
        embed_bias=True, nhead=cfg.encoder_nhead,
        num_layers=cfg.encoder_layers, jepa=True,
    ).to(cfg.device)
    for p in scratch_enc.parameters(): p.requires_grad = False
    scratch_enc.eval()
    def scratch_fn(x):
        return torch.sum(scratch_enc(x.to(cfg.device)), dim=1)
    dec_A = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.context)
    dec_A = train_forecast_head(scratch_fn, dec_A,
                                train_loader, val_loader, cfg, "A_Scratch")
    results["A_Scratch"] = rolling_test_autoregressive(
        scratch_fn, dec_A, test_df, cfg, "A_Scratch")

    # ── B_JEPA ────────────────────────────────────────────
    print(f"\n{'='*65}\n【B_JEPA】")
    dec_B = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.context)
    dec_B = train_forecast_head(base_enc_fn, dec_B,
                                train_loader, val_loader, cfg, "B_JEPA")
    results["B_JEPA"] = rolling_test_autoregressive(
        base_enc_fn, dec_B, test_df, cfg, "B_JEPA")

    # ── C_Direct ──────────────────────────────────────────
    print(f"\n{'='*65}\n【C_Direct】")
    sh_C = SpectralHead(cfg.embed_dim)
    finetune_freq_module(base_enc_fn, sh_C,
                         train_loader, val_loader, cfg, "C_Direct")
    dec_C = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.context)
    dec_C = train_forecast_head(base_enc_fn, dec_C,
                                train_loader, val_loader, cfg, "C_Direct")
    results["C_Direct"] = rolling_test_autoregressive(
        base_enc_fn, dec_C, test_df, cfg, "C_Direct")

    # ── D_Decoupled ───────────────────────────────────────
    print(f"\n{'='*65}\n【D_Decoupled】")
    decoupler = DecoupledProjector(cfg.embed_dim, cfg.spectral_dim).to(cfg.device)
    sh_D      = SpectralHead(cfg.spectral_dim).to(cfg.device)

    class _FreqModD(nn.Module):
        def __init__(self, d, s):
            super().__init__()
            self.dec = d; self.sh = s
        def forward(self, z):
            _, zs = self.dec(z)
            return self.sh(zs)

    freq_mod_D = finetune_freq_module(
        base_enc_fn, _FreqModD(decoupler, sh_D),
        train_loader, val_loader, cfg, "D_Decoupled"
    )
    freq_mod_D.eval()
    for p in freq_mod_D.parameters(): p.requires_grad = False

    def decoupled_fn(x):
        z = base_enc_fn(x)
        zi, zs = freq_mod_D.dec(z)
        return torch.cat([zi, zs], dim=-1)

    dec_D = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.context)
    dec_D = train_forecast_head(decoupled_fn, dec_D,
                                train_loader, val_loader, cfg, "D_Decoupled")
    results["D_Decoupled"] = rolling_test_autoregressive(
        decoupled_fn, dec_D, test_df, cfg, "D_Decoupled")

    # ── E_KoopProp ────────────────────────────────────────
    print(f"\n{'='*65}\n【E_KoopProp】")
    propagator = ODOKoopmanPropagator(
        spectral_dim=cfg.spectral_dim,
        spectral_radius_bound=cfg.koop_spectral_radius,
        lyapunov_lambda=cfg.koop_lyap_lambda,
        lyapunov_epsilon=cfg.koop_lyap_epsilon,
    )
    ekoop_enc = EKoopPropEncoder(
        official_enc=official_enc,
        decoupler=freq_mod_D.dec,
        propagator=propagator,
        pred_steps=cfg.koop_pred_steps,
    )
    dec_E = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.context)
    ekoop_enc, dec_E = train_koopprop_head(
        ekoop_enc, dec_E, train_loader, val_loader, cfg, "E_KoopProp"
    )
    def e_enc_fn(x): return ekoop_enc(x.to(cfg.device))
    results["E_KoopProp"] = rolling_test_autoregressive(
        e_enc_fn, dec_E, test_df, cfg, "E_KoopProp")

    # ── 单组消融表 ────────────────────────────────────────
    bm = results["B_JEPA"]["MSE"]
    print(f"\n{'─'*65}")
    print(f"  {dataset_name} | pred_len={pred_len} "
          f"(ar_steps={cfg.autoregressive_steps}) 消融结果:")
    for name, res in results.items():
        delta = (f"{(res['MSE']-bm)/bm*100:+.1f}%"
                 if name != "B_JEPA" else "(基准)")
        print(f"    {name:<20} MSE={res['MSE']:.6f}  "
              f"MAE={res['MAE']:.6f}  {delta}")
    print(f"{'─'*65}")

    return results


# ============================================================
# 9. 网格实验主函数（零修改）
# ============================================================

def run_grid(
    datasets  = ["Weather", "ETTh1", "ETTm1", "ETTh2", "ETTm2"],
    pred_lens = PRED_LENS,
):
    # ── 启动前检查所有 checkpoint 是否存在 ────────────────
    print("=" * 70)
    print("📋 Checkpoint 可用性检查")
    all_ok = True
    for ds in datasets:
        ckpt = CHECKPOINT_META[ds]
        status = "✅" if os.path.exists(ckpt) else "❌ 缺失"
        print(f"  {ds:<10} {status}  {os.path.basename(ckpt)}")
        if not os.path.exists(ckpt):
            all_ok = False
    if not all_ok:
        print("\n⚠️  部分 checkpoint 缺失，请先完成预训练再运行网格实验。")
        print("     预训练命令见文件末尾注释。")
        print("=" * 70)
        # 只跑有 checkpoint 的数据集
        datasets = [ds for ds in datasets
                    if os.path.exists(CHECKPOINT_META[ds])]
        print(f"  → 仅运行已有 checkpoint 的数据集: {datasets}")

    all_records = []

    print("\n" + "=" * 70)
    print("📊 KoopJEPA 网格实验（自回归多步预测 + 独立checkpoint）")
    print(f"   数据集: {datasets}")
    print(f"   预测长度: {pred_lens}")
    print(f"   训练目标: 始终预测下一个patch（32点）")
    print(f"   测试方式: 自回归滚动 ceil(pred_len/32) 步")
    print("=" * 70)

    for dataset_name, pred_len in itertools.product(datasets, pred_lens):
        results = run_single(dataset_name, pred_len)
        for model_name, res in results.items():
            all_records.append({
                "dataset" : dataset_name,
                "pred_len": pred_len,
                "model"   : model_name,
                "MSE"     : res["MSE"],
                "MAE"     : res["MAE"],
            })

    df = pd.DataFrame(all_records)

    # ── 打印论文主表 ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊  论文主表：E_KoopProp vs B_JEPA（基准）")
    print("=" * 80)
    for dataset_name in datasets:
        print(f"\n  {dataset_name}:")
        print(f"  {'pred_len':>10} │ {'B_MSE':>8} │ {'E_MSE':>8} │ "
              f"{'ΔMSE':>8} │ {'B_MAE':>8} │ {'E_MAE':>8} │ {'ΔMAE':>8}")
        print(f"  {'─'*70}")
        for pred_len in pred_lens:
            sub   = df[(df.dataset==dataset_name) & (df.pred_len==pred_len)]
            b_mse = sub[sub.model=="B_JEPA"]["MSE"].values
            e_mse = sub[sub.model=="E_KoopProp"]["MSE"].values
            b_mae = sub[sub.model=="B_JEPA"]["MAE"].values
            e_mae = sub[sub.model=="E_KoopProp"]["MAE"].values
            if len(b_mse) and len(e_mse):
                d_mse = f"{(e_mse[0]-b_mse[0])/b_mse[0]*100:+.1f}%"
                d_mae = f"{(e_mae[0]-b_mae[0])/b_mae[0]*100:+.1f}%"
                print(f"  {pred_len:>10} │ {b_mse[0]:>8.4f} │ "
                      f"{e_mse[0]:>8.4f} │ {d_mse:>8} │ "
                      f"{b_mae[0]:>8.4f} │ {e_mae[0]:>8.4f} │ {d_mae:>8}")

    save_path = "/data_p6/zjx/TS_JEPA/results/koopjepa_grid_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n  ✅ 完整结果已保存至: {save_path}")
    return df


# ============================================================
# 10. 入口
# ============================================================

if __name__ == "__main__":
    run_grid(
        datasets  = [ "ETTm1", "ETTh2", "ETTm2", "Weather","ETTh1"],
        pred_lens = [32,96],
        # pred_lens = [32,96, 192, 336, 720],
    )


# ============================================================
# 附：各数据集预训练命令（在 /data_p6/zjx/TS_JEPA 目录下执行）
# ============================================================
#
# # Step 0: 建立数据软链接（只需执行一次）
# cd /data_p6/zjx/TS_JEPA
# mkdir -p data/ETTh1 && ln -sf $(pwd)/data/ETT-small/ETTh1.csv data/ETTh1/ETTh1.csv
# mkdir -p data/ETTh2 && ln -sf $(pwd)/data/ETT-small/ETTh2.csv data/ETTh2/ETTh2.csv
# mkdir -p data/ETTm1 && ln -sf $(pwd)/data/ETT-small/ETTm1.csv data/ETTm1/ETTm1.csv
# mkdir -p data/ETTm2 && ln -sf $(pwd)/data/ETT-small/ETTm2.csv data/ETTm2/ETTm2.csv
#
# # Step 1: 并行预训练（每个数据集用不同 GPU）
# ARGS="--batch_size 32 --lr 1e-07 --ema_momentum 0.998 --ratio_patches 10
#       --mask_ratio 0.7 --encoder_embed_dim 128 --encoder_nhead 2
#       --encoder_num_layers 1 --predictor_embed 128 --predictor_nhead 2
#       --predictor_num_layers 1"
#
# CUDA_VISIBLE_DEVICES=0 nohup python pretrain.py --data ETTh1 $ARGS > logs/ETTh1_pretrain.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python pretrain.py --data ETTh2 $ARGS > logs/ETTh2_pretrain.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python pretrain.py --data ETTm1 $ARGS > logs/ETTm1_pretrain.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python pretrain.py --data ETTm2 $ARGS > logs/ETTm2_pretrain.log 2>&1 &
#
# # Step 2: 监控进度
# tail -f logs/ETTh1_pretrain.log
# ls logs/output_model/ETTh1/   # 看 epoch_5000.pt 是否生成