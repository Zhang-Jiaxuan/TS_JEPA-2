"""
============================================================
Step3+E+F+G+E2+G2 — 完整消融链（含谱排序软约束变体）
============================================================
消融链：
  A_Scratch       — 随机初始化 encoder（下界）
  B_JEPA          — 官方预训练 checkpoint（基准）
  C_Direct        — B_JEPA + 频率辅助头（对照）
  D_Decoupled     — B_JEPA + 频率解耦模块（Step3 核心）
  E_KoopProp      — D + ODO Koopman（统一谱半径）
  F_PerDimKoop    — D + ODO Koopman（每维独立谱半径）
  G_JointKoop     — 多频带联合解耦 + 分组 Koopman
  E2_SpectralE    — E + 奇异值排序正则（谱自然分化）★★
  G2_SpectralKoop — 全维度 Koopman + 软频率语义约束   ★★★

E2 设计要点：
  继承 E 的全维度 16×16 Koopman，新增 spectral_sort_loss：
  将 16 个奇异值按索引等分为 low/mid/high 三段，
  强制 mean(σ_low) > mean(σ_mid) > mean(σ_high)（软 margin 约束）
  → 谱分化自然涌现，无需硬切割子空间

G2 设计要点：
  (1) 全维度单一 Koopman（16×16，继承 E 的信息容量）
  (2) 奇异值排序正则（继承 E2 的谱分化）
  (3) 频率对齐损失：让 K 的主奇异向量对应 z_osc 的低频成分
      具体：对 K 的右奇异向量 V_i 在 z_osc 上的投影做 FFT，
            用频率重心作为代理，大奇异值方向应有低频率重心
  (4) 复用 D 的解耦器（冻结），与 E/F 完全公平对比
============================================================
"""

import os, sys, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fft import fft, fftfreq

sys.path.insert(0, "/data_p6/zjx/TS_JEPA")
from src.models.encoder import Encoder as OfficialEncoder
from src.models.decoder import LinearDecoder
from src.data_loaders.data_loader import get_evaluation_loaders


# ============================================================
# 0. 配置
# ============================================================

class Config:
    # ── 路径 ──────────────────────────────────────────────
    checkpoint_path = (
        "/data_p6/zjx/TS_JEPA/logs/output_model/weather/"
        "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_ratio_patches_10_"
        "encoder_128_2_1_predictor_128_2_1_epoch_5000.pt"
    )
    data_path = "/data_p6/zjx/TS_JEPA/data/weather/weather.csv"

    # ── 官方固定参数 ────────────────────────────────────────
    context     = 32
    num_patches = 10
    pred_len    = 32

    # ── encoder 参数 ────────────────────────────────────────
    embed_dim      = 128
    encoder_nhead  = 2
    encoder_layers = 1
    kernel_size    = 3

    # ── 训练参数 ────────────────────────────────────────────
    batch_size  = 32
    lr          = 1e-5
    num_epochs  = 500
    patience    = 50
    min_delta   = 1e-6

    # ── 频率解耦参数（D/E/F/E2/G2 共用）────────────────────
    spectral_dim = 16           # z_osc 维度；z_inv = 112

    # ── 频率模块微调参数（C/D 共用）────────────────────────
    finetune_lr        = 1e-4
    finetune_epochs    = 200
    finetune_patience  = 30
    finetune_min_delta = 1e-6
    finetune_warmup    = 10

    # ── E_KoopProp 专属参数 ─────────────────────────────────
    koop_spectral_radius = 0.99
    koop_lyap_lambda     = 1e-3
    koop_lyap_epsilon    = 0.1
    koop_pred_steps      = 1

    # ── F_PerDimKoop 专属参数 ───────────────────────────────
    perdim_rho_min      = 0.10
    perdim_rho_max      = 0.99
    perdim_lyap_lambda  = 1e-3
    perdim_lyap_epsilon = 0.1

    # ── G_JointKoop 专属参数 ────────────────────────────────
    joint_dim_low   = 6
    joint_dim_mid   = 6
    joint_dim_high  = 4
    joint_freq_lo   = 0.33
    joint_freq_hi   = 0.67
    joint_lyap_lambda  = 1e-3
    joint_lyap_epsilon = 0.1
    joint_spec_lambda  = 1e-2
    joint_rho_soft     = 0.99
    joint_rank_lambda  = 1e-3
    joint_rank_margin  = 0.05
    cross_rank_lambda  = 1e-2
    cross_rank_margin  = 0.02
    joint_orth_lambda  = 1e-2
    joint_finetune_lr        = 1e-4
    joint_finetune_epochs    = 200
    joint_finetune_patience  = 30
    joint_finetune_min_delta = 1e-6
    joint_finetune_warmup    = 10

    # ── E2_SpectralE 专属参数 ───────────────────────────────
    # 在 E 基础上加奇异值排序正则，谱自然分化
    e2_spectral_radius = 0.99   # 全局谱半径上界（同 E）
    e2_lyap_lambda     = 1e-3   # Lyapunov 正则（同 E）
    e2_lyap_epsilon    = 0.1
    e2_sort_lambda     = 5e-2   # 奇异值排序正则强度 ← 核心新增
    e2_sort_margin     = 0.05   # 排序间隔 margin
    e2_n_bands         = 3      # 分段数（low/mid/high）

    # ── G2_SpectralKoop 专属参数 ────────────────────────────
    # 全维度 Koopman + 软频率语义约束
    g2_spectral_radius   = 0.99
    g2_lyap_lambda       = 1e-3
    g2_lyap_epsilon      = 0.1
    g2_sort_lambda       = 5e-2   # 奇异值排序正则（同 E2）
    g2_sort_margin       = 0.05
    g2_freq_align_lambda = 2e-2   # 频率对齐损失强度 ← G2 核心新增
    g2_n_bands           = 3      # 频带数

    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    seed   = 42

    @property
    def inv_dim(self):
        return self.embed_dim - self.spectral_dim   # 112


cfg = Config()


# ============================================================
# 1. 评估指标
# ============================================================

def mse(pred, true):
    return np.mean((pred - true) ** 2)

def mae(pred, true):
    return np.mean(np.abs(pred - true))


# ============================================================
# 2. 加载官方 Encoder
# ============================================================

def load_official_encoder(cfg: Config) -> OfficialEncoder:
    print(f"\n[加载] {os.path.basename(cfg.checkpoint_path)}")
    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    enc = OfficialEncoder(
        num_patches=cfg.num_patches, dim_in=cfg.context,
        kernel_size=cfg.kernel_size, embed_dim=cfg.embed_dim,
        embed_bias=True, nhead=cfg.encoder_nhead,
        num_layers=cfg.encoder_layers, jepa=True,
    ).to(cfg.device)
    enc.load_state_dict(ckpt["encoder"], strict=True)
    n = sum(p.numel() for p in enc.parameters())
    print(f"  ✅ 加载成功 | epoch={ckpt.get('epoch','?')} | 参数量={n:,}")
    x_t = torch.randn(2, cfg.num_patches, cfg.context).to(cfg.device)
    with torch.no_grad():
        z_t = enc(x_t)
        z_s = torch.sum(z_t, dim=1)
    print(f"  ✅ encoder输出: patch_repr={tuple(z_t.shape)}, "
          f"summed={tuple(z_s.shape)}, "
          f"mean={z_s.mean():.4f}, std={z_s.std():.4f}")
    return enc


# ============================================================
# 3. 官方数据加载器
# ============================================================

def load_official_data(cfg: Config):
    print(f"\n[数据] 使用官方 EvaluationDataLoader")
    from torch.utils.data import DataLoader as DL
    import copy as _copy
    full_loader = get_evaluation_loaders(
        cfg.data_path, cfg.batch_size,
        ratio_patches=cfg.num_patches, mask_ratio=0.7,
    )
    dataset  = full_loader.dataset
    train_df = dataset.train_df
    test_df  = dataset.test_df
    n_train_full = len(train_df)
    n_val        = int(n_train_full * 0.1)
    n_tr         = n_train_full - n_val
    dataset.train_df = train_df[:n_tr]
    val_df           = train_df[n_tr:]
    train_loader = DL(dataset, batch_size=cfg.batch_size,
                      shuffle=True, drop_last=True, num_workers=4)
    val_dataset          = _copy.copy(dataset)
    val_dataset.train_df = val_df
    val_loader = DL(val_dataset, batch_size=cfg.batch_size,
                    shuffle=False, num_workers=4)
    n_te_steps = (len(test_df) - cfg.context * cfg.num_patches) // cfg.context
    print(f"  train={len(train_df[:n_tr]):,} | val={len(val_df):,} | "
          f"test滚动步数={n_te_steps:,}")
    return train_loader, val_loader, dataset.test_df


# ============================================================
# 4. 基础频率辅助模块（A~D 共用）
# ============================================================

class SpectralHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, z):
        return self.net(z).squeeze(-1)


class DecoupledProjector(nn.Module):
    """128维 latent → z_inv(112维) + z_osc(16维)"""
    def __init__(self, embed_dim, spectral_dim):
        super().__init__()
        self.proj_inv  = nn.Linear(embed_dim, embed_dim - spectral_dim)
        self.proj_spec = nn.Linear(embed_dim, spectral_dim)
    def forward(self, z):
        return self.proj_inv(z), self.proj_spec(z)


def compute_dominant_freq(patch_np: np.ndarray) -> float:
    spec  = np.abs(fft(patch_np))
    freqs = fftfreq(len(patch_np))
    pos   = freqs > 0
    return float(freqs[pos][np.argmax(spec[pos])])


# ============================================================
# 5. 训练工具
# ============================================================

def warmup_cosine_lr(opt, warmup, total):
    def fn(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        p = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ============================================================
# 6. 下游预测头训练（A / B / C / D 共用）
# ============================================================

def train_forecast_head(encoder_fn, decoder, train_loader, val_loader,
                        cfg: Config, tag: str):
    print(f"\n  ── 训练预测头: {tag} ──")
    decoder = decoder.to(cfg.device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=cfg.lr, weight_decay=1e-4)
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
        decoder.eval()
        vl = 0.0
        with torch.no_grad():
            for context_patches, target_patch in val_loader:
                z    = encoder_fn(context_patches.to(cfg.device))
                pred = decoder(z)
                vl  += F.mse_loss(pred, target_patch.to(cfg.device)).item()
        vl /= max(1, len(val_loader))
        if ep % 50 == 0 or ep <= 3:
            print(f"    ep {ep:3d} | val_mse={vl:.6f}")
        if vl < best - cfg.min_delta:
            best, pat, best_st = vl, 0, copy.deepcopy(decoder.state_dict())
        else:
            pat += 1
        if pat >= cfg.patience:
            print(f"    [早停] ep={ep} | best_val_mse={best:.6f}")
            break
    if best_st:
        decoder.load_state_dict(best_st)
    decoder.eval()
    return decoder


# ============================================================
# 7. 官方滚动测试
# ============================================================

def rolling_test(encoder_fn, decoder, test_df, cfg: Config, tag: str) -> dict:
    print(f"\n  ── 滚动测试: {tag} ──")
    context     = cfg.context
    num_patches = cfg.num_patches
    num_steps   = (len(test_df) - context * num_patches) // context
    print(f"    test_df 长度={len(test_df):,} | 滚动步数={num_steps:,}")
    decoder.eval()
    l_mse, l_mae = [], []
    for step in range(num_steps):
        current_context = (
            test_df[context * step : context * num_patches + context * step]
            .reshape(num_patches, context).unsqueeze(0)
        )
        target_value = test_df[
            context * num_patches + step * context :
            context * num_patches + (step + 1) * context
        ]
        with torch.no_grad():
            z    = encoder_fn(current_context.to(cfg.device))
            pred = decoder(z)
        l_mse.append(mse(pred.flatten().cpu().numpy(), target_value.numpy()))
        l_mae.append(mae(pred.flatten().cpu().numpy(), target_value.numpy()))
    result = {"MSE": float(np.mean(l_mse)), "MAE": float(np.mean(l_mae))}
    print(f"    ✅ MSE={result['MSE']:.6f} | MAE={result['MAE']:.6f}")
    return result


# ============================================================
# 8. 频率模块微调（C / D 共用）
# ============================================================

def finetune_freq_module(encoder_fn, freq_module, train_loader,
                         val_loader, cfg: Config, tag: str):
    print(f"\n  [频率模块微调] {tag}")
    freq_module = freq_module.to(cfg.device)
    print("    计算频率标签...")
    train_freqs = []
    for context_patches, _ in train_loader:
        last_patch = context_patches[:, -1, :].numpy()
        for i in range(len(last_patch)):
            train_freqs.append(compute_dominant_freq(last_patch[i]))
    freq_arr  = np.array(train_freqs)
    freq_mean = float(freq_arr.mean())
    freq_std  = float(freq_arr.std()) + 1e-8
    print(f"    频率统计: mean={freq_mean:.4f}, std={freq_std:.4f}")

    def get_freq_label(context_patches_np):
        last   = context_patches_np[:, -1, :]
        labels = np.array([compute_dominant_freq(last[i])
                           for i in range(len(last))])
        return torch.tensor((labels - freq_mean) / freq_std, dtype=torch.float32)

    opt = torch.optim.AdamW(freq_module.parameters(),
                            lr=cfg.finetune_lr, weight_decay=1e-4)
    sch = warmup_cosine_lr(opt, cfg.finetune_warmup, cfg.finetune_epochs)
    best, pat, best_st = float('inf'), 0, None
    for ep in range(1, cfg.finetune_epochs + 1):
        freq_module.train()
        for context_patches, _ in train_loader:
            fl = get_freq_label(context_patches.numpy()).to(cfg.device)
            with torch.no_grad():
                z = encoder_fn(context_patches.to(cfg.device))
            loss = F.mse_loss(freq_module(z), fl)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(freq_module.parameters(), 1.0)
            opt.step()
        sch.step()
        freq_module.eval()
        vl = 0.0
        with torch.no_grad():
            for context_patches, _ in val_loader:
                fl = get_freq_label(context_patches.numpy()).to(cfg.device)
                z  = encoder_fn(context_patches.to(cfg.device))
                vl += F.mse_loss(freq_module(z), fl).item()
        vl /= max(1, len(val_loader))
        if ep % 30 == 0 or ep <= 3:
            print(f"    ep {ep:3d} | val_freq_mse={vl:.6f}")
        if vl < best - cfg.finetune_min_delta:
            best, pat, best_st = vl, 0, copy.deepcopy(freq_module.state_dict())
        else:
            pat += 1
        if pat >= cfg.finetune_patience:
            print(f"    [早停] ep={ep} | best={best:.6f}")
            break
    if best_st:
        freq_module.load_state_dict(best_st)
    freq_module.eval()
    return freq_module


# ============================================================
# 9. Cayley 变换（E / F / G / E2 / G2 共用）
# ============================================================

def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    n      = A.shape[-1]
    I      = torch.eye(n, device=A.device, dtype=A.dtype)
    A_skew = (A - A.transpose(-2, -1)) / 2.0
    return torch.linalg.solve(I + A_skew, I - A_skew)


# ============================================================
# 10. E_KoopProp 核心模块（原版，不变）
# ============================================================

class ODOKoopmanPropagator(nn.Module):
    def __init__(self, spectral_dim: int,
                 spectral_radius_bound: float = 0.99,
                 lyapunov_lambda: float = 1e-3,
                 lyapunov_epsilon: float = 0.1):
        super().__init__()
        self.d   = spectral_dim
        self.rho = spectral_radius_bound
        self.lam = lyapunov_lambda
        self.eps = lyapunov_epsilon
        self.A_U       = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.A_V       = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.zeros(spectral_dim))

    def get_K(self) -> torch.Tensor:
        U     = cayley_transform(self.A_U)
        V     = cayley_transform(self.A_V)
        sigma = torch.clamp(F.softplus(self.log_sigma), max=self.rho)
        return U @ torch.diag(sigma) @ V.t()

    def forward(self, z_osc: torch.Tensor, steps: int = 1) -> torch.Tensor:
        K = self.get_K()
        if steps == 1:
            return z_osc @ K.t()
        return z_osc @ torch.linalg.matrix_power(K, steps).t()

    def lyapunov_loss(self) -> torch.Tensor:
        K = self.get_K()
        I = torch.eye(self.d, device=K.device, dtype=K.dtype)
        return self.lam * F.relu(
            torch.norm(K.t() @ K - I, p='fro') - self.eps)

    @torch.no_grad()
    def spectral_radius(self) -> float:
        return torch.linalg.eigvals(self.get_K()).abs().max().item()

    @torch.no_grad()
    def singular_values(self) -> torch.Tensor:
        return self.get_K().svd().S.cpu()


class EKoopPropEncoder(nn.Module):
    """E / F / E2 / G2 共用的 Encoder 包装类"""
    def __init__(self, official_enc, decoupler, propagator, pred_steps: int = 1):
        super().__init__()
        self.official_enc = official_enc
        self.decoupler    = decoupler
        self.propagator   = propagator
        self.pred_steps   = pred_steps
        for p in self.official_enc.parameters():
            p.requires_grad_(False)
        for p in self.decoupler.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z            = torch.sum(self.official_enc(x), dim=1)
            z_inv, z_osc = self.decoupler(z)
        z_osc_prop = self.propagator(z_osc, self.pred_steps)
        return torch.cat([z_inv, z_osc_prop], dim=-1)


def train_koopprop_head(ekoop_enc, decoder, train_loader, val_loader,
                        cfg: Config, tag: str = "E_KoopProp"):
    """E / F / E2 / G2 共用训练函数"""
    print(f"\n  ── 训练 {tag} ──")
    ekoop_enc = ekoop_enc.to(cfg.device)
    decoder   = decoder.to(cfg.device)
    trainable = (list(ekoop_enc.propagator.parameters()) +
                 list(decoder.parameters()))
    n_prop = sum(p.numel() for p in ekoop_enc.propagator.parameters())
    n_dec  = sum(p.numel() for p in decoder.parameters())
    print(f"    可训练参数: propagator={n_prop:,} | decoder={n_dec:,} | total={n_prop+n_dec:,}")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=1e-7)
    best, pat = float('inf'), 0
    best_prop_st = best_dec_st = None

    for ep in range(1, cfg.num_epochs + 1):
        ekoop_enc.train(); decoder.train()
        ep_pred = ep_lyap = 0.0; nb = 0
        for context_patches, target_patch in train_loader:
            context_patches = context_patches.to(cfg.device)
            target_patch    = target_patch.to(cfg.device)
            z      = ekoop_enc(context_patches)
            pred   = decoder(z)
            L_pred = F.mse_loss(pred, target_patch)
            L_lyap = ekoop_enc.propagator.lyapunov_loss()
            loss   = L_pred + L_lyap
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            ep_pred += L_pred.item(); ep_lyap += L_lyap.item(); nb += 1
        sch.step()

        ekoop_enc.eval(); decoder.eval()
        vl = 0.0
        with torch.no_grad():
            for context_patches, target_patch in val_loader:
                z    = ekoop_enc(context_patches.to(cfg.device))
                pred = decoder(z)
                vl  += F.mse_loss(pred, target_patch.to(cfg.device)).item()
        vl /= max(1, len(val_loader))

        if ep % 50 == 0 or ep <= 3:
            rho = ekoop_enc.propagator.spectral_radius()
            print(f"    ep {ep:3d} | pred={ep_pred/nb:.5f} | "
                  f"lyap={ep_lyap/nb:.5f} | val={vl:.5f} | ρ(K)={rho:.4f}")
            if hasattr(ekoop_enc.propagator, 'print_spectral_report'):
                ekoop_enc.propagator.print_spectral_report(prefix="      ")
            elif hasattr(ekoop_enc.propagator, 'print_rho_report'):
                ekoop_enc.propagator.print_rho_report(prefix="      ")

        if vl < best - cfg.min_delta:
            best, pat = vl, 0
            best_prop_st = copy.deepcopy(ekoop_enc.propagator.state_dict())
            best_dec_st  = copy.deepcopy(decoder.state_dict())
        else:
            pat += 1
        if pat >= cfg.patience:
            print(f"    [早停] ep={ep} | best_val_mse={best:.6f}")
            break

    if best_prop_st: ekoop_enc.propagator.load_state_dict(best_prop_st)
    if best_dec_st:  decoder.load_state_dict(best_dec_st)
    rho_f = ekoop_enc.propagator.spectral_radius()
    sv    = ekoop_enc.propagator.singular_values()
    print(f"    ✅ 训练完成 | best_val={best:.6f} | "
          f"ρ(K)={rho_f:.6f} | σ∈[{sv.min():.4f},{sv.max():.4f}]")
    ekoop_enc.eval(); decoder.eval()
    return ekoop_enc, decoder


# ============================================================
# 11. F_PerDimKoop 核心模块（原版，不变）
# ============================================================

class PerDimODOKoopmanPropagator(nn.Module):
    def __init__(self, spectral_dim: int,
                 rho_min: float = 0.10, rho_max: float = 0.99,
                 lyapunov_lambda: float = 1e-3,
                 lyapunov_epsilon: float = 0.1):
        super().__init__()
        self.d       = spectral_dim
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.lam     = lyapunov_lambda
        self.eps     = lyapunov_epsilon
        self.A_U       = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.A_V       = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.zeros(spectral_dim))
        self.a_rho     = nn.Parameter(torch.zeros(spectral_dim))

    def get_rho_per_dim(self) -> torch.Tensor:
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(self.a_rho)

    def get_K(self) -> torch.Tensor:
        U     = cayley_transform(self.A_U)
        V     = cayley_transform(self.A_V)
        sigma = torch.min(F.softplus(self.log_sigma), self.get_rho_per_dim())
        return U @ torch.diag(sigma) @ V.t()

    def forward(self, z_osc: torch.Tensor, steps: int = 1) -> torch.Tensor:
        K = self.get_K()
        if steps == 1:
            return z_osc @ K.t()
        return z_osc @ torch.linalg.matrix_power(K, steps).t()

    def lyapunov_loss(self) -> torch.Tensor:
        K = self.get_K()
        I = torch.eye(self.d, device=K.device, dtype=K.dtype)
        return self.lam * F.relu(
            torch.norm(K.t() @ K - I, p='fro') - self.eps)

    @torch.no_grad()
    def spectral_radius(self) -> float:
        return torch.linalg.eigvals(self.get_K()).abs().max().item()

    @torch.no_grad()
    def singular_values(self) -> torch.Tensor:
        return torch.min(F.softplus(self.log_sigma),
                         self.get_rho_per_dim()).detach().cpu()

    @torch.no_grad()
    def print_rho_report(self, prefix: str = ""):
        rho_i  = self.get_rho_per_dim().cpu()
        mid    = (self.rho_min + self.rho_max) / 2.0
        n_slow = int((rho_i >= mid).sum())
        n_fast = int((rho_i <  mid).sum())
        print(f"{prefix}+-- PerDim ρ 报告 ------------------------------------")
        print(f"{prefix}|  ρ(K)={self.spectral_radius():.4f} | "
              f"ρ_i ∈ [{rho_i.min():.4f}, {rho_i.max():.4f}] | std={rho_i.std():.4f}")
        print(f"{prefix}|  慢变维度(ρ≥{mid:.2f}): {n_slow:2d}/16 | "
              f"快变维度(ρ<{mid:.2f}): {n_fast:2d}/16")
        print(f"{prefix}+------------------------------------------------------")


# ============================================================
# 12. G_JointKoop 核心模块（原版，不变）
# ============================================================

class MultiFreqDecoupledProjector(nn.Module):
    def __init__(self, embed_dim: int, dim_low: int, dim_mid: int, dim_high: int):
        super().__init__()
        inv_dim = embed_dim - dim_low - dim_mid - dim_high
        self.proj_inv  = nn.Linear(embed_dim, inv_dim)
        self.proj_low  = nn.Linear(embed_dim, dim_low)
        self.proj_mid  = nn.Linear(embed_dim, dim_mid)
        self.proj_high = nn.Linear(embed_dim, dim_high)
    def forward(self, z):
        return (self.proj_inv(z), self.proj_low(z),
                self.proj_mid(z), self.proj_high(z))


class MultiFreqSpectralHead(nn.Module):
    def __init__(self, dim_low: int, dim_mid: int, dim_high: int):
        super().__init__()
        self.head_low  = nn.Sequential(
            nn.Linear(dim_low,  32), nn.GELU(), nn.Linear(32, 1))
        self.head_mid  = nn.Sequential(
            nn.Linear(dim_mid,  32), nn.GELU(), nn.Linear(32, 1))
        self.head_high = nn.Sequential(
            nn.Linear(dim_high, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, z_low, z_mid, z_high):
        return (self.head_low(z_low).squeeze(-1),
                self.head_mid(z_mid).squeeze(-1),
                self.head_high(z_high).squeeze(-1))


def compute_band_energy(patch_np: np.ndarray,
                        freq_lo: float, freq_hi: float) -> tuple:
    n     = len(patch_np)
    spec  = np.abs(fft(patch_np)) ** 2
    freqs = fftfreq(n)
    pos_mask  = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_spec  = spec[pos_mask]
    total     = pos_spec.sum() + 1e-8
    low_e  = float(pos_spec[pos_freqs <  freq_lo].sum() / total)
    mid_e  = float(pos_spec[(pos_freqs >= freq_lo) &
                             (pos_freqs <  freq_hi)].sum() / total)
    high_e = float(pos_spec[pos_freqs >= freq_hi].sum() / total)
    e_min = min(low_e, mid_e, high_e)
    e_max = max(low_e, mid_e, high_e)
    span  = e_max - e_min + 1e-8
    return ((low_e - e_min) / span,
            (mid_e - e_min) / span,
            (high_e - e_min) / span)


class SingleBandKoopman(nn.Module):
    def __init__(self, dim: int, lyap_lambda=1e-3, lyap_epsilon=0.1,
                 spec_lambda=1e-2, rho_soft=0.99,
                 rank_lambda=1e-3, rank_margin=0.05):
        super().__init__()
        self.d = dim
        self.lyap_lambda = lyap_lambda; self.lyap_eps = lyap_epsilon
        self.spec_lambda = spec_lambda; self.rho_soft = rho_soft
        self.rank_lambda = rank_lambda; self.rank_margin = rank_margin
        self.A_U = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.A_V = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.log_sigma = nn.Parameter(
            torch.linspace(math.log(0.1), math.log(0.9), dim))

    def get_sigma(self): return F.softplus(self.log_sigma)

    def get_K(self):
        U = cayley_transform(self.A_U); V = cayley_transform(self.A_V)
        return U @ torch.diag(self.get_sigma()) @ V.t()

    def forward(self, z, steps=1):
        K = self.get_K()
        if steps == 1: return z @ K.t()
        return z @ torch.linalg.matrix_power(K, steps).t()

    def compute_loss(self):
        K = self.get_K()
        I = torch.eye(self.d, device=K.device, dtype=K.dtype)
        sigma = self.get_sigma()
        L_lyap = self.lyap_lambda * F.relu(
            torch.norm(K.t() @ K - I, p='fro') - self.lyap_eps)
        L_spec = self.spec_lambda * F.softplus(sigma - self.rho_soft).sum()
        sigma_s, _ = torch.sort(sigma, descending=True)
        L_rank = (self.rank_lambda * F.relu(
            self.rank_margin - (sigma_s[:-1] - sigma_s[1:])).sum()
                  if self.d > 1 else torch.tensor(0., device=K.device))
        return L_lyap + L_spec + L_rank

    @torch.no_grad()
    def spectral_radius(self):
        return torch.linalg.eigvals(self.get_K()).abs().max().item()

    @torch.no_grad()
    def sigma_stats(self):
        s = self.get_sigma().cpu()
        return {'min': float(s.min()), 'max': float(s.max()),
                'mean': float(s.mean()), 'std': float(s.std()),
                'rho': self.spectral_radius()}


class GroupedKoopmanPropagator(nn.Module):
    def __init__(self, dim_low, dim_mid, dim_high,
                 lyap_lambda=1e-3, lyap_epsilon=0.1,
                 spec_lambda=1e-2, rho_soft=0.99,
                 rank_lambda=1e-3, rank_margin=0.05,
                 cross_rank_lambda=1e-2, cross_rank_margin=0.02):
        super().__init__()
        shared = dict(lyap_lambda=lyap_lambda, lyap_epsilon=lyap_epsilon,
                      spec_lambda=spec_lambda, rho_soft=rho_soft,
                      rank_lambda=rank_lambda, rank_margin=rank_margin)
        self.K_low  = SingleBandKoopman(dim_low,  **shared)
        self.K_mid  = SingleBandKoopman(dim_mid,  **shared)
        self.K_high = SingleBandKoopman(dim_high, **shared)
        self.cross_rank_lambda = cross_rank_lambda
        self.cross_rank_margin = cross_rank_margin

    def forward(self, z_low, z_mid, z_high, steps=1):
        return (self.K_low(z_low, steps),
                self.K_mid(z_mid, steps),
                self.K_high(z_high, steps))

    def cross_band_rank_loss(self):
        rho_low  = self.K_low.get_sigma().mean()
        rho_mid  = self.K_mid.get_sigma().mean()
        rho_high = self.K_high.get_sigma().mean()
        return self.cross_rank_lambda * (
            F.relu(rho_mid  - rho_low  + self.cross_rank_margin) +
            F.relu(rho_high - rho_mid  + self.cross_rank_margin))

    def total_loss(self):
        return (self.K_low.compute_loss() + self.K_mid.compute_loss() +
                self.K_high.compute_loss() + self.cross_band_rank_loss())

    def lyapunov_loss(self): return self.total_loss()

    @torch.no_grad()
    def spectral_radius(self):
        return max(self.K_low.spectral_radius(),
                   self.K_mid.spectral_radius(),
                   self.K_high.spectral_radius())

    @torch.no_grad()
    def singular_values(self):
        return torch.cat([self.K_low.get_sigma().cpu(),
                          self.K_mid.get_sigma().cpu(),
                          self.K_high.get_sigma().cpu()])

    @torch.no_grad()
    def print_band_report(self, prefix=""):
        sl = self.K_low.sigma_stats(); sm = self.K_mid.sigma_stats()
        sh = self.K_high.sigma_stats()
        pl = float(self.K_low.get_sigma().mean())
        pm = float(self.K_mid.get_sigma().mean())
        ph = float(self.K_high.get_sigma().mean())
        ok = "✓" if pl > pm > ph else "✗"
        print(f"{prefix}+-- 分组 Koopman 谱半径报告 ---------------------------")
        print(f"{prefix}|  K_low (dim={self.K_low.d:2d}) rho={sl['rho']:.4f} "
              f"proxy={pl:.4f} σ=[{sl['min']:.3f},{sl['max']:.3f}]")
        print(f"{prefix}|  K_mid (dim={self.K_mid.d:2d}) rho={sm['rho']:.4f} "
              f"proxy={pm:.4f} σ=[{sm['min']:.3f},{sm['max']:.3f}]")
        print(f"{prefix}|  K_high(dim={self.K_high.d:2d}) rho={sh['rho']:.4f} "
              f"proxy={ph:.4f} σ=[{sh['min']:.3f},{sh['max']:.3f}]")
        print(f"{prefix}|  涌现分化 {ok}: low({pl:.4f})>mid({pm:.4f})>high({ph:.4f})")
        print(f"{prefix}+------------------------------------------------------")


class JointKoopmanEncoder(nn.Module):
    def __init__(self, official_enc, multi_freq_dec,
                 propagator: GroupedKoopmanPropagator, pred_steps=1):
        super().__init__()
        self.official_enc = official_enc; self.multi_freq_dec = multi_freq_dec
        self.propagator = propagator; self.pred_steps = pred_steps
        for p in self.official_enc.parameters(): p.requires_grad_(False)

    def freeze_dec(self):
        for p in self.multi_freq_dec.parameters(): p.requires_grad_(False)
        self.multi_freq_dec.eval()

    def forward(self, x):
        with torch.no_grad():
            z = torch.sum(self.official_enc(x), dim=1)
            z_inv, z_low, z_mid, z_high = self.multi_freq_dec(z)
        z_lp, z_mp, z_hp = self.propagator(z_low, z_mid, z_high, self.pred_steps)
        return torch.cat([z_inv, z_lp, z_mp, z_hp], dim=-1)


def orthogonality_loss(z_low, z_mid, z_high):
    b = z_low.shape[0]
    zl = F.normalize(z_low, dim=-1); zm = F.normalize(z_mid, dim=-1)
    zh = F.normalize(z_high, dim=-1)
    return (torch.norm((zl.t() @ zm) / b, p='fro') +
            torch.norm((zl.t() @ zh) / b, p='fro') +
            torch.norm((zm.t() @ zh) / b, p='fro'))


def finetune_multifreq_module(encoder_fn, multi_freq_dec, spectral_heads,
                              train_loader, val_loader, cfg: Config):
    print(f"\n  [多频带微调] G_JointKoop")
    multi_freq_dec = multi_freq_dec.to(cfg.device)
    spectral_heads = spectral_heads.to(cfg.device)
    print("    预计算频带能量标签（对比度拉伸）...")
    all_lo, all_mi, all_hi = [], [], []
    for cp, _ in train_loader:
        last = cp[:, -1, :].numpy()
        for i in range(len(last)):
            lo, mi, hi = compute_band_energy(
                last[i], cfg.joint_freq_lo, cfg.joint_freq_hi)
            all_lo.append(lo); all_mi.append(mi); all_hi.append(hi)
    for name, arr in [("低频", all_lo), ("中频", all_mi), ("高频", all_hi)]:
        print(f"    {name}: mean={np.mean(arr):.4f} "
              f"range=[{np.min(arr):.4f},{np.max(arr):.4f}]")

    def get_band_labels(cp_np):
        last = cp_np[:, -1, :]
        lo_l, mi_l, hi_l = [], [], []
        for i in range(len(last)):
            lo, mi, hi = compute_band_energy(
                last[i], cfg.joint_freq_lo, cfg.joint_freq_hi)
            lo_l.append(lo); mi_l.append(mi); hi_l.append(hi)
        return (torch.tensor(lo_l, dtype=torch.float32),
                torch.tensor(mi_l, dtype=torch.float32),
                torch.tensor(hi_l, dtype=torch.float32))

    trainable = (list(multi_freq_dec.parameters()) +
                 list(spectral_heads.parameters()))
    opt = torch.optim.AdamW(trainable, lr=cfg.joint_finetune_lr, weight_decay=1e-4)
    sch = warmup_cosine_lr(opt, cfg.joint_finetune_warmup, cfg.joint_finetune_epochs)
    best, pat, best_dec_st, best_head_st = float('inf'), 0, None, None

    for ep in range(1, cfg.joint_finetune_epochs + 1):
        multi_freq_dec.train(); spectral_heads.train()
        for cp, _ in train_loader:
            ll, lm, lh = get_band_labels(cp.numpy())
            ll = ll.to(cfg.device); lm = lm.to(cfg.device); lh = lh.to(cfg.device)
            with torch.no_grad():
                z = encoder_fn(cp.to(cfg.device))
            _, z_low, z_mid, z_high = multi_freq_dec(z)
            p_low, p_mid, p_high = spectral_heads(z_low, z_mid, z_high)
            L_band = (F.mse_loss(p_low, ll) + F.mse_loss(p_mid, lm) +
                      F.mse_loss(p_high, lh))
            L_orth = cfg.joint_orth_lambda * orthogonality_loss(z_low, z_mid, z_high)
            loss = L_band + L_orth
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        sch.step()
        multi_freq_dec.eval(); spectral_heads.eval()
        vl = 0.0
        with torch.no_grad():
            for cp, _ in val_loader:
                ll, lm, lh = get_band_labels(cp.numpy())
                ll = ll.to(cfg.device); lm = lm.to(cfg.device); lh = lh.to(cfg.device)
                z = encoder_fn(cp.to(cfg.device))
                _, z_low, z_mid, z_high = multi_freq_dec(z)
                p_low, p_mid, p_high = spectral_heads(z_low, z_mid, z_high)
                vl += (F.mse_loss(p_low, ll) + F.mse_loss(p_mid, lm) +
                       F.mse_loss(p_high, lh)).item()
        vl /= max(1, len(val_loader))
        if ep % 30 == 0 or ep <= 3:
            print(f"    ep {ep:3d} | val_band_mse={vl:.6f}")
        if vl < best - cfg.joint_finetune_min_delta:
            best, pat = vl, 0
            best_dec_st  = copy.deepcopy(multi_freq_dec.state_dict())
            best_head_st = copy.deepcopy(spectral_heads.state_dict())
        else:
            pat += 1
        if pat >= cfg.joint_finetune_patience:
            print(f"    [早停] ep={ep} | best={best:.6f}"); break
    if best_dec_st:  multi_freq_dec.load_state_dict(best_dec_st)
    if best_head_st: spectral_heads.load_state_dict(best_head_st)
    multi_freq_dec.eval(); spectral_heads.eval()
    print(f"    [OK] 多频带微调完成 | best_val={best:.6f}")
    return multi_freq_dec, spectral_heads


def train_joint_koopman_head(joint_enc, decoder, train_loader, val_loader, cfg):
    print(f"\n  ── 训练 G_JointKoop ──")
    joint_enc = joint_enc.to(cfg.device); decoder = decoder.to(cfg.device)
    joint_enc.freeze_dec()
    trainable = (list(joint_enc.propagator.parameters()) +
                 list(decoder.parameters()))
    n_prop = sum(p.numel() for p in joint_enc.propagator.parameters())
    n_dec  = sum(p.numel() for p in decoder.parameters())
    print(f"    可训练参数: propagator={n_prop:,} | decoder={n_dec:,}")
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=1e-7)
    best, pat = float('inf'), 0; best_prop_st = best_dec_st = None

    for ep in range(1, cfg.num_epochs + 1):
        joint_enc.propagator.train(); decoder.train()
        ep_pred = ep_reg = 0.0; nb = 0
        for context_patches, target_patch in train_loader:
            context_patches = context_patches.to(cfg.device)
            target_patch    = target_patch.to(cfg.device)
            z = joint_enc(context_patches); pred = decoder(z)
            L_pred = F.mse_loss(pred, target_patch)
            L_reg  = joint_enc.propagator.total_loss()
            loss   = L_pred + L_reg
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            ep_pred += L_pred.item(); ep_reg += L_reg.item(); nb += 1
        sch.step()
        joint_enc.propagator.eval(); decoder.eval()
        vl = 0.0
        with torch.no_grad():
            for context_patches, target_patch in val_loader:
                z    = joint_enc(context_patches.to(cfg.device))
                pred = decoder(z)
                vl  += F.mse_loss(pred, target_patch.to(cfg.device)).item()
        vl /= max(1, len(val_loader))
        if ep % 50 == 0 or ep <= 3:
            print(f"    ep {ep:3d} | pred={ep_pred/nb:.5f} | "
                  f"reg={ep_reg/nb:.5f} | val={vl:.5f}")
            joint_enc.propagator.print_band_report(prefix="      ")
        if vl < best - cfg.min_delta:
            best, pat = vl, 0
            best_prop_st = copy.deepcopy(joint_enc.propagator.state_dict())
            best_dec_st  = copy.deepcopy(decoder.state_dict())
        else:
            pat += 1
        if pat >= cfg.patience:
            print(f"    [早停] ep={ep} | best_val_mse={best:.6f}"); break
    if best_prop_st: joint_enc.propagator.load_state_dict(best_prop_st)
    if best_dec_st:  decoder.load_state_dict(best_dec_st)
    sv = joint_enc.propagator.singular_values()
    print(f"    ✅ 训练完成 | best_val={best:.6f} | "
          f"rho_max={joint_enc.propagator.spectral_radius():.6f} | "
          f"σ∈[{sv.min():.4f},{sv.max():.4f}]")
    joint_enc.propagator.print_band_report(prefix="    ")
    joint_enc.eval(); decoder.eval()
    return joint_enc, decoder


# ============================================================
# 13. E2_SpectralE 核心模块  ★★ 新增
# ============================================================
# 核心思路：
#   E 的问题：所有奇异值共享同一上界，趋向相同值，无频率分化
#   E2 的解法：将 spectral_dim(16) 个奇异值按索引等分为 n_bands 段
#             对段间均值施加排序正则：
#             mean(σ[0:s]) > mean(σ[s:2s]) > mean(σ[2s:])
#   效果：Koopman 谱自然分化，无需硬切割子空间
#         K 的主奇异方向对应慢变（低频）动力学
#         K 的次奇异方向对应快变（高频）动力学
# ============================================================

class E2SpectralKoopmanPropagator(nn.Module):
    """
    E2 = E + 奇异值段排序正则

    与 E 的唯一区别：
      lyapunov_loss() 额外包含 spectral_sort_loss()
      forward() 与 E 完全相同，与 EKoopPropEncoder 完全兼容
    """
    def __init__(self, spectral_dim: int,
                 spectral_radius_bound: float = 0.99,
                 lyapunov_lambda: float  = 1e-3,
                 lyapunov_epsilon: float = 0.1,
                 sort_lambda: float      = 5e-2,
                 sort_margin: float      = 0.05,
                 n_bands: int            = 3):
        super().__init__()
        self.d           = spectral_dim
        self.rho         = spectral_radius_bound
        self.lam         = lyapunov_lambda
        self.eps         = lyapunov_epsilon
        self.sort_lambda = sort_lambda
        self.sort_margin = sort_margin
        self.n_bands     = n_bands

        self.A_U = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.A_V = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        # 分散初始化，有利于排序损失收敛
        log_sigma_init = torch.linspace(math.log(0.3), math.log(0.9), spectral_dim)
        self.log_sigma = nn.Parameter(log_sigma_init)

    def get_sigma(self) -> torch.Tensor:
        return torch.clamp(F.softplus(self.log_sigma), max=self.rho)

    def get_K(self) -> torch.Tensor:
        U = cayley_transform(self.A_U)
        V = cayley_transform(self.A_V)
        return U @ torch.diag(self.get_sigma()) @ V.t()

    def forward(self, z_osc: torch.Tensor, steps: int = 1) -> torch.Tensor:
        K = self.get_K()
        if steps == 1:
            return z_osc @ K.t()
        return z_osc @ torch.linalg.matrix_power(K, steps).t()

    def spectral_sort_loss(self) -> torch.Tensor:
        """
        将 spectral_dim 个奇异值等分为 n_bands 段，
        强制相邻段均值满足降序：σ_band[i].mean() > σ_band[i+1].mean()
        使用 softplus(margin - gap) 作为可微分惩罚。
        """
        sigma = self.get_sigma()
        # 等分为 n_bands 段（最后一段可能稍短）
        band_size = self.d // self.n_bands
        bands = []
        for i in range(self.n_bands):
            start = i * band_size
            end   = start + band_size if i < self.n_bands - 1 else self.d
            bands.append(sigma[start:end].mean())

        loss = torch.tensor(0.0, device=sigma.device)
        for i in range(len(bands) - 1):
            # 惩罚 bands[i+1] >= bands[i] - margin 的情况
            loss = loss + F.relu(bands[i+1] - bands[i] + self.sort_margin)
        return self.sort_lambda * loss

    def lyapunov_loss(self) -> torch.Tensor:
        """包含 Lyapunov 稳定性正则 + 奇异值排序正则"""
        K = self.get_K()
        I = torch.eye(self.d, device=K.device, dtype=K.dtype)
        L_lyap = self.lam * F.relu(
            torch.norm(K.t() @ K - I, p='fro') - self.eps)
        L_sort = self.spectral_sort_loss()
        return L_lyap + L_sort

    @torch.no_grad()
    def spectral_radius(self) -> float:
        return torch.linalg.eigvals(self.get_K()).abs().max().item()

    @torch.no_grad()
    def singular_values(self) -> torch.Tensor:
        return self.get_sigma().detach().cpu()

    @torch.no_grad()
    def print_spectral_report(self, prefix: str = ""):
        """打印奇异值分段报告，验证谱分化是否涌现"""
        sigma     = self.get_sigma().cpu()
        band_size = self.d // self.n_bands
        band_means = []
        for i in range(self.n_bands):
            start = i * band_size
            end   = start + band_size if i < self.n_bands - 1 else self.d
            band_means.append(float(sigma[start:end].mean()))

        ok = "✓" if all(band_means[i] > band_means[i+1]
                        for i in range(len(band_means)-1)) else "✗"
        labels = ["low", "mid", "high"]
        print(f"{prefix}+-- E2 谱分化报告 ------------------------------------")
        print(f"{prefix}|  ρ(K)={self.spectral_radius():.4f} | "
              f"σ∈[{float(sigma.min()):.4f},{float(sigma.max()):.4f}] | "
              f"std={float(sigma.std()):.4f}")
        band_str = " > ".join(
            f"{labels[i]}({band_means[i]:.4f})" for i in range(self.n_bands))
        print(f"{prefix}|  谱分化 {ok}: {band_str}")
        print(f"{prefix}+------------------------------------------------------")


# ============================================================
# 14. G2_SpectralKoop 核心模块  ★★★ 新增
# ============================================================
# 核心思路：
#   G 的问题：硬切割子空间（6+6+4），每组维度不足，
#             解耦器与 Koopman 梯度不对齐，三个 K 退化为相同谱
#
#   G2 的解法：不切割子空间，改用：
#     (1) 全维度单一 Koopman（16×16，继承 E 的信息容量）
#     (2) 奇异值排序正则（继承 E2 的谱分化机制）
#     (3) 频率对齐损失：让 K 的右奇异向量 V_i 在 z_osc 上
#         的投影具有与奇异值大小匹配的频率特性
#         大奇异值方向 → 低频率重心（慢变）
#         小奇异值方向 → 高频率重心（快变）
#     (4) 复用 D 的解耦器（冻结），与 E/E2 完全公平对比
#
#   频率对齐损失的实现：
#     对 batch 中的 z_osc (B, d)，计算每个右奇异方向的投影：
#       proj_i = z_osc @ V_i  →  (B,)
#     对 proj_i 做 FFT，计算频率重心（spectral centroid）：
#       centroid_i = Σ(k * |FFT(proj_i)|) / Σ(|FFT(proj_i)|)
#     期望：σ_i 越大，centroid_i 越小（低频）
#     损失：Σ_i σ_i * centroid_i  （最小化 → 大奇异值对应低频）
# ============================================================

class G2SpectralKoopmanPropagator(nn.Module):
    """
    G2 = 全维度 Koopman + 奇异值排序正则 + 频率对齐损失

    与 E 的区别：
      (1) 奇异值排序正则（同 E2）
      (2) 频率对齐损失（新增，需要 z_osc batch 参与计算）

    与 G 的区别：
      (1) 不切割子空间，保留全维度 16×16 K
      (2) 频率语义通过软约束引入，而非硬分组

    使用方式：
      训练时调用 freq_align_loss(z_osc) 加入总损失
      forward() 与 E 完全相同
    """
    def __init__(self, spectral_dim: int,
                 spectral_radius_bound: float = 0.99,
                 lyapunov_lambda: float    = 1e-3,
                 lyapunov_epsilon: float   = 0.1,
                 sort_lambda: float        = 5e-2,
                 sort_margin: float        = 0.05,
                 freq_align_lambda: float  = 2e-2,
                 n_bands: int              = 3):
        super().__init__()
        self.d                = spectral_dim
        self.rho              = spectral_radius_bound
        self.lam              = lyapunov_lambda
        self.eps              = lyapunov_epsilon
        self.sort_lambda      = sort_lambda
        self.sort_margin      = sort_margin
        self.freq_align_lambda = freq_align_lambda
        self.n_bands          = n_bands

        self.A_U = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.A_V = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        # 分散初始化，有利于排序损失收敛
        log_sigma_init = torch.linspace(math.log(0.3), math.log(0.9), spectral_dim)
        self.log_sigma = nn.Parameter(log_sigma_init)

    def get_sigma(self) -> torch.Tensor:
        return torch.clamp(F.softplus(self.log_sigma), max=self.rho)

    def get_K(self) -> torch.Tensor:
        U = cayley_transform(self.A_U)
        V = cayley_transform(self.A_V)
        return U @ torch.diag(self.get_sigma()) @ V.t()

    def forward(self, z_osc: torch.Tensor, steps: int = 1) -> torch.Tensor:
        K = self.get_K()
        if steps == 1:
            return z_osc @ K.t()
        return z_osc @ torch.linalg.matrix_power(K, steps).t()

    def spectral_sort_loss(self) -> torch.Tensor:
        """奇异值段排序正则（同 E2）"""
        sigma     = self.get_sigma()
        band_size = self.d // self.n_bands
        bands = []
        for i in range(self.n_bands):
            start = i * band_size
            end   = start + band_size if i < self.n_bands - 1 else self.d
            bands.append(sigma[start:end].mean())
        loss = torch.tensor(0.0, device=sigma.device)
        for i in range(len(bands) - 1):
            loss = loss + F.relu(bands[i+1] - bands[i] + self.sort_margin)
        return self.sort_lambda * loss

    def freq_align_loss(self, z_osc: torch.Tensor) -> torch.Tensor:
        """
        频率对齐损失：让大奇异值方向对应低频投影。

        步骤：
          1. 从 K = U·Σ·V^T 中取右奇异向量 V（即 cayley(A_V) 的列）
          2. 计算 z_osc 在每个右奇异方向上的投影 proj_i = z_osc @ V_i
          3. 对 proj_i 做 rfft，计算频率重心 centroid_i
          4. 损失 = Σ_i σ_i * centroid_i（最小化 → 大σ对应低频率重心）

        注意：频率重心归一化到 [0,1]，避免量纲问题。
        """
        V     = cayley_transform(self.A_V)   # (d, d)，列为右奇异向量
        sigma = self.get_sigma()              # (d,)

        # z_osc: (B, d) → 在每个右奇异方向上投影
        projs = z_osc @ V                     # (B, d)

        # 对每个方向计算频率重心
        # rfft 沿最后一维，但 projs 是 (B, d)，d=16 较小
        # 将 batch 维度视为"时间"，对 B 个样本的投影做 rfft
        # 即：把 batch 内的投影序列当作一段时序来分析其频率特性
        projs_t  = projs.t()                          # (d, B)
        fft_mag  = torch.fft.rfft(projs_t, dim=-1).abs()  # (d, B//2+1)
        n_freq   = fft_mag.shape[-1]
        freq_idx = torch.arange(n_freq, device=z_osc.device, dtype=z_osc.dtype)

        # 频率重心：加权平均频率索引，归一化到 [0,1]
        centroid = (fft_mag * freq_idx).sum(dim=-1) / (
            fft_mag.sum(dim=-1) + 1e-8)               # (d,)
        centroid = centroid / (n_freq - 1 + 1e-8)     # 归一化到 [0,1]

        # 损失：大奇异值 × 高频率重心 → 最小化迫使大σ对应低频
        loss = (sigma * centroid).sum()
        return self.freq_align_lambda * loss

    def lyapunov_loss(self, z_osc: torch.Tensor = None) -> torch.Tensor:
        """
        总正则损失：
          L_lyap：Lyapunov 稳定性
          L_sort：奇异值排序（谱分化）
          L_freq：频率对齐（可选，需传入 z_osc）
        """
        K = self.get_K()
        I = torch.eye(self.d, device=K.device, dtype=K.dtype)
        L_lyap = self.lam * F.relu(
            torch.norm(K.t() @ K - I, p='fro') - self.eps)
        L_sort = self.spectral_sort_loss()
        L_freq = (self.freq_align_loss(z_osc)
                  if z_osc is not None else torch.tensor(0.0, device=K.device))
        return L_lyap + L_sort + L_freq

    @torch.no_grad()
    def spectral_radius(self) -> float:
        return torch.linalg.eigvals(self.get_K()).abs().max().item()

    @torch.no_grad()
    def singular_values(self) -> torch.Tensor:
        return self.get_sigma().detach().cpu()

    @torch.no_grad()
    def print_spectral_report(self, prefix: str = ""):
        sigma     = self.get_sigma().cpu()
        band_size = self.d // self.n_bands
        band_means = []
        for i in range(self.n_bands):
            start = i * band_size
            end   = start + band_size if i < self.n_bands - 1 else self.d
            band_means.append(float(sigma[start:end].mean()))
        ok = "✓" if all(band_means[i] > band_means[i+1]
                        for i in range(len(band_means)-1)) else "✗"
        labels = ["low", "mid", "high"]
        print(f"{prefix}+-- G2 谱分化报告 ------------------------------------")
        print(f"{prefix}|  ρ(K)={self.spectral_radius():.4f} | "
              f"σ∈[{float(sigma.min()):.4f},{float(sigma.max()):.4f}] | "
              f"std={float(sigma.std()):.4f}")
        band_str = " > ".join(
            f"{labels[i]}({band_means[i]:.4f})" for i in range(self.n_bands))
        print(f"{prefix}|  谱分化 {ok}: {band_str}")
        print(f"{prefix}+------------------------------------------------------")


# ── G2 专用 Encoder 包装（需在 forward 中暴露 z_osc 给损失函数）─────

class G2KoopEncoder(nn.Module):
    """
    G2 的 Encoder 包装。
    与 EKoopPropEncoder 的唯一区别：
      forward() 额外返回 z_osc（用于 freq_align_loss）
      提供 encode_with_zosc() 方法供训练循环调用。
    """
    def __init__(self, official_enc, decoupler,
                 propagator: G2SpectralKoopmanPropagator,
                 pred_steps: int = 1):
        super().__init__()
        self.official_enc = official_enc
        self.decoupler    = decoupler
        self.propagator   = propagator
        self.pred_steps   = pred_steps
        for p in self.official_enc.parameters():
            p.requires_grad_(False)
        for p in self.decoupler.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准 forward，供 rolling_test 使用"""
        z, _ = self.encode_with_zosc(x)
        return z

    def encode_with_zosc(self, x: torch.Tensor):
        """
        返回 (z_full, z_osc)，供训练循环计算 freq_align_loss
        """
        with torch.no_grad():
            z_raw        = torch.sum(self.official_enc(x), dim=1)
            z_inv, z_osc = self.decoupler(z_raw)
        z_osc_prop = self.propagator(z_osc, self.pred_steps)
        z_full     = torch.cat([z_inv, z_osc_prop], dim=-1)
        return z_full, z_osc  # z_osc 未经传播，用于频率分析


def train_g2_koopman_head(g2_enc: G2KoopEncoder, decoder,
                           train_loader, val_loader,
                           cfg: Config, tag: str = "G2_SpectralKoop"):
    """
    G2 专用训练函数。
    与 train_koopprop_head 的区别：
      损失中额外包含 freq_align_loss(z_osc)
      需要调用 encode_with_zosc() 获取 z_osc
    """
    print(f"\n  ── 训练 {tag} ──")
    g2_enc  = g2_enc.to(cfg.device)
    decoder = decoder.to(cfg.device)

    trainable = (list(g2_enc.propagator.parameters()) +
                 list(decoder.parameters()))
    n_prop = sum(p.numel() for p in g2_enc.propagator.parameters())
    n_dec  = sum(p.numel() for p in decoder.parameters())
    print(f"    可训练参数: propagator={n_prop:,} | decoder={n_dec:,} | "
          f"total={n_prop+n_dec:,}")
    print(f"    sort_λ={cfg.g2_sort_lambda} | "
          f"freq_align_λ={cfg.g2_freq_align_lambda} | "
          f"n_bands={cfg.g2_n_bands}")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=1e-7)
    best, pat = float('inf'), 0
    best_prop_st = best_dec_st = None

    for ep in range(1, cfg.num_epochs + 1):
        g2_enc.train(); decoder.train()
        ep_pred = ep_lyap = ep_freq = 0.0; nb = 0

        for context_patches, target_patch in train_loader:
            context_patches = context_patches.to(cfg.device)
            target_patch    = target_patch.to(cfg.device)

            z_full, z_osc = g2_enc.encode_with_zosc(context_patches)
            pred   = decoder(z_full)
            L_pred = F.mse_loss(pred, target_patch)

            # lyapunov_loss 包含 Lyapunov + sort；freq_align 单独计算便于监控
            K      = g2_enc.propagator.get_K()
            I      = torch.eye(g2_enc.propagator.d,
                               device=K.device, dtype=K.dtype)
            L_lyap = g2_enc.propagator.lam * F.relu(
                torch.norm(K.t() @ K - I, p='fro') - g2_enc.propagator.eps)
            L_sort = g2_enc.propagator.spectral_sort_loss()
            L_freq = g2_enc.propagator.freq_align_loss(z_osc)
            loss   = L_pred + L_lyap + L_sort + L_freq

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            ep_pred += L_pred.item()
            ep_lyap += (L_lyap + L_sort).item()
            ep_freq += L_freq.item()
            nb += 1
        sch.step()

        g2_enc.eval(); decoder.eval()
        vl = 0.0
        with torch.no_grad():
            for context_patches, target_patch in val_loader:
                z_full, _ = g2_enc.encode_with_zosc(
                    context_patches.to(cfg.device))
                pred = decoder(z_full)
                vl  += F.mse_loss(pred, target_patch.to(cfg.device)).item()
        vl /= max(1, len(val_loader))

        if ep % 50 == 0 or ep <= 3:
            print(f"    ep {ep:3d} | pred={ep_pred/nb:.5f} | "
                  f"lyap+sort={ep_lyap/nb:.5f} | "
                  f"freq_align={ep_freq/nb:.5f} | val={vl:.5f}")
            g2_enc.propagator.print_spectral_report(prefix="      ")

        if vl < best - cfg.min_delta:
            best, pat = vl, 0
            best_prop_st = copy.deepcopy(g2_enc.propagator.state_dict())
            best_dec_st  = copy.deepcopy(decoder.state_dict())
        else:
            pat += 1
        if pat >= cfg.patience:
            print(f"    [早停] ep={ep} | best_val_mse={best:.6f}")
            break

    if best_prop_st: g2_enc.propagator.load_state_dict(best_prop_st)
    if best_dec_st:  decoder.load_state_dict(best_dec_st)

    sv = g2_enc.propagator.singular_values()
    print(f"    ✅ 训练完成 | best_val={best:.6f} | "
          f"ρ(K)={g2_enc.propagator.spectral_radius():.6f} | "
          f"σ∈[{sv.min():.4f},{sv.max():.4f}]")
    g2_enc.propagator.print_spectral_report(prefix="    ")
    g2_enc.eval(); decoder.eval()
    return g2_enc, decoder


# ============================================================
# 15. 主流程（A → B → C → D → E → F → G → E2 → G2）
# ============================================================

def run_all():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print("=" * 70)
    print("TS-JEPA — 消融链 A→B→C→D→E→F→G→E2→G2")
    print(f"device={cfg.device} | embed_dim={cfg.embed_dim} | "
          f"spectral_dim={cfg.spectral_dim} | lr={cfg.lr}")
    print(f"[E2] sort_λ={cfg.e2_sort_lambda} margin={cfg.e2_sort_margin} "
          f"n_bands={cfg.e2_n_bands}")
    print(f"[G2] sort_λ={cfg.g2_sort_lambda} "
          f"freq_align_λ={cfg.g2_freq_align_lambda} "
          f"n_bands={cfg.g2_n_bands}")
    print("=" * 70)

    # ── 数据 & 官方 Encoder ───────────────────────────────
    train_loader, val_loader, test_df = load_official_data(cfg)
    official_enc = load_official_encoder(cfg)
    for p in official_enc.parameters():
        p.requires_grad = False
    official_enc.eval()

    def base_enc_fn(x):
        return torch.sum(official_enc(x.to(cfg.device)), dim=1)

    results = {}

    # ══════════════════════════════════════════════════════
    # A_Scratch
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【A_Scratch】随机初始化 encoder（下界）")
    scratch_enc = OfficialEncoder(
        num_patches=cfg.num_patches, dim_in=cfg.context,
        kernel_size=cfg.kernel_size, embed_dim=cfg.embed_dim,
        embed_bias=True, nhead=cfg.encoder_nhead,
        num_layers=cfg.encoder_layers, jepa=True,
    ).to(cfg.device)
    for p in scratch_enc.parameters(): p.requires_grad = False
    scratch_enc.eval()
    def scratch_enc_fn(x):
        return torch.sum(scratch_enc(x.to(cfg.device)), dim=1)
    dec_A = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_A = train_forecast_head(scratch_enc_fn, dec_A,
                                train_loader, val_loader, cfg, "A_Scratch")
    results["A_Scratch"] = rolling_test(scratch_enc_fn, dec_A,
                                        test_df, cfg, "A_Scratch")

    # ══════════════════════════════════════════════════════
    # B_JEPA
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【B_JEPA】官方预训练 checkpoint（基准）")
    dec_B = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_B = train_forecast_head(base_enc_fn, dec_B,
                                train_loader, val_loader, cfg, "B_JEPA")
    results["B_JEPA"] = rolling_test(base_enc_fn, dec_B,
                                     test_df, cfg, "B_JEPA")

    # ══════════════════════════════════════════════════════
    # C_Direct
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【C_Direct】频率辅助头，不解耦（对照组）")
    sh_C = SpectralHead(cfg.embed_dim)
    finetune_freq_module(base_enc_fn, sh_C,
                         train_loader, val_loader, cfg, "C_Direct")
    dec_C = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_C = train_forecast_head(base_enc_fn, dec_C,
                                train_loader, val_loader, cfg, "C_Direct")
    results["C_Direct"] = rolling_test(base_enc_fn, dec_C,
                                       test_df, cfg, "C_Direct")

    # ══════════════════════════════════════════════════════
    # D_Decoupled（E / F / E2 / G2 均复用此解耦器）
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【D_Decoupled】频率解耦 latent（Step3 核心）")
    print("  注：E / F / E2 / G2 将复用此解耦器（冻结），保证消融对比公平")
    decoupler = DecoupledProjector(cfg.embed_dim, cfg.spectral_dim).to(cfg.device)
    sh_D      = SpectralHead(cfg.spectral_dim).to(cfg.device)

    class _FreqModuleD(nn.Module):
        def __init__(self, dec, sh):
            super().__init__(); self.dec = dec; self.sh = sh
        def forward(self, z):
            _, z_spec = self.dec(z); return self.sh(z_spec)

    freq_mod_D = finetune_freq_module(
        base_enc_fn, _FreqModuleD(decoupler, sh_D),
        train_loader, val_loader, cfg, "D_Decoupled")
    freq_mod_D.eval()
    for p in freq_mod_D.parameters(): p.requires_grad = False

    def decoupled_enc_fn(x):
        z = base_enc_fn(x)
        z_inv, z_spec = freq_mod_D.dec(z)
        return torch.cat([z_inv, z_spec], dim=-1)

    dec_D = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_D = train_forecast_head(decoupled_enc_fn, dec_D,
                                train_loader, val_loader, cfg, "D_Decoupled")
    results["D_Decoupled"] = rolling_test(decoupled_enc_fn, dec_D,
                                          test_df, cfg, "D_Decoupled")

    # ══════════════════════════════════════════════════════
    # E_KoopProp
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【E_KoopProp】解耦 + ODO Koopman（统一谱半径）★")
    prop_E = ODOKoopmanPropagator(
        spectral_dim=cfg.spectral_dim,
        spectral_radius_bound=cfg.koop_spectral_radius,
        lyapunov_lambda=cfg.koop_lyap_lambda,
        lyapunov_epsilon=cfg.koop_lyap_epsilon,
    )
    ekoop_enc = EKoopPropEncoder(
        official_enc=official_enc, decoupler=freq_mod_D.dec,
        propagator=prop_E, pred_steps=cfg.koop_pred_steps)
    dec_E = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    ekoop_enc, dec_E = train_koopprop_head(
        ekoop_enc, dec_E, train_loader, val_loader, cfg, tag="E_KoopProp")
    def e_enc_fn(x): return ekoop_enc(x.to(cfg.device))
    results["E_KoopProp"] = rolling_test(e_enc_fn, dec_E,
                                         test_df, cfg, "E_KoopProp")

    # ══════════════════════════════════════════════════════
    # F_PerDimKoop
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【F_PerDimKoop】解耦 + ODO Koopman（每维度独立谱半径）★")
    prop_F = PerDimODOKoopmanPropagator(
        spectral_dim=cfg.spectral_dim,
        rho_min=cfg.perdim_rho_min, rho_max=cfg.perdim_rho_max,
        lyapunov_lambda=cfg.perdim_lyap_lambda,
        lyapunov_epsilon=cfg.perdim_lyap_epsilon,
    )
    fkoop_enc = EKoopPropEncoder(
        official_enc=official_enc, decoupler=freq_mod_D.dec,
        propagator=prop_F, pred_steps=cfg.koop_pred_steps)
    dec_F = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    fkoop_enc, dec_F = train_koopprop_head(
        fkoop_enc, dec_F, train_loader, val_loader, cfg, tag="F_PerDimKoop")
    prop_F.print_rho_report(prefix="    ")
    def f_enc_fn(x): return fkoop_enc(x.to(cfg.device))
    results["F_PerDimKoop"] = rolling_test(f_enc_fn, dec_F,
                                           test_df, cfg, "F_PerDimKoop")

    # ══════════════════════════════════════════════════════
    # G_JointKoop
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【G_JointKoop】多频带联合解耦 + 分组 Koopman ★★★")
    multi_freq_dec = MultiFreqDecoupledProjector(
        embed_dim=cfg.embed_dim, dim_low=cfg.joint_dim_low,
        dim_mid=cfg.joint_dim_mid, dim_high=cfg.joint_dim_high,
    ).to(cfg.device)
    multi_freq_heads = MultiFreqSpectralHead(
        dim_low=cfg.joint_dim_low, dim_mid=cfg.joint_dim_mid,
        dim_high=cfg.joint_dim_high).to(cfg.device)
    multi_freq_dec, multi_freq_heads = finetune_multifreq_module(
        base_enc_fn, multi_freq_dec, multi_freq_heads,
        train_loader, val_loader, cfg)
    for p in multi_freq_heads.parameters(): p.requires_grad = False
    grouped_prop = GroupedKoopmanPropagator(
        dim_low=cfg.joint_dim_low, dim_mid=cfg.joint_dim_mid,
        dim_high=cfg.joint_dim_high,
        lyap_lambda=cfg.joint_lyap_lambda, lyap_epsilon=cfg.joint_lyap_epsilon,
        spec_lambda=cfg.joint_spec_lambda, rho_soft=cfg.joint_rho_soft,
        rank_lambda=cfg.joint_rank_lambda, rank_margin=cfg.joint_rank_margin,
        cross_rank_lambda=cfg.cross_rank_lambda,
        cross_rank_margin=cfg.cross_rank_margin)
    joint_enc = JointKoopmanEncoder(
        official_enc=official_enc, multi_freq_dec=multi_freq_dec,
        propagator=grouped_prop, pred_steps=cfg.koop_pred_steps)
    dec_G = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    joint_enc, dec_G = train_joint_koopman_head(
        joint_enc, dec_G, train_loader, val_loader, cfg)
    def g_enc_fn(x): return joint_enc(x.to(cfg.device))
    results["G_JointKoop"] = rolling_test(g_enc_fn, dec_G,
                                          test_df, cfg, "G_JointKoop")

    # ══════════════════════════════════════════════════════
    # E2_SpectralE：E + 奇异值排序正则  ★★ 新增
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【E2_SpectralE】E + 奇异值排序正则（谱自然分化）★★")
    print(f"  复用 D 的 decoupler（冻结）| "
          f"sort_λ={cfg.e2_sort_lambda} | margin={cfg.e2_sort_margin} | "
          f"n_bands={cfg.e2_n_bands}")
    print(f"  设计：将 {cfg.spectral_dim} 个奇异值等分为 {cfg.e2_n_bands} 段，"
          f"强制段均值降序 → 谱自然分化，无需硬切割")

    prop_E2 = E2SpectralKoopmanPropagator(
        spectral_dim          = cfg.spectral_dim,
        spectral_radius_bound = cfg.e2_spectral_radius,
        lyapunov_lambda       = cfg.e2_lyap_lambda,
        lyapunov_epsilon      = cfg.e2_lyap_epsilon,
        sort_lambda           = cfg.e2_sort_lambda,
        sort_margin           = cfg.e2_sort_margin,
        n_bands               = cfg.e2_n_bands,
    )
    # 复用 EKoopPropEncoder，与 E 完全相同的包装
    e2koop_enc = EKoopPropEncoder(
        official_enc = official_enc,
        decoupler    = freq_mod_D.dec,   # 同一个 D 解耦器，冻结
        propagator   = prop_E2,
        pred_steps   = cfg.koop_pred_steps,
    )
    dec_E2 = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    e2koop_enc, dec_E2 = train_koopprop_head(
        e2koop_enc, dec_E2, train_loader, val_loader,
        cfg, tag="E2_SpectralE")
    print("  [E2 最终谱报告]")
    prop_E2.print_spectral_report(prefix="    ")

    def e2_enc_fn(x): return e2koop_enc(x.to(cfg.device))
    results["E2_SpectralE"] = rolling_test(e2_enc_fn, dec_E2,
                                           test_df, cfg, "E2_SpectralE")

    # ══════════════════════════════════════════════════════
    # G2_SpectralKoop：全维度 Koopman + 软频率语义约束  ★★★ 新增
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【G2_SpectralKoop】全维度 Koopman + 软频率语义约束 ★★★")
    print(f"  复用 D 的 decoupler（冻结）| "
          f"sort_λ={cfg.g2_sort_lambda} | "
          f"freq_align_λ={cfg.g2_freq_align_lambda} | "
          f"n_bands={cfg.g2_n_bands}")
    print(f"  设计：全维度 {cfg.spectral_dim}×{cfg.spectral_dim} K（继承E信息容量）")
    print(f"        + 奇异值排序正则（继承E2谱分化）")
    print(f"        + 频率对齐损失（大奇异值方向→低频，新增）")

    prop_G2 = G2SpectralKoopmanPropagator(
        spectral_dim          = cfg.spectral_dim,
        spectral_radius_bound = cfg.g2_spectral_radius,
        lyapunov_lambda       = cfg.g2_lyap_lambda,
        lyapunov_epsilon      = cfg.g2_lyap_epsilon,
        sort_lambda           = cfg.g2_sort_lambda,
        sort_margin           = cfg.g2_sort_margin,
        freq_align_lambda     = cfg.g2_freq_align_lambda,
        n_bands               = cfg.g2_n_bands,
    )
    g2_enc = G2KoopEncoder(
        official_enc = official_enc,
        decoupler    = freq_mod_D.dec,   # 同一个 D 解耦器，冻结
        propagator   = prop_G2,
        pred_steps   = cfg.koop_pred_steps,
    )
    dec_G2 = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    g2_enc, dec_G2 = train_g2_koopman_head(
        g2_enc, dec_G2, train_loader, val_loader, cfg, tag="G2_SpectralKoop")
    print("  [G2 最终谱报告]")
    prop_G2.print_spectral_report(prefix="    ")

    def g2_enc_fn(x): return g2_enc(x.to(cfg.device))
    results["G2_SpectralKoop"] = rolling_test(g2_enc_fn, dec_G2,
                                              test_df, cfg, "G2_SpectralKoop")

    # ══════════════════════════════════════════════════════
    # 完整消融对比表（A→B→C→D→E→F→G→E2→G2）
    # ══════════════════════════════════════════════════════
    bm = results["B_JEPA"]["MSE"]
    ba = results["B_JEPA"]["MAE"]

    components = {
        "A_Scratch"      : "无预训练",
        "B_JEPA"         : "JEPA预训练",
        "C_Direct"       : "+频率辅助头",
        "D_Decoupled"    : "+频率解耦",
        "E_KoopProp"     : "+Koopman(统一谱半径)",
        "F_PerDimKoop"   : "+Koopman(每维独立谱半径)",
        "G_JointKoop"    : "+多频带联合解耦+分组Koopman",
        "E2_SpectralE"   : "+Koopman(统一谱半径+排序正则) ★★",
        "G2_SpectralKoop": "+全维度Koopman+软频率语义约束  ★★★",
    }

    print("\n" + "=" * 90)
    print("📊  完整消融对比表（滚动测试）—— Weather OT 列")
    print("=" * 90)
    print(f"{'组别':<22}│{'MSE':>10}│{'MAE':>10}│"
          f"{'ΔMSE vs B':>12}│{'ΔMAE vs B':>12}│ 新增组件")
    print("─" * 90)
    for name, res in results.items():
        comp = components.get(name, "")
        if name == "B_JEPA":
            ds = da = "  (基准)"
        else:
            ds = f"{(res['MSE']-bm)/bm*100:+.2f}%"
            da = f"{(res['MAE']-ba)/ba*100:+.2f}%"
        print(f"{name:<22}│{res['MSE']:>10.6f}│{res['MAE']:>10.6f}│"
              f"{ds:>12}│{da:>12}│ {comp}")
    print("=" * 90)

    print("\n📌 消融结论：")
    print("  A > B      → JEPA 预训练有效                                    ✅")
    print("  B > C      → 频率信息对预测有益                                  ✅")
    print("  C > D      → 解耦优于直接融合（Step3 核心结论）                   ✅")
    print("  D > E      → Koopman 动力学传播提升（统一谱半径）                 ✅")
    print("  E > E2     → 奇异值排序正则驱动谱分化，进一步提升               ✅?")
    print("  E/E2 > F   → 统一/排序约束优于每维独立谱半径                    ✅?")
    print("  E/E2 > G   → 全维度 K 优于硬切割分组（信息容量更大）            ✅?")
    print("  E2 > G2    → 排序正则 vs 排序+频率对齐（G2 是否额外有效）       ✅?")
    print("  G2 = best  → 全维度 K + 软频率语义 = 最优（Step4 核心结论）     ✅?")
    print("\n  ✅? 表示待实验验证的预期结论")
    print("\n  理论对齐（Ruiz-Morales et al. AAAI 2026）：")
    print("  E2/G2 的奇异值排序对应 Koopman 谱的频率分层：")
    print("  大奇异值 ↔ 慢衰减模态 ↔ 低频动力学（接近不变子空间）")
    print("  小奇异值 ↔ 快衰减模态 ↔ 高频动力学（远离不变子空间）")

    return results


if __name__ == "__main__":
    run_all()