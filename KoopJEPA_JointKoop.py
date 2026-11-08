"""
============================================================
Step3+G — 多频带联合解耦 + 分组 Koopman 传播（修复版）
============================================================
修复清单：
  Bug1: Config 新增 joint_finetune_min_delta
  Bug2: JointKoopmanEncoder.forward 对 multi_freq_dec 加 no_grad；
        train_joint_koopman_head 中显式 multi_freq_dec.eval()
  Bug3: train_koopman_head / train_joint_koopman_head 结束前加 .eval()
  Bug4: joint_rho_soft 从 1.00 改为 0.99
  Warn1: compute_band_energy 对能量取 log1p 放大高频监督信号
  Warn2: finetune_multifreq_module 去掉独立归一化，直接用原始能量占比
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

    # ── 频率解耦参数（D/E/F 共用）──────────────────────────
    spectral_dim = 16

    # ── 频率模块微调参数（D 专用）──────────────────────────
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
    perdim_rho_max      = 1.00
    perdim_lyap_lambda  = 1e-3
    perdim_lyap_epsilon = 0.1

    # ── G_JointKoop 专属参数 ────────────────────────────────
    # 子空间维度：低频+中频+高频 = spectral_dim = 16
    joint_dim_low   = 6
    joint_dim_mid   = 6
    joint_dim_high  = 4

    # FFT 频带边界（归一化到 [0,1]，对应正频率部分）
    joint_freq_lo   = 0.33
    joint_freq_hi   = 0.67

    # Lyapunov 正则
    joint_lyap_lambda  = 1e-3
    joint_lyap_epsilon = 0.1

    # 软惩罚（替代 clamp，避免梯度死区）
    # [Bug4 修复] joint_rho_soft 从 1.00 改为 0.99，与 E 保持一致
    joint_spec_lambda  = 1e-2
    joint_rho_soft     = 0.99

    # 排序损失
    joint_rank_lambda  = 1e-3
    joint_rank_margin  = 0.05

    # 多频带微调参数（G 专用，与 D 的微调参数独立）
    # [Bug1 修复] 新增 joint_finetune_min_delta，不再复用 finetune_min_delta
    joint_finetune_lr        = 1e-4
    joint_finetune_epochs    = 200
    joint_finetune_patience  = 30
    joint_finetune_min_delta = 1e-6   # Bug1 修复：G 专属早停 delta
    joint_finetune_warmup    = 10

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
    print(f"  [OK] 加载成功 | epoch={ckpt.get('epoch','?')} | 参数量={n:,}")
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

    print(f"  train={len(train_df[:n_tr]):,} | val={len(val_df):,}")
    return train_loader, val_loader, dataset.test_df


# ============================================================
# 4. 基础频率工具（D/G 共用）
# ============================================================

class SpectralHead(nn.Module):
    """单频带预测头，D_Decoupled 使用"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)


class DecoupledProjector(nn.Module):
    """单频解耦投影，D/E/F 使用"""
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


def compute_band_energy(patch_np: np.ndarray,
                        freq_lo: float, freq_hi: float) -> tuple:
    """
    计算三个频带的 log1p 归一化能量占比。
    [Warn1 修复] 对能量取 log1p，放大高频带的监督信号强度。
    返回 (low_log_ratio, mid_log_ratio, high_log_ratio)，各自 in [0, log(2)]。
    """
    n     = len(patch_np)
    spec  = np.abs(fft(patch_np)) ** 2
    freqs = fftfreq(n)

    pos_mask  = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_spec  = spec[pos_mask]
    total     = pos_spec.sum() + 1e-8

    low_e  = pos_spec[pos_freqs <  freq_lo].sum() / total
    mid_e  = pos_spec[(pos_freqs >= freq_lo) & (pos_freqs < freq_hi)].sum() / total
    high_e = pos_spec[pos_freqs >= freq_hi].sum() / total

    # [Warn1 修复] log1p 变换，放大高频带（原始值极小）的梯度信号
    return (float(np.log1p(low_e)),
            float(np.log1p(mid_e)),
            float(np.log1p(high_e)))


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
# 6. 下游预测头训练（B / D 共用）
# ============================================================

def train_forecast_head(encoder_fn, decoder, train_loader, val_loader,
                        cfg: Config, tag: str):
    print(f"\n  -- 训练预测头: {tag} --")
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
    decoder.eval()   # Bug3 修复：确保返回时处于 eval 模式
    return decoder


# ============================================================
# 7. 官方滚动测试
# ============================================================

def rolling_test(encoder_fn, decoder, test_df, cfg: Config, tag: str) -> dict:
    print(f"\n  -- 滚动测试: {tag} --")
    context     = cfg.context
    num_patches = cfg.num_patches
    num_steps   = (len(test_df) - context * num_patches) // context
    decoder.eval()
    l_mse, l_mae = [], []

    for step in range(num_steps):
        ctx = (
            test_df[context*step : context*num_patches + context*step]
            .reshape(num_patches, context).unsqueeze(0).to(cfg.device)
        )
        tgt = test_df[
            context*num_patches + step*context :
            context*num_patches + (step+1)*context
        ]
        with torch.no_grad():
            pred = decoder(encoder_fn(ctx))
        l_mse.append(mse(pred.flatten().cpu().numpy(), tgt.numpy()))
        l_mae.append(mae(pred.flatten().cpu().numpy(), tgt.numpy()))

    result = {"MSE": float(np.mean(l_mse)), "MAE": float(np.mean(l_mae))}
    print(f"    [OK] MSE={result['MSE']:.6f} | MAE={result['MAE']:.6f}")
    return result


# ============================================================
# 8. 单频带微调（D 依赖，保留）
# ============================================================

def finetune_freq_module(encoder_fn, freq_module, train_loader,
                         val_loader, cfg: Config, tag: str):
    print(f"\n  [单频微调] {tag}")
    freq_module = freq_module.to(cfg.device)

    train_freqs = []
    for context_patches, _ in train_loader:
        last = context_patches[:, -1, :].numpy()
        for i in range(len(last)):
            train_freqs.append(compute_dominant_freq(last[i]))
    freq_arr  = np.array(train_freqs)
    freq_mean = float(freq_arr.mean())
    freq_std  = float(freq_arr.std()) + 1e-8

    def get_freq_label(cp_np):
        last   = cp_np[:, -1, :]
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
    freq_module.eval()   # Bug3 修复
    return freq_module


# ============================================================
# 9. Cayley 变换（E / F / G 共用）
# ============================================================

def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    n      = A.shape[-1]
    I      = torch.eye(n, device=A.device, dtype=A.dtype)
    A_skew = (A - A.transpose(-2, -1)) / 2.0
    return torch.linalg.solve(I + A_skew, I - A_skew)


# ============================================================
# 10. E_KoopProp：原版单一谱半径（保留用于消融对比）
# ============================================================

class ODOKoopmanPropagator(nn.Module):
    def __init__(self, spectral_dim, spectral_radius_bound=0.99,
                 lyapunov_lambda=1e-3, lyapunov_epsilon=0.1):
        super().__init__()
        self.d   = spectral_dim
        self.rho = spectral_radius_bound
        self.lam = lyapunov_lambda
        self.eps = lyapunov_epsilon
        self.A_U       = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.A_V       = nn.Parameter(torch.randn(spectral_dim, spectral_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.zeros(spectral_dim))

    def get_K(self):
        U     = cayley_transform(self.A_U)
        V     = cayley_transform(self.A_V)
        sigma = torch.clamp(F.softplus(self.log_sigma), max=self.rho)
        return U @ torch.diag(sigma) @ V.t()

    def forward(self, z_osc, steps=1):
        K = self.get_K()
        if steps == 1: return z_osc @ K.t()
        return z_osc @ torch.linalg.matrix_power(K, steps).t()

    def lyapunov_loss(self):
        K = self.get_K()
        I = torch.eye(self.d, device=K.device, dtype=K.dtype)
        return self.lam * F.relu(torch.norm(K.t() @ K - I, p='fro') - self.eps)

    @torch.no_grad()
    def spectral_radius(self):
        return torch.linalg.eigvals(self.get_K()).abs().max().item()

    @torch.no_grad()
    def singular_values(self):
        return self.get_K().svd().S.cpu()


# ============================================================
# 11. F_PerDimKoop：每维度独立可学习谱半径（保留用于消融对比）
# ============================================================

class PerDimODOKoopmanPropagator(nn.Module):
    def __init__(self, spectral_dim, rho_min=0.10, rho_max=1.00,
                 lyapunov_lambda=1e-3, lyapunov_epsilon=0.1):
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

    def get_rho_per_dim(self):
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(self.a_rho)

    def get_K(self):
        U     = cayley_transform(self.A_U)
        V     = cayley_transform(self.A_V)
        sigma = torch.min(F.softplus(self.log_sigma), self.get_rho_per_dim())
        return U @ torch.diag(sigma) @ V.t()

    def forward(self, z_osc, steps=1):
        K = self.get_K()
        if steps == 1: return z_osc @ K.t()
        return z_osc @ torch.linalg.matrix_power(K, steps).t()

    def lyapunov_loss(self):
        K = self.get_K()
        I = torch.eye(self.d, device=K.device, dtype=K.dtype)
        return self.lam * F.relu(torch.norm(K.t() @ K - I, p='fro') - self.eps)

    @torch.no_grad()
    def spectral_radius(self):
        return torch.linalg.eigvals(self.get_K()).abs().max().item()

    @torch.no_grad()
    def singular_values(self):
        return torch.min(F.softplus(self.log_sigma),
                         self.get_rho_per_dim()).detach().cpu()

    @torch.no_grad()
    def print_rho_report(self, prefix=""):
        rho_i = self.get_rho_per_dim().cpu()
        mid   = (self.rho_min + self.rho_max) / 2.0
        print(f"{prefix}+-- PerDim rho 报告 -----------------------------------")
        print(f"{prefix}|  rho(K)={self.spectral_radius():.4f} | "
              f"rho in [{rho_i.min():.4f}, {rho_i.max():.4f}] | "
              f"std={rho_i.std():.4f}")
        print(f"{prefix}|  衰减(rho<{mid:.2f}):{int((rho_i<mid).sum()):2d}/16 | "
              f"振荡(rho>={mid:.2f}):{int((rho_i>=mid).sum()):2d}/16")
        print(f"{prefix}+------------------------------------------------------")


# ============================================================
# 12. 通用 Koopman 编码器包装（E / F 共用）
# ============================================================

class KoopmanEncoder(nn.Module):
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
            z = torch.sum(self.official_enc(x), dim=1)
        with torch.no_grad():
            z_inv, z_osc = self.decoupler(z)
        z_osc_prop = self.propagator(z_osc, self.pred_steps)
        return torch.cat([z_inv, z_osc_prop], dim=-1)


# ============================================================
# 13. 通用 Koopman 训练函数（E / F 共用）
# ============================================================

def train_koopman_head(koop_enc, decoder, train_loader, val_loader,
                       cfg: Config, tag: str):
    print(f"\n  -- 训练 {tag} --")
    koop_enc = koop_enc.to(cfg.device)
    decoder  = decoder.to(cfg.device)
    trainable = (list(koop_enc.propagator.parameters()) +
                 list(decoder.parameters()))
    n_prop = sum(p.numel() for p in koop_enc.propagator.parameters())
    n_dec  = sum(p.numel() for p in decoder.parameters())
    print(f"    可训练参数: propagator={n_prop:,} | decoder={n_dec:,}")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=1e-7)
    best, pat = float('inf'), 0
    best_prop_st = best_dec_st = None

    for ep in range(1, cfg.num_epochs + 1):
        koop_enc.train(); decoder.train()
        ep_pred = ep_lyap = 0.0; nb = 0
        for context_patches, target_patch in train_loader:
            context_patches = context_patches.to(cfg.device)
            target_patch    = target_patch.to(cfg.device)
            z      = koop_enc(context_patches)
            pred   = decoder(z)
            L_pred = F.mse_loss(pred, target_patch)
            L_lyap = koop_enc.propagator.lyapunov_loss()
            loss   = L_pred + L_lyap
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            ep_pred += L_pred.item(); ep_lyap += L_lyap.item(); nb += 1
        sch.step()

        koop_enc.eval(); decoder.eval()
        vl = 0.0
        with torch.no_grad():
            for context_patches, target_patch in val_loader:
                z    = koop_enc(context_patches.to(cfg.device))
                pred = decoder(z)
                vl  += F.mse_loss(pred, target_patch.to(cfg.device)).item()
        vl /= max(1, len(val_loader))

        if ep % 50 == 0 or ep <= 3:
            rho = koop_enc.propagator.spectral_radius()
            print(f"    ep {ep:3d} | pred={ep_pred/nb:.5f} | "
                  f"lyap={ep_lyap/nb:.5f} | val={vl:.5f} | rho(K)={rho:.4f}")
            if hasattr(koop_enc.propagator, 'print_rho_report'):
                koop_enc.propagator.print_rho_report(prefix="      ")

        if vl < best - cfg.min_delta:
            best = vl; pat = 0
            best_prop_st = copy.deepcopy(koop_enc.propagator.state_dict())
            best_dec_st  = copy.deepcopy(decoder.state_dict())
        else:
            pat += 1
        if pat >= cfg.patience:
            print(f"    [早停] ep={ep} | best_val_mse={best:.6f}")
            break

    if best_prop_st: koop_enc.propagator.load_state_dict(best_prop_st)
    if best_dec_st:  decoder.load_state_dict(best_dec_st)
    sv = koop_enc.propagator.singular_values()
    print(f"    [OK] best_val={best:.6f} | "
          f"rho(K)={koop_enc.propagator.spectral_radius():.6f} | "
          f"sigma in [{sv.min():.4f}, {sv.max():.4f}]")
    # Bug3 修复：训练结束后显式切换到 eval 模式
    koop_enc.eval(); decoder.eval()
    return koop_enc, decoder


# ============================================================
# 14. G_JointKoop 核心模块
# ============================================================

# ── 14.1 多频带解耦投影 ──────────────────────────────────────

class MultiFreqDecoupledProjector(nn.Module):
    """
    多频带解耦投影：z(128) -> z_inv(112) + z_low(6) + z_mid(6) + z_high(4)
    各子空间由独立的频带能量监督驱动，使其在解耦阶段就具备频率语义。
    """
    def __init__(self, embed_dim: int,
                 dim_low: int, dim_mid: int, dim_high: int):
        super().__init__()
        self.dim_low  = dim_low
        self.dim_mid  = dim_mid
        self.dim_high = dim_high
        inv_dim = embed_dim - dim_low - dim_mid - dim_high

        self.proj_inv  = nn.Linear(embed_dim, inv_dim)
        self.proj_low  = nn.Linear(embed_dim, dim_low)
        self.proj_mid  = nn.Linear(embed_dim, dim_mid)
        self.proj_high = nn.Linear(embed_dim, dim_high)

    def forward(self, z):
        return (self.proj_inv(z),
                self.proj_low(z),
                self.proj_mid(z),
                self.proj_high(z))


# ── 14.2 多频带监督头 ────────────────────────────────────────

class MultiFreqSpectralHead(nn.Module):
    """
    三个独立的频带能量预测头。
    [Warn2 修复] 输出不加 Sigmoid，直接回归 log1p 变换后的能量值（无界正数）。
    与 compute_band_energy 的 log1p 输出对齐。
    """
    def __init__(self, dim_low: int, dim_mid: int, dim_high: int):
        super().__init__()
        # [Warn2 修复] 去掉末尾 Sigmoid，改为直接线性输出，与 log1p 标签对齐
        self.head_low  = nn.Sequential(
            nn.Linear(dim_low,  32), nn.GELU(), nn.Linear(32, 1)
        )
        self.head_mid  = nn.Sequential(
            nn.Linear(dim_mid,  32), nn.GELU(), nn.Linear(32, 1)
        )
        self.head_high = nn.Sequential(
            nn.Linear(dim_high, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, z_low, z_mid, z_high):
        return (self.head_low(z_low).squeeze(-1),
                self.head_mid(z_mid).squeeze(-1),
                self.head_high(z_high).squeeze(-1))


# ── 14.3 分组 Koopman 传播器（G 核心）──────────────────────

class SingleBandKoopman(nn.Module):
    """
    单频带 ODO Koopman 算子。
    无 clamp 硬约束，谱半径完全由频带监督涌现。
    [Bug4 修复] rho_soft 默认值改为 0.99。
    """
    def __init__(self, dim: int,
                 lyap_lambda: float    = 1e-3,
                 lyap_epsilon: float   = 0.1,
                 spec_lambda: float    = 1e-2,
                 rho_soft: float       = 0.99,   # Bug4 修复
                 rank_lambda: float    = 1e-3,
                 rank_margin: float    = 0.05,
                 sigma_init_min: float = 0.1,
                 sigma_init_max: float = 0.9):
        super().__init__()
        self.d           = dim
        self.lyap_lambda = lyap_lambda
        self.lyap_eps    = lyap_epsilon
        self.spec_lambda = spec_lambda
        self.rho_soft    = rho_soft
        self.rank_lambda = rank_lambda
        self.rank_margin = rank_margin

        self.A_U = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.A_V = nn.Parameter(torch.randn(dim, dim) * 0.01)

        # 对数均匀初始化：覆盖 [sigma_init_min, sigma_init_max]
        log_sigma_init = torch.linspace(
            math.log(sigma_init_min),
            math.log(sigma_init_max),
            dim
        )
        self.log_sigma = nn.Parameter(log_sigma_init)

    def get_sigma(self) -> torch.Tensor:
        return F.softplus(self.log_sigma)

    def get_K(self) -> torch.Tensor:
        U     = cayley_transform(self.A_U)
        V     = cayley_transform(self.A_V)
        sigma = self.get_sigma()
        return U @ torch.diag(sigma) @ V.t()

    def forward(self, z: torch.Tensor, steps: int = 1) -> torch.Tensor:
        K = self.get_K()
        if steps == 1: return z @ K.t()
        return z @ torch.linalg.matrix_power(K, steps).t()

    def compute_loss(self) -> torch.Tensor:
        K     = self.get_K()
        I     = torch.eye(self.d, device=K.device, dtype=K.dtype)
        sigma = self.get_sigma()

        # Lyapunov 损失
        L_lyap = self.lyap_lambda * F.relu(
            torch.norm(K.t() @ K - I, p='fro') - self.lyap_eps
        )
        # 软上界惩罚（softplus 替代 relu，梯度永不死）
        L_spec = self.spec_lambda * F.softplus(sigma - self.rho_soft).sum()

        # 排序损失：鼓励奇异值在子空间内分散
        sigma_sorted, _ = torch.sort(sigma, descending=True)
        if self.d > 1:
            gaps   = sigma_sorted[:-1] - sigma_sorted[1:]
            L_rank = self.rank_lambda * F.relu(self.rank_margin - gaps).sum()
        else:
            L_rank = torch.tensor(0.0, device=K.device)

        return L_lyap + L_spec + L_rank

    @torch.no_grad()
    def spectral_radius(self) -> float:
        return torch.linalg.eigvals(self.get_K()).abs().max().item()

    @torch.no_grad()
    def sigma_stats(self) -> dict:
        sigma = self.get_sigma().cpu()
        return {
            'min':  float(sigma.min()),
            'max':  float(sigma.max()),
            'mean': float(sigma.mean()),
            'std':  float(sigma.std()),
            'rho':  self.spectral_radius(),
        }


class GroupedKoopmanPropagator(nn.Module):
    """
    分组 Koopman 传播器：三个独立的 SingleBandKoopman。
    谱半径分化完全由上游多频带监督涌现，无任何硬约束。
    训练后预期：rho(K_low) > rho(K_mid) > rho(K_high)
    """
    def __init__(self,
                 dim_low: int, dim_mid: int, dim_high: int,
                 lyap_lambda: float  = 1e-3,
                 lyap_epsilon: float = 0.1,
                 spec_lambda: float  = 1e-2,
                 rho_soft: float     = 0.99,   # Bug4 修复
                 rank_lambda: float  = 1e-3,
                 rank_margin: float  = 0.05):
        super().__init__()
        shared = dict(
            lyap_lambda  = lyap_lambda,
            lyap_epsilon = lyap_epsilon,
            spec_lambda  = spec_lambda,
            rho_soft     = rho_soft,
            rank_lambda  = rank_lambda,
            rank_margin  = rank_margin,
        )
        self.K_low  = SingleBandKoopman(dim_low,  **shared)
        self.K_mid  = SingleBandKoopman(dim_mid,  **shared)
        self.K_high = SingleBandKoopman(dim_high, **shared)

    def forward(self, z_low, z_mid, z_high, steps=1):
        return (self.K_low(z_low,   steps),
                self.K_mid(z_mid,   steps),
                self.K_high(z_high, steps))

    def total_loss(self) -> torch.Tensor:
        return (self.K_low.compute_loss() +
                self.K_mid.compute_loss() +
                self.K_high.compute_loss())

    def lyapunov_loss(self) -> torch.Tensor:
        """兼容接口"""
        return self.total_loss()

    @torch.no_grad()
    def spectral_radius(self) -> float:
        return max(self.K_low.spectral_radius(),
                   self.K_mid.spectral_radius(),
                   self.K_high.spectral_radius())

    @torch.no_grad()
    def singular_values(self) -> torch.Tensor:
        return torch.cat([
            self.K_low.get_sigma().cpu(),
            self.K_mid.get_sigma().cpu(),
            self.K_high.get_sigma().cpu(),
        ])

    @torch.no_grad()
    def print_band_report(self, prefix: str = ""):
        sl = self.K_low.sigma_stats()
        sm = self.K_mid.sigma_stats()
        sh = self.K_high.sigma_stats()
        print(f"{prefix}+-- 分组 Koopman 谱半径报告 ---------------------------")
        print(f"{prefix}|  K_low  (dim={self.K_low.d:2d}) | "
              f"rho={sl['rho']:.4f} | "
              f"sigma=[{sl['min']:.4f},{sl['max']:.4f}] | "
              f"std={sl['std']:.4f}")
        print(f"{prefix}|  K_mid  (dim={self.K_mid.d:2d}) | "
              f"rho={sm['rho']:.4f} | "
              f"sigma=[{sm['min']:.4f},{sm['max']:.4f}] | "
              f"std={sm['std']:.4f}")
        print(f"{prefix}|  K_high (dim={self.K_high.d:2d}) | "
              f"rho={sh['rho']:.4f} | "
              f"sigma=[{sh['min']:.4f},{sh['max']:.4f}] | "
              f"std={sh['std']:.4f}")
        print(f"{prefix}|  预期：rho(low) > rho(mid) > rho(high) -> 涌现分化")
        print(f"{prefix}+------------------------------------------------------")


# ── 14.4 联合 Koopman 编码器 ─────────────────────────────────

class JointKoopmanEncoder(nn.Module):
    """
    联合多频带解耦 + 分组 Koopman 传播编码器。

    [Bug2 修复]
      1. forward 中对 multi_freq_dec 使用 torch.no_grad()（当已冻结时）
      2. 通过 _dec_frozen 标志位控制是否跳过梯度计算
    """
    def __init__(self, official_enc, multi_freq_dec,
                 propagator: GroupedKoopmanPropagator, pred_steps=1):
        super().__init__()
        self.official_enc   = official_enc
        self.multi_freq_dec = multi_freq_dec
        self.propagator     = propagator
        self.pred_steps     = pred_steps
        self._dec_frozen    = False   # Bug2 修复：冻结状态标志位

        for p in self.official_enc.parameters():
            p.requires_grad_(False)

    def freeze_dec(self):
        """冻结解耦器：停止梯度 + 标记状态"""
        for p in self.multi_freq_dec.parameters():
            p.requires_grad_(False)
        self.multi_freq_dec.eval()   # Bug2 修复：同步切换 eval 模式
        self._dec_frozen = True

    def unfreeze_dec(self):
        for p in self.multi_freq_dec.parameters():
            p.requires_grad_(True)
        self._dec_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = torch.sum(self.official_enc(x), dim=1)

        # Bug2 修复：解耦器已冻结时，用 no_grad 避免无效计算图构建
        if self._dec_frozen:
            with torch.no_grad():
                z_inv, z_low, z_mid, z_high = self.multi_freq_dec(z)
        else:
            z_inv, z_low, z_mid, z_high = self.multi_freq_dec(z)

        z_low_p, z_mid_p, z_high_p = self.propagator(
            z_low, z_mid, z_high, self.pred_steps
        )
        return torch.cat([z_inv, z_low_p, z_mid_p, z_high_p], dim=-1)


# ============================================================
# 15. 多频带微调函数（G 专用）
# ============================================================

def finetune_multifreq_module(encoder_fn, multi_freq_dec,
                              spectral_heads, train_loader,
                              val_loader, cfg: Config):
    """
    多频带解耦微调。
    [Warn2 修复] 直接使用 log1p 变换后的原始能量占比作为标签，
                 不再做独立归一化，保留频带间的相对大小关系。
    [Bug1 修复] 使用 cfg.joint_finetune_min_delta 替代 cfg.finetune_min_delta。
    """
    print(f"\n  [多频带微调] G_JointKoop")
    multi_freq_dec = multi_freq_dec.to(cfg.device)
    spectral_heads = spectral_heads.to(cfg.device)

    # 预计算并打印训练集频带能量统计（用于观察，不再用于归一化）
    print("    预计算频带能量标签（log1p 变换）...")
    all_low, all_mid, all_high = [], [], []
    for context_patches, _ in train_loader:
        last = context_patches[:, -1, :].numpy()
        for i in range(len(last)):
            lo, mi, hi = compute_band_energy(
                last[i], cfg.joint_freq_lo, cfg.joint_freq_hi
            )
            all_low.append(lo); all_mid.append(mi); all_high.append(hi)

    # [Warn2 修复] 只打印统计，不用于归一化
    print(f"    低频 log1p_energy: mean={np.mean(all_low):.4f}  "
          f"std={np.std(all_low):.4f}  "
          f"range=[{np.min(all_low):.4f},{np.max(all_low):.4f}]")
    print(f"    中频 log1p_energy: mean={np.mean(all_mid):.4f}  "
          f"std={np.std(all_mid):.4f}  "
          f"range=[{np.min(all_mid):.4f},{np.max(all_mid):.4f}]")
    print(f"    高频 log1p_energy: mean={np.mean(all_high):.4f} "
          f"std={np.std(all_high):.4f}  "
          f"range=[{np.min(all_high):.4f},{np.max(all_high):.4f}]")

    def get_band_labels(cp_np):
        """
        [Warn2 修复] 直接返回 log1p 能量值，不归一化。
        标签范围与 MultiFreqSpectralHead 的无界线性输出对齐。
        """
        last = cp_np[:, -1, :]
        lo_l, mi_l, hi_l = [], [], []
        for i in range(len(last)):
            lo, mi, hi = compute_band_energy(
                last[i], cfg.joint_freq_lo, cfg.joint_freq_hi
            )
            lo_l.append(lo); mi_l.append(mi); hi_l.append(hi)
        return (torch.tensor(lo_l, dtype=torch.float32),
                torch.tensor(mi_l, dtype=torch.float32),
                torch.tensor(hi_l, dtype=torch.float32))

    trainable = (list(multi_freq_dec.parameters()) +
                 list(spectral_heads.parameters()))
    opt = torch.optim.AdamW(trainable,
                            lr=cfg.joint_finetune_lr, weight_decay=1e-4)
    sch = warmup_cosine_lr(opt, cfg.joint_finetune_warmup,
                           cfg.joint_finetune_epochs)
    best, pat, best_dec_st, best_head_st = float('inf'), 0, None, None

    for ep in range(1, cfg.joint_finetune_epochs + 1):
        multi_freq_dec.train(); spectral_heads.train()
        for context_patches, _ in train_loader:
            ll, lm, lh = get_band_labels(context_patches.numpy())
            ll = ll.to(cfg.device)
            lm = lm.to(cfg.device)
            lh = lh.to(cfg.device)
            with torch.no_grad():
                z = encoder_fn(context_patches.to(cfg.device))
            _, z_low, z_mid, z_high = multi_freq_dec(z)
            p_low, p_mid, p_high = spectral_heads(z_low, z_mid, z_high)
            loss = (F.mse_loss(p_low,  ll) +
                    F.mse_loss(p_mid,  lm) +
                    F.mse_loss(p_high, lh))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        sch.step()

        multi_freq_dec.eval(); spectral_heads.eval()
        vl = 0.0
        with torch.no_grad():
            for context_patches, _ in val_loader:
                ll, lm, lh = get_band_labels(context_patches.numpy())
                ll = ll.to(cfg.device)
                lm = lm.to(cfg.device)
                lh = lh.to(cfg.device)
                z = encoder_fn(context_patches.to(cfg.device))
                _, z_low, z_mid, z_high = multi_freq_dec(z)
                p_low, p_mid, p_high = spectral_heads(z_low, z_mid, z_high)
                vl += (F.mse_loss(p_low,  ll) +
                       F.mse_loss(p_mid,  lm) +
                       F.mse_loss(p_high, lh)).item()
        vl /= max(1, len(val_loader))

        if ep % 30 == 0 or ep <= 3:
            print(f"    ep {ep:3d} | val_band_mse={vl:.6f}")

        # Bug1 修复：使用 joint_finetune_min_delta
        if vl < best - cfg.joint_finetune_min_delta:
            best, pat = vl, 0
            best_dec_st  = copy.deepcopy(multi_freq_dec.state_dict())
            best_head_st = copy.deepcopy(spectral_heads.state_dict())
        else:
            pat += 1
        if pat >= cfg.joint_finetune_patience:
            print(f"    [早停] ep={ep} | best={best:.6f}")
            break

    if best_dec_st:  multi_freq_dec.load_state_dict(best_dec_st)
    if best_head_st: spectral_heads.load_state_dict(best_head_st)
    # Bug3 修复：确保返回 eval 模式
    multi_freq_dec.eval(); spectral_heads.eval()
    print(f"    [OK] 多频带微调完成 | best_val={best:.6f}")
    return multi_freq_dec, spectral_heads


# ============================================================
# 16. G 专用 Koopman 训练函数
# ============================================================

def train_joint_koopman_head(joint_enc: JointKoopmanEncoder,
                             decoder, train_loader, val_loader,
                             cfg: Config):
    """
    G_JointKoop 训练。
    [Bug2 修复] freeze_dec() 已在内部同步 eval()，此处无需重复调用。
    [Bug3 修复] 训练结束后显式切换 eval 模式。
    """
    print(f"\n  -- 训练 G_JointKoop --")
    joint_enc = joint_enc.to(cfg.device)
    decoder   = decoder.to(cfg.device)

    # Bug2 修复：freeze_dec 内部已处理 eval + requires_grad=False
    joint_enc.freeze_dec()

    trainable = (list(joint_enc.propagator.parameters()) +
                 list(decoder.parameters()))
    n_prop = sum(p.numel() for p in joint_enc.propagator.parameters())
    n_dec  = sum(p.numel() for p in decoder.parameters())
    print(f"    可训练参数: propagator={n_prop:,} | decoder={n_dec:,}")
    print(f"    K_low={joint_enc.propagator.K_low.d}维 | "
          f"K_mid={joint_enc.propagator.K_mid.d}维 | "
          f"K_high={joint_enc.propagator.K_high.d}维")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=1e-7)
    best, pat = float('inf'), 0
    best_prop_st = best_dec_st = None

    for ep in range(1, cfg.num_epochs + 1):
        # Bug2 修复：只让 propagator 和 decoder 进入 train 模式
        # multi_freq_dec 已被 freeze_dec() 固定为 eval，不受影响
        joint_enc.propagator.train()
        decoder.train()
        ep_pred = ep_reg = 0.0; nb = 0

        for context_patches, target_patch in train_loader:
            context_patches = context_patches.to(cfg.device)
            target_patch    = target_patch.to(cfg.device)
            z      = joint_enc(context_patches)
            pred   = decoder(z)
            L_pred = F.mse_loss(pred, target_patch)
            L_reg  = joint_enc.propagator.total_loss()
            loss   = L_pred + L_reg
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
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
            print(f"    [早停] ep={ep} | best_val_mse={best:.6f}")
            # Bug3 修复：早停时也打印最终谱半径报告
            joint_enc.propagator.print_band_report(prefix="    ")
            break

    if best_prop_st: joint_enc.propagator.load_state_dict(best_prop_st)
    if best_dec_st:  decoder.load_state_dict(best_dec_st)

    sv = joint_enc.propagator.singular_values()
    print(f"    [OK] best_val={best:.6f} | "
          f"rho_max={joint_enc.propagator.spectral_radius():.6f} | "
          f"sigma in [{sv.min():.4f}, {sv.max():.4f}]")
    joint_enc.propagator.print_band_report(prefix="    ")

    # Bug3 修复：训练结束后显式切换 eval 模式
    joint_enc.eval(); decoder.eval()
    return joint_enc, decoder


# ============================================================
# 17. 主流程
# ============================================================

def run_all():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print("=" * 70)
    print("TS-JEPA -- 消融链 B->D->E->F->G")
    print(f"device={cfg.device} | embed_dim={cfg.embed_dim} | "
          f"spectral_dim={cfg.spectral_dim} | lr={cfg.lr}")
    print(f"[G_JointKoop] dim_low={cfg.joint_dim_low} | "
          f"dim_mid={cfg.joint_dim_mid} | dim_high={cfg.joint_dim_high}")
    print(f"[G_JointKoop] rho_soft={cfg.joint_rho_soft} | "
          f"spec_lambda={cfg.joint_spec_lambda} | "
          f"rank_lambda={cfg.joint_rank_lambda}")
    print("=" * 70)

    train_loader, val_loader, test_df = load_official_data(cfg)
    official_enc = load_official_encoder(cfg)
    for p in official_enc.parameters():
        p.requires_grad = False
    official_enc.eval()

    def base_enc_fn(x):
        return torch.sum(official_enc(x.to(cfg.device)), dim=1)

    results = {}

    # ══════════════════════════════════════════════════════
    # B_JEPA：官方预训练（基准）
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【B_JEPA】官方预训练 checkpoint（基准）")
    dec_B = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_B = train_forecast_head(
        base_enc_fn, dec_B, train_loader, val_loader, cfg, "B_JEPA"
    )
    results["B_JEPA"] = rolling_test(
        base_enc_fn, dec_B, test_df, cfg, "B_JEPA"
    )
    # ══════════════════════════════════════════════════════
    # D_Decoupled：单频解耦（E / F 依赖此模块）
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【D_Decoupled】单频解耦 latent（E/F 依赖）")
    decoupler = DecoupledProjector(
        cfg.embed_dim, cfg.spectral_dim
    ).to(cfg.device)
    sh_D = SpectralHead(cfg.spectral_dim).to(cfg.device)

    class _FreqModuleD(nn.Module):
        def __init__(self, dec, sh):
            super().__init__()
            self.dec = dec
            self.sh  = sh
        def forward(self, z):
            _, z_spec = self.dec(z)
            return self.sh(z_spec)

    freq_mod_D = finetune_freq_module(
        base_enc_fn, _FreqModuleD(decoupler, sh_D),
        train_loader, val_loader, cfg, "D_Decoupled"
    )
    # finetune_freq_module 已在内部调用 .eval()，此处直接冻结
    for p in freq_mod_D.parameters():
        p.requires_grad = False

    def decoupled_enc_fn(x):
        z = base_enc_fn(x)
        with torch.no_grad():
            z_inv, z_spec = freq_mod_D.dec(z)
        return torch.cat([z_inv, z_spec], dim=-1)

    dec_D = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_D = train_forecast_head(
        decoupled_enc_fn, dec_D, train_loader, val_loader, cfg, "D_Decoupled"
    )
    results["D_Decoupled"] = rolling_test(
        decoupled_enc_fn, dec_D, test_df, cfg, "D_Decoupled"
    )

    # ══════════════════════════════════════════════════════
    # E_KoopProp：单一谱半径 Koopman（对照）
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【E_KoopProp】解耦 + ODO Koopman（单一谱半径，对照）")
    prop_E = ODOKoopmanPropagator(
        spectral_dim          = cfg.spectral_dim,
        spectral_radius_bound = cfg.koop_spectral_radius,
        lyapunov_lambda       = cfg.koop_lyap_lambda,
        lyapunov_epsilon      = cfg.koop_lyap_epsilon,
    )
    koop_enc_E = KoopmanEncoder(
        official_enc = official_enc,
        decoupler    = freq_mod_D.dec,
        propagator   = prop_E,
        pred_steps   = cfg.koop_pred_steps,
    )
    dec_E = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    koop_enc_E, dec_E = train_koopman_head(
        koop_enc_E, dec_E, train_loader, val_loader, cfg, "E_KoopProp"
    )
    # train_koopman_head 已在内部调用 .eval()
    results["E_KoopProp"] = rolling_test(
        lambda x: koop_enc_E(x.to(cfg.device)),
        dec_E, test_df, cfg, "E_KoopProp"
    )

    # ══════════════════════════════════════════════════════
    # F_PerDimKoop：每维度独立谱半径（对照）
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【F_PerDimKoop】解耦 + 每维度独立可学习谱半径（对照）")
    prop_F = PerDimODOKoopmanPropagator(
        spectral_dim     = cfg.spectral_dim,
        rho_min          = cfg.perdim_rho_min,
        rho_max          = cfg.perdim_rho_max,
        lyapunov_lambda  = cfg.perdim_lyap_lambda,
        lyapunov_epsilon = cfg.perdim_lyap_epsilon,
    )
    koop_enc_F = KoopmanEncoder(
        official_enc = official_enc,
        decoupler    = freq_mod_D.dec,   # 复用 D 的解耦器（已冻结）
        propagator   = prop_F,
        pred_steps   = cfg.koop_pred_steps,
    )
    dec_F = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    koop_enc_F, dec_F = train_koopman_head(
        koop_enc_F, dec_F, train_loader, val_loader, cfg, "F_PerDimKoop"
    )
    results["F_PerDimKoop"] = rolling_test(
        lambda x: koop_enc_F(x.to(cfg.device)),
        dec_F, test_df, cfg, "F_PerDimKoop"
    )
    # ══════════════════════════════════════════════════════
    # G_JointKoop：多频带联合解耦 + 分组 Koopman ★★★
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【G_JointKoop】多频带联合解耦 + 分组 Koopman ★★★")
    print(f"  子空间：z_low({cfg.joint_dim_low}) + "
          f"z_mid({cfg.joint_dim_mid}) + "
          f"z_high({cfg.joint_dim_high}) = {cfg.spectral_dim} 维")
    print(f"  谱半径：无硬约束，由多频带监督涌现分化")
    print(f"  软惩罚：rho_soft={cfg.joint_rho_soft} | "
          f"spec_lambda={cfg.joint_spec_lambda}")
    print(f"  排序损失：rank_lambda={cfg.joint_rank_lambda} | "
          f"margin={cfg.joint_rank_margin}")
    # Step G-1：构建多频带解耦器 + 监督头
    multi_freq_dec = MultiFreqDecoupledProjector(
        embed_dim = cfg.embed_dim,
        dim_low   = cfg.joint_dim_low,
        dim_mid   = cfg.joint_dim_mid,
        dim_high  = cfg.joint_dim_high,
    ).to(cfg.device)
    multi_freq_heads = MultiFreqSpectralHead(
        dim_low  = cfg.joint_dim_low,
        dim_mid  = cfg.joint_dim_mid,
        dim_high = cfg.joint_dim_high,
    ).to(cfg.device)

    # Step G-2：多频带微调（训练解耦器 + 监督头）
    multi_freq_dec, multi_freq_heads = finetune_multifreq_module(
        base_enc_fn, multi_freq_dec, multi_freq_heads,
        train_loader, val_loader, cfg
    )
    # finetune_multifreq_module 已在内部调用 .eval()，直接冻结监督头
    for p in multi_freq_heads.parameters():
        p.requires_grad = False

    # Step G-3：构建分组 Koopman 传播器
    grouped_prop = GroupedKoopmanPropagator(
        dim_low      = cfg.joint_dim_low,
        dim_mid      = cfg.joint_dim_mid,
        dim_high     = cfg.joint_dim_high,
        lyap_lambda  = cfg.joint_lyap_lambda,
        lyap_epsilon = cfg.joint_lyap_epsilon,
        spec_lambda  = cfg.joint_spec_lambda,
        rho_soft     = cfg.joint_rho_soft,
        rank_lambda  = cfg.joint_rank_lambda,
        rank_margin  = cfg.joint_rank_margin,
    )

    # Step G-4：组装联合编码器
    joint_enc = JointKoopmanEncoder(
        official_enc   = official_enc,
        multi_freq_dec = multi_freq_dec,
        propagator     = grouped_prop,
        pred_steps     = cfg.koop_pred_steps,
    )

    # Step G-5：训练 Koopman 传播器 + 预测头
    dec_G = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    joint_enc, dec_G = train_joint_koopman_head(
        joint_enc, dec_G, train_loader, val_loader, cfg
    )
    # train_joint_koopman_head 已在内部调用 .eval()
    results["G_JointKoop"] = rolling_test(
        lambda x: joint_enc(x.to(cfg.device)),
        dec_G, test_df, cfg, "G_JointKoop"
    )

    # ══════════════════════════════════════════════════════
    # 完整消融对比表
    # ══════════════════════════════════════════════════════
    bm = results["B_JEPA"]["MSE"]
    ba = results["B_JEPA"]["MAE"]

    components = {
        "B_JEPA"       : "JEPA预训练（基准）",
        "D_Decoupled"  : "+单频解耦",
        "E_KoopProp"   : "+Koopman(单一谱半径)",
        "F_PerDimKoop" : "+Koopman(每维度独立谱半径)",
        "G_JointKoop"  : "+多频带联合解耦+分组Koopman ★★★",
    }

    print("\n" + "=" * 90)
    print("  完整消融对比表（滚动测试）-- Weather OT 列")
    print("=" * 90)
    print(f"  {'组别':<22} | {'MSE':>10} | {'MAE':>10} | "
           f"{'ΔMSE vs B':>12} | {'ΔMAE vs B':>12} | 新增组件")
    print("-" * 90)
    for name, res in results.items():
       comp = components.get(name, "")
       if name == "B_JEPA":
           ds = da = "(基准)"
       else:
           ds = f"{(res['MSE'] - bm) / bm * 100:+.2f}%"
           da = f"{(res['MAE'] - ba) / ba * 100:+.2f}%"
       print(f"  {name:<22} | {res['MSE']:>10.6f} | {res['MAE']:>10.6f} | "
              f"{ds:>12} | {da:>12} | {comp}")
    print("=" * 90)

    # 消融结论
    print("\n  消融结论：")
    print("  D vs B  -> 单频解耦是否有效                                  [待验证]")
    print("  E vs D  -> Koopman 动力学传播是否有效                        [待验证]")
    print("  F vs E  -> 每维度独立谱半径是否优于单一约束                   [待验证]")
    print("  G vs F  -> 多频带联合解耦+分组Koopman是否优于解耦后分层       [待验证]")
    print()
    print("  G 的核心论文结论（需从 print_band_report 日志中确认）：")
    print("  1. rho(K_low) > rho(K_mid) > rho(K_high) 自发涌现（无硬约束）")
    print("  2. 多频带监督是谱半径分化的根本驱动力")
    print("  3. 解耦与传播联合设计优于解耦后再分层")

    return results


if __name__ == "__main__":
    run_all()
