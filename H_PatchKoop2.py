"""
============================================================
H_PatchKoop — Patch-level Conditioned Koopman（修复版）
============================================================
修复要点：
  1. decoder 输入维度 = embed_dim = 128（与 A-E 对齐，公平对比）
     cat[z_inv_mean(112), ẑ_osc_pred(16)] = 128
     z_global 仅用于条件化 K，不进 decoder

  2. 每路输出加 LayerNorm，防止数值尺度不匹配

  3. z_trend 通过残差加到 z_inv_mean，而非拼接
     → 保持 decoder 输入 128 维的同时引入全局趋势信息

  4. 使用官方 LinearDecoder（与 B/D/E 完全一致）
============================================================
"""

import os
import sys
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

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

    # ── Encoder 参数 ────────────────────────────────────────
    embed_dim      = 128
    encoder_nhead  = 2
    encoder_layers = 1
    kernel_size    = 3

    # ── 训练参数（与 A-E 完全一致）──────────────────────────
    batch_size = 32
    lr         = 1e-5       # 与 B/D/E 对齐
    num_epochs = 500
    patience   = 50
    min_delta  = 1e-6

    # ── H_PatchKoop 维度设计 ────────────────────────────────
    # patch_inv_dim + patch_osc_dim = embed_dim = 128
    patch_inv_dim  = 112
    patch_osc_dim  = 16

    # global_trend_dim = patch_inv_dim（用于残差，维度必须匹配）
    global_trend_dim = 112
    global_cycle_dim = 16   # 仅用于 FiLM 条件化（可选）

    # decoder 输入 = embed_dim = 128（与 A-E 完全一致！）
    # cat[z_inv_fused(112), ẑ_osc_pred(16)] = 128
    decoder_input_dim = 128

    # ── Koopman 参数 ────────────────────────────────────────
    koop_spectral_radius = 0.99
    koop_lyap_lambda     = 1e-3
    koop_lyap_epsilon    = 0.1

    # FiLM
    film_hidden_dim = 64

    # 损失权重
    lambda_dyn  = 0.1
    lambda_lyap = 1e-3

    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    seed   = 42


cfg = Config()


# ============================================================
# 1. 工具函数
# ============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mse_np(pred, true):
    return float(np.mean((pred - true) ** 2))


def mae_np(pred, true):
    return float(np.mean(np.abs(pred - true)))


def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    """Q = (I - A_skew)(I + A_skew)^{-1}，天然满足 Q^T Q = I"""
    n      = A.shape[-1]
    I      = torch.eye(n, device=A.device, dtype=A.dtype)
    A_skew = (A - A.transpose(-2, -1)) / 2.0
    return torch.linalg.solve(I + A_skew, I - A_skew)


# ============================================================
# 2. 数据加载
# ============================================================

def load_data(cfg: Config):
    print(f"\n[数据] 加载 {cfg.data_path}")
    full_loader = get_evaluation_loaders(
        cfg.data_path,
        cfg.batch_size,
        ratio_patches=cfg.num_patches,
        mask_ratio=0.7,
    )
    dataset  = full_loader.dataset
    train_df = dataset.train_df
    test_df  = dataset.test_df

    n_full = len(train_df)
    n_val  = int(n_full * 0.1)
    n_tr   = n_full - n_val

    dataset.train_df = train_df[:n_tr]
    val_df           = train_df[n_tr:]

    train_loader = DataLoader(
        dataset, batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=4
    )
    val_dataset          = copy.copy(dataset)
    val_dataset.train_df = val_df
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4
    )

    print(f"  train={n_tr:,} | val={n_val:,} | test_df_len={len(test_df):,}")
    return train_loader, val_loader, test_df


# ============================================================
# 3. 加载官方 Encoder
# ============================================================

def load_encoder(cfg: Config) -> OfficialEncoder:
    print(f"\n[Encoder] 加载 {os.path.basename(cfg.checkpoint_path)}")
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

    # 验证输出形状（注意：不做 sum pooling，保留序列维度）
    with torch.no_grad():
        dummy = torch.randn(2, cfg.num_patches, cfg.context).to(cfg.device)
        z_seq = enc(dummy)
    assert z_seq.shape == (2, cfg.num_patches, cfg.embed_dim), \
        f"Encoder 输出形状异常: {z_seq.shape}"

    print(f"  ✅ 加载成功 | epoch={ckpt.get('epoch','?')} | "
          f"z_seq 形状={tuple(z_seq.shape)} | "
          f"mean={z_seq.mean():.4f} | std={z_seq.std():.4f}")

    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    return enc


# ============================================================
# 4. Temporal Attention Pooling（Path B 全局聚合）
# ============================================================

class TemporalAttentionPool(nn.Module):
    """
    (B, T, D) → (B, D)
    可学习 query 的单头注意力聚合。
    输出经过 LayerNorm，保证数值尺度稳定。
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query    = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.val_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale    = embed_dim ** -0.5
        self.norm     = nn.LayerNorm(embed_dim)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """z_seq: (B, T, D) → z_global: (B, D)"""
        B = z_seq.shape[0]
        q    = self.query.expand(B, 1, -1)
        k    = self.key_proj(z_seq)
        v    = self.val_proj(z_seq)
        attn = F.softmax(
            torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1
        )                                               # (B, 1, T)
        out  = torch.bmm(attn, v).squeeze(1)           # (B, D)
        return self.norm(out)


# ============================================================
# 5. Patch-level Decoupler（Path A）
# ============================================================

class PatchDecoupler(nn.Module):
    """
    (B, T, D) → (B, T, inv_dim) + (B, T, osc_dim)
    共享权重，逐 patch 独立解耦。
    两路输出分别 LayerNorm，保证数值尺度一致。
    """
    def __init__(self, embed_dim: int, inv_dim: int, osc_dim: int):
        super().__init__()
        assert inv_dim + osc_dim == embed_dim
        self.proj_inv  = nn.Linear(embed_dim, inv_dim)
        self.proj_osc  = nn.Linear(embed_dim, osc_dim)
        self.norm_inv  = nn.LayerNorm(inv_dim)
        self.norm_osc  = nn.LayerNorm(osc_dim)

    def forward(self, z_seq: torch.Tensor):
        """(B,T,D) → (B,T,inv_dim), (B,T,osc_dim)"""
        return (self.norm_inv(self.proj_inv(z_seq)),
                self.norm_osc(self.proj_osc(z_seq)))


# ============================================================
# 6. Global Decoupler（Path B）
# ============================================================

class GlobalDecoupler(nn.Module):
    """
    (B, D) → (B, trend_dim) + (B, cycle_dim)
    trend 用于与 z_inv_mean 做残差融合。
    cycle 用于辅助 FiLM 条件化（可选，此处直接用 z_global）。
    两路输出分别 LayerNorm。
    """
    def __init__(self, embed_dim: int, trend_dim: int, cycle_dim: int):
        super().__init__()
        assert trend_dim + cycle_dim == embed_dim
        self.proj_trend = nn.Linear(embed_dim, trend_dim)
        self.proj_cycle = nn.Linear(embed_dim, cycle_dim)
        self.norm_trend = nn.LayerNorm(trend_dim)
        self.norm_cycle = nn.LayerNorm(cycle_dim)

    def forward(self, z_global: torch.Tensor):
        """(B,D) → (B,trend_dim), (B,cycle_dim)"""
        return (self.norm_trend(self.proj_trend(z_global)),
                self.norm_cycle(self.proj_cycle(z_global)))


# ============================================================
# 7. FiLM Hyper-network
# ============================================================

class FiLMHyperNetwork(nn.Module):
    """
    z_global(B,D) → γ(B,d,d) + β(B,d,d)
    K(z_global) = γ ⊙ K_base + β

    初始化：γ → I（单位矩阵展平），β → 0
    保证训练初期 K(z_global) ≈ K_base，训练稳定。
    """
    def __init__(self, input_dim: int, koop_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.koop_dim = koop_dim
        out_dim = koop_dim * koop_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden_dim, out_dim)
        self.beta_head  = nn.Linear(hidden_dim, out_dim)

        # 关键初始化
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
        with torch.no_grad():
            self.gamma_head.bias.copy_(torch.eye(koop_dim).flatten())

    def forward(self, z_global: torch.Tensor):
        """(B, D) → gamma(B,d,d), beta(B,d,d)"""
        h     = self.net(z_global)
        gamma = self.gamma_head(h).view(-1, self.koop_dim, self.koop_dim)
        beta  = self.beta_head(h).view(-1, self.koop_dim, self.koop_dim)
        return gamma, beta


# ============================================================
# 8. Conditioned Koopman Operator
# ============================================================

class ConditionedKoopmanOperator(nn.Module):
    """
    K(z_global) = γ(z_global) ⊙ K_base + β(z_global)
    K_base = U · diag(σ) · V^T，σ_i ≤ ρ_max
    """
    def __init__(self,
                 koop_dim            : int,
                 embed_dim           : int,
                 film_hidden_dim     : int   = 64,
                 spectral_radius_max : float = 0.99,
                 lyap_lambda         : float = 1e-3,
                 lyap_epsilon        : float = 0.1):
        super().__init__()
        self.d        = koop_dim
        self.rho_max  = spectral_radius_max
        self.lam_lyap = lyap_lambda
        self.eps_lyap = lyap_epsilon

        self.A_U       = nn.Parameter(torch.randn(koop_dim, koop_dim) * 0.01)
        self.A_V       = nn.Parameter(torch.randn(koop_dim, koop_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.zeros(koop_dim))

        self.film = FiLMHyperNetwork(
            input_dim  = embed_dim,
            koop_dim   = koop_dim,
            hidden_dim = film_hidden_dim,
        )

    def _get_K_base(self) -> torch.Tensor:
        U     = cayley_transform(self.A_U)
        V     = cayley_transform(self.A_V)
        sigma = torch.clamp(F.softplus(self.log_sigma), max=self.rho_max)
        return U @ torch.diag(sigma) @ V.t()   # (d, d)

    def get_K(self, z_global: torch.Tensor) -> torch.Tensor:
        """(B, D) → K: (B, d, d)"""
        K_base      = self._get_K_base()                    # (d, d)
        gamma, beta = self.film(z_global)                   # (B, d, d)
        return gamma * K_base.unsqueeze(0) + beta           # (B, d, d)

    def forward(self,
                z_osc_seq  : torch.Tensor,
                z_global   : torch.Tensor,
                pred_steps : int = 1) -> torch.Tensor:
        """
        z_osc_seq: (B, T, d)
        z_global:  (B, D)
        → z_osc_pred: (B, d)
        """
        K = self.get_K(z_global)               # (B, d, d)
        z = z_osc_seq[:, -1, :]                # (B, d)
        for _ in range(pred_steps):
            z = torch.bmm(K, z.unsqueeze(-1)).squeeze(-1)
        return z

    def dynamics_loss(self,
                      z_osc_seq : torch.Tensor,
                      z_global  : torch.Tensor) -> torch.Tensor:
        """
        L_dyn = mean_{t} || K·z_osc[t] - z_osc[t+1] ||²
        z_osc_seq: (B, T, d)，z_global: (B, D)
        """
        K      = self.get_K(z_global)               # (B, d, d)
        z_src  = z_osc_seq[:, :-1, :]               # (B, T-1, d)
        z_tgt  = z_osc_seq[:, 1:,  :]               # (B, T-1, d)
        z_pred = torch.bmm(z_src, K.transpose(1, 2))# (B, T-1, d)
        return F.mse_loss(z_pred, z_tgt)

    def lyapunov_loss(self) -> torch.Tensor:
        K_base = self._get_K_base()
        I      = torch.eye(self.d, device=K_base.device, dtype=K_base.dtype)
        return self.lam_lyap * F.relu(
            torch.norm(K_base.t() @ K_base - I, p='fro') - self.eps_lyap
        )

    @torch.no_grad()
    def spectral_radius_base(self) -> float:
        return torch.linalg.eigvals(self._get_K_base()).abs().max().item()

    @torch.no_grad()
    def singular_values_base(self) -> torch.Tensor:
        return torch.clamp(
            F.softplus(self.log_sigma), max=self.rho_max
        ).detach().cpu()


# ============================================================
# 9. H_PatchKoop Encoder
# ============================================================

class HPatchKoopEncoder(nn.Module):
    """
    双路编码器，输出 128 维向量（与 A-E 对齐）。

    输出结构：
      z_inv_fused (112) = z_inv_mean + z_trend  （残差融合全局趋势）
      ẑ_osc_pred  (16)  = K(z_global) · z_osc[T]（Koopman 外推）
      cat → (128)

    z_global 仅用于：
      1. 条件化 K（FiLM）
      2. 生成 z_trend 做残差
    不直接进 decoder，保持维度 = 128。
    """
    def __init__(self,
                 official_enc : nn.Module,
                 patch_dec    : PatchDecoupler,
                 global_pool  : TemporalAttentionPool,
                 global_dec   : GlobalDecoupler,
                 koop_op      : ConditionedKoopmanOperator,
                 pred_steps   : int = 1):
        super().__init__()
        self.official_enc = official_enc
        self.patch_dec    = patch_dec
        self.global_pool  = global_pool
        self.global_dec   = global_dec
        self.koop_op      = koop_op
        self.pred_steps   = pred_steps

        for p in self.official_enc.parameters():
            p.requires_grad_(False)

    def encode(self, x: torch.Tensor):
        """
        返回所有中间表示，训练时使用。
        x: (B, T, context)
        """
        with torch.no_grad():
            z_seq = self.official_enc(x)          # (B, T, D)

        # Path A：逐 patch 解耦
        z_inv_seq, z_osc_seq = self.patch_dec(z_seq)   # (B,T,112), (B,T,16)

        # Path B：全局聚合 + 解耦
        z_global         = self.global_pool(z_seq)      # (B, 128)
        z_trend, z_cycle = self.global_dec(z_global)    # (B,112), (B,16)

        return z_inv_seq, z_osc_seq, z_global, z_trend, z_cycle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, context)
        → h: (B, 128)
        """
        z_inv_seq, z_osc_seq, z_global, z_trend, _ = self.encode(x)

        # Koopman 外推：(B, 16)
        z_osc_pred = self.koop_op(z_osc_seq, z_global, self.pred_steps)

        # 不变分量均值 + 全局趋势残差：(B, 112)
        z_inv_mean   = z_inv_seq.mean(dim=1)             # (B, 112)
        z_inv_fused  = z_inv_mean + z_trend              # 残差融合，维度不变

        # 拼接：(B, 128)
        return torch.cat([z_inv_fused, z_osc_pred], dim=-1)


# ============================================================
# 10. 训练函数
# ============================================================

def train_patch_koop(h_enc        : HPatchKoopEncoder,
                     decoder      : nn.Module,
                     train_loader,
                     val_loader,
                     cfg          : Config) -> tuple:
    print(f"\n  ── 训练 H_PatchKoop ──")
    h_enc   = h_enc.to(cfg.device)
    decoder = decoder.to(cfg.device)

    modules = {
        'patch_dec'  : h_enc.patch_dec,
        'global_pool': h_enc.global_pool,
        'global_dec' : h_enc.global_dec,
        'koop_op'    : h_enc.koop_op,
        'decoder'    : decoder,
    }
    trainable = []
    for name, mod in modules.items():
        params = list(mod.parameters())
        n = sum(p.numel() for p in params)
        print(f"    {name:<14}: {n:>8,} params")
        trainable.extend(params)
    print(f"    {'total':<14}: {sum(p.numel() for p in trainable):>8,} params")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=1e-7
    )

    best      = float('inf')
    pat       = 0
    best_ckpt = {}

    for ep in range(1, cfg.num_epochs + 1):
        # ── 训练 ──
        h_enc.train()
        decoder.train()
        h_enc.official_enc.eval()   # encoder 始终 eval

        ep_pred = ep_dyn = ep_lyap = 0.0
        nb = 0

        for context_patches, target_patch in train_loader:
            context_patches = context_patches.to(cfg.device)
            target_patch    = target_patch.to(cfg.device)

            (z_inv_seq, z_osc_seq,
             z_global, z_trend, _) = h_enc.encode(context_patches)

            z_osc_pred  = h_enc.koop_op(z_osc_seq, z_global, h_enc.pred_steps)
            z_inv_mean  = z_inv_seq.mean(dim=1)
            z_inv_fused = z_inv_mean + z_trend
            h_vec       = torch.cat([z_inv_fused, z_osc_pred], dim=-1)
            pred        = decoder(h_vec)

            L_pred = F.mse_loss(pred, target_patch)
            L_dyn  = h_enc.koop_op.dynamics_loss(z_osc_seq, z_global)
            L_lyap = h_enc.koop_op.lyapunov_loss()
            loss   = (L_pred
                      + cfg.lambda_dyn  * L_dyn
                      + cfg.lambda_lyap * L_lyap)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            ep_pred += L_pred.item()
            ep_dyn  += L_dyn.item()
            ep_lyap += L_lyap.item()
            nb      += 1

        sch.step()

        # ── 验证 ──
        h_enc.eval()
        decoder.eval()
        vl = 0.0
        with torch.no_grad():
            for context_patches, target_patch in val_loader:
                h_vec = h_enc(context_patches.to(cfg.device))
                pred  = decoder(h_vec)
                vl   += F.mse_loss(
                    pred, target_patch.to(cfg.device)
                ).item()
        vl /= max(1, len(val_loader))

        # ── 日志 ──
        if ep % 50 == 0 or ep <= 5:
            rho = h_enc.koop_op.spectral_radius_base()
            sv  = h_enc.koop_op.singular_values_base()
            print(f"    ep {ep:4d} | "
                  f"pred={ep_pred/nb:.5f} | "
                  f"dyn={ep_dyn/nb:.5f} | "
                  f"lyap={ep_lyap/nb:.6f} | "
                  f"val={vl:.5f} | "
                  f"ρ={rho:.4f} | "
                  f"σ∈[{sv.min():.3f},{sv.max():.3f}]")

        # ── 早停 ──
        if vl < best - cfg.min_delta:
            best = vl
            pat  = 0
            best_ckpt = {
                'patch_dec'  : copy.deepcopy(h_enc.patch_dec.state_dict()),
                'global_pool': copy.deepcopy(h_enc.global_pool.state_dict()),
                'global_dec' : copy.deepcopy(h_enc.global_dec.state_dict()),
                'koop_op'    : copy.deepcopy(h_enc.koop_op.state_dict()),
                'decoder'    : copy.deepcopy(decoder.state_dict()),
            }
        else:
            pat += 1

        if pat >= cfg.patience:
            print(f"    [早停] ep={ep} | best_val={best:.6f}")
            break

    if best_ckpt:
        h_enc.patch_dec.load_state_dict(best_ckpt['patch_dec'])
        h_enc.global_pool.load_state_dict(best_ckpt['global_pool'])
        h_enc.global_dec.load_state_dict(best_ckpt['global_dec'])
        h_enc.koop_op.load_state_dict(best_ckpt['koop_op'])
        decoder.load_state_dict(best_ckpt['decoder'])

    rho_f = h_enc.koop_op.spectral_radius_base()
    sv_f  = h_enc.koop_op.singular_values_base()
    print(f"    ✅ 完成 | best_val={best:.6f} | "
          f"ρ={rho_f:.6f} | σ∈[{sv_f.min():.4f},{sv_f.max():.4f}]")

    h_enc.eval()
    decoder.eval()
    return h_enc, decoder


# ============================================================
# 11. 滚动测试
# ============================================================

def rolling_test(enc_fn, decoder, test_df,
                 cfg: Config, tag: str) -> dict:
    print(f"\n  ── 滚动测试: {tag} ──")
    context     = cfg.context
    num_patches = cfg.num_patches
    num_steps   = (len(test_df) - context * num_patches) // context
    print(f"    test_df={len(test_df):,} | 滚动步数={num_steps:,}")

    decoder.eval()
    l_mse, l_mae = [], []

    for step in range(num_steps):
        ctx = (
            test_df[context * step : context * num_patches + context * step]
            .reshape(num_patches, context)
            .unsqueeze(0)
            .to(cfg.device)
        )
        tgt = test_df[
            context * num_patches + step * context :
            context * num_patches + (step + 1) * context
        ]
        with torch.no_grad():
            h    = enc_fn(ctx)
            pred = decoder(h)

        l_mse.append(mse_np(pred.flatten().cpu().numpy(), tgt.numpy()))
        l_mae.append(mae_np(pred.flatten().cpu().numpy(), tgt.numpy()))

    result = {"MSE": float(np.mean(l_mse)), "MAE": float(np.mean(l_mae))}
    print(f"    ✅ MSE={result['MSE']:.6f} | MAE={result['MAE']:.6f}")
    return result


# ============================================================
# 12. 主流程
# ============================================================

def main():
    set_seed(cfg.seed)

    print("=" * 65)
    print("H_PatchKoop — Patch-level Conditioned Koopman（修复版）")
    print(f"device={cfg.device} | lr={cfg.lr}")
    print(f"patch_inv={cfg.patch_inv_dim} | patch_osc={cfg.patch_osc_dim}")
    print(f"decoder_input={cfg.decoder_input_dim}（与 A-E 对齐）")
    print(f"λ_dyn={cfg.lambda_dyn} | λ_lyap={cfg.lambda_lyap}")
    print("=" * 65)

    # ── 数据 & Encoder ────────────────────────────────────
    train_loader, val_loader, test_df = load_data(cfg)
    official_enc = load_encoder(cfg)

    # ── 构建各模块 ────────────────────────────────────────
    patch_dec   = PatchDecoupler(
        embed_dim = cfg.embed_dim,
        inv_dim   = cfg.patch_inv_dim,
        osc_dim   = cfg.patch_osc_dim,
    )
    global_pool = TemporalAttentionPool(embed_dim=cfg.embed_dim)
    global_dec  = GlobalDecoupler(
        embed_dim  = cfg.embed_dim,
        trend_dim  = cfg.global_trend_dim,
        cycle_dim  = cfg.global_cycle_dim,
    )
    koop_op     = ConditionedKoopmanOperator(
        koop_dim            = cfg.patch_osc_dim,
        embed_dim           = cfg.embed_dim,
        film_hidden_dim     = cfg.film_hidden_dim,
        spectral_radius_max = cfg.koop_spectral_radius,
        lyap_lambda         = cfg.koop_lyap_lambda,
        lyap_epsilon        = cfg.koop_lyap_epsilon,
    )

    # ── 组装 Encoder ──────────────────────────────────────
    h_enc = HPatchKoopEncoder(
        official_enc = official_enc,
        patch_dec    = patch_dec,
        global_pool  = global_pool,
        global_dec   = global_dec,
        koop_op      = koop_op,
        pred_steps   = 1,
    )

    # ── Decoder（与 B/D/E 完全一致的 LinearDecoder）────────
    decoder = LinearDecoder(
        emb_dim    = cfg.decoder_input_dim,   # 128
        patch_size = cfg.pred_len,
    )

    # ── 维度验证 ──────────────────────────────────────────
    print("\n[维度验证]")
    with torch.no_grad():
        dummy    = torch.randn(2, cfg.num_patches, cfg.context).to(cfg.device)
        h_enc_d  = h_enc.to(cfg.device)
        dec_d    = decoder.to(cfg.device)
        h_out    = h_enc_d(dummy)
        pred_out = dec_d(h_out)
    print(f"  encoder 输出: {tuple(h_out.shape)}  "
          f"(期望: (2, {cfg.decoder_input_dim}))")
    print(f"  decoder 输出: {tuple(pred_out.shape)}  "
          f"(期望: (2, {cfg.pred_len}))")
    assert h_out.shape == (2, cfg.decoder_input_dim)
    assert pred_out.shape == (2, cfg.pred_len)
    print("  ✅ 维度验证通过")

    # ── 训练 ──────────────────────────────────────────────
    h_enc, decoder = train_patch_koop(
        h_enc, decoder, train_loader, val_loader, cfg
    )

    # ── 测试 ──────────────────────────────────────────────
    def enc_fn(x):
        return h_enc(x.to(cfg.device))

    result = rolling_test(enc_fn, decoder, test_df, cfg, "H_PatchKoop")

    print("\n" + "=" * 65)
    print("📊 H_PatchKoop 最终结果")
    print("=" * 65)
    print(f"  MSE = {result['MSE']:.6f}")
    print(f"  MAE = {result['MAE']:.6f}")
    print("=" * 65)
    return result


if __name__ == "__main__":
    main()