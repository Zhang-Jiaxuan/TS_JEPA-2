"""
对比实验：原始JEPA vs Koopman-JEPA
在完全相同的H_PatchKoop下游结构下，只换预训练checkpoint
输出对比表格
"""

import os, sys, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, "/data_p6/zjx/TS_JEPA")
from src.models.encoder import Encoder as OfficialEncoder
from src.data_loaders.data_loader import get_evaluation_loaders

# ============================================================
# 配置：两个checkpoint，其余完全相同
# ============================================================
CHECKPOINTS = {
    "原始_JEPA": (
        "/data_p6/zjx/TS_JEPA/logs/output_model/ETTm2/"
        "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_ratio_patches_10_"
        "encoder_128_2_1_predictor_128_2_1_epoch_5000.pt"
    ),
    "Koopman_JEPA": (
        "/data_p6/zjx/TS_JEPA/logs/output_model/ETTm2/"
        "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_ratio_patches_10_"
        "encoder_128_2_1_predictor_128_2_1_epoch_5000try3.pt"
    ),
}

class Config:
    data_path        = "/data_p6/zjx/TS_JEPA/data/ETTm2/ETTm2.csv"
    context          = 32
    num_patches      = 10
    pred_len         = 32
    embed_dim        = 128
    encoder_nhead    = 2
    encoder_layers   = 1
    kernel_size      = 3
    batch_size       = 32
    lr               = 1e-4
    num_epochs       = 500
    patience         = 50
    min_delta        = 1e-6
    patch_inv_dim    = 112
    patch_osc_dim    = 16
    global_trend_dim = 112
    global_cycle_dim = 16
    decoder_input_dim= 240
    koop_spectral_radius = 0.99
    koop_lyap_lambda     = 1e-3
    koop_lyap_epsilon    = 0.1
    film_hidden_dim  = 64
    lambda_dyn       = 0.1
    lambda_lyap      = 1e-3
    device           = "cuda:2" if torch.cuda.is_available() else "cpu"
    seed             = 42   # 固定seed保证公平对比

cfg = Config()

# ============================================================
# 工具函数（与原H_PatchKoop完全相同）
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mse(p, t): return float(np.mean((p - t) ** 2))
def mae(p, t): return float(np.mean(np.abs(p - t)))

def cayley_transform(A):
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    A_skew = (A - A.transpose(-2, -1)) / 2.0
    return torch.linalg.solve(I + A_skew, I - A_skew)

# ============================================================
# 模型定义（与原H_PatchKoop完全相同，不做任何修改）
# ============================================================
class TemporalAttentionPool(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query    = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.val_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale    = embed_dim ** -0.5
    def forward(self, z_seq):
        B = z_seq.shape[0]
        q = self.query.expand(B, 1, -1)
        k = self.key_proj(z_seq)
        v = self.val_proj(z_seq)
        attn = F.softmax(torch.bmm(q, k.transpose(1,2)) * self.scale, dim=-1)
        return torch.bmm(attn, v).squeeze(1)

class PatchDecoupler(nn.Module):
    def __init__(self, embed_dim, inv_dim, osc_dim):
        super().__init__()
        self.proj_inv = nn.Linear(embed_dim, inv_dim)
        self.proj_osc = nn.Linear(embed_dim, osc_dim)
    def forward(self, z_seq):
        return self.proj_inv(z_seq), self.proj_osc(z_seq)

class GlobalDecoupler(nn.Module):
    def __init__(self, embed_dim, trend_dim, cycle_dim):
        super().__init__()
        self.proj_trend = nn.Linear(embed_dim, trend_dim)
        self.proj_cycle = nn.Linear(embed_dim, cycle_dim)
    def forward(self, z_global):
        return self.proj_trend(z_global), self.proj_cycle(z_global)

class FiLMHyperNetwork(nn.Module):
    def __init__(self, input_dim, koop_dim, hidden_dim=64):
        super().__init__()
        self.koop_dim = koop_dim
        out_dim = koop_dim * koop_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden_dim, out_dim)
        self.beta_head  = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
        with torch.no_grad():
            self.gamma_head.bias.copy_(torch.eye(koop_dim).flatten())
    def forward(self, z_global):
        h = self.net(z_global)
        gamma = self.gamma_head(h).view(-1, self.koop_dim, self.koop_dim)
        beta  = self.beta_head(h).view(-1, self.koop_dim, self.koop_dim)
        return gamma, beta

class ConditionedKoopmanOperator(nn.Module):
    def __init__(self, koop_dim, embed_dim, film_hidden_dim=64,
                 spectral_radius_max=0.99, lyap_lambda=1e-3, lyap_epsilon=0.1):
        super().__init__()
        self.d = koop_dim
        self.rho_max  = spectral_radius_max
        self.lam_lyap = lyap_lambda
        self.eps_lyap = lyap_epsilon
        self.A_U       = nn.Parameter(torch.randn(koop_dim, koop_dim) * 0.01)
        self.A_V       = nn.Parameter(torch.randn(koop_dim, koop_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.zeros(koop_dim))
        self.film = FiLMHyperNetwork(embed_dim, koop_dim, film_hidden_dim)
    def _get_K_base(self):
        U     = cayley_transform(self.A_U)
        V     = cayley_transform(self.A_V)
        sigma = torch.clamp(F.softplus(self.log_sigma), max=self.rho_max)
        return U @ torch.diag(sigma) @ V.t()
    def get_K(self, z_global):
        K_base      = self._get_K_base()
        gamma, beta = self.film(z_global)
        return gamma * K_base.unsqueeze(0) + beta
    def forward(self, z_osc_seq, z_global, pred_steps=1):
        K = self.get_K(z_global)
        z = z_osc_seq[:, -1, :]
        for _ in range(pred_steps):
            z = torch.bmm(K, z.unsqueeze(-1)).squeeze(-1)
        return z
    def dynamics_loss(self, z_osc_seq, z_global):
        K     = self.get_K(z_global)
        z_src = z_osc_seq[:, :-1, :]
        z_tgt = z_osc_seq[:, 1:,  :]
        z_pred= torch.bmm(z_src, K.transpose(1, 2))
        return F.mse_loss(z_pred, z_tgt)
    def lyapunov_loss(self):
        K_base = self._get_K_base()
        I = torch.eye(self.d, device=K_base.device, dtype=K_base.dtype)
        return self.lam_lyap * F.relu(
            torch.norm(K_base.t() @ K_base - I, p='fro') - self.eps_lyap
        )
    @torch.no_grad()
    def spectral_radius_base(self):
        return torch.linalg.eigvals(self._get_K_base()).abs().max().item()

class HPatchKoopEncoder(nn.Module):
    def __init__(self, official_enc, patch_dec, global_pool, global_dec, koop_op, pred_steps=1):
        super().__init__()
        self.official_enc = official_enc
        self.patch_dec    = patch_dec
        self.global_pool  = global_pool
        self.global_dec   = global_dec
        self.koop_op      = koop_op
        self.pred_steps   = pred_steps
        for p in self.official_enc.parameters():
            p.requires_grad_(False)
    def encode(self, x):
        with torch.no_grad():
            z_seq = self.official_enc(x)
        z_inv_seq, z_osc_seq = self.patch_dec(z_seq)
        z_global             = self.global_pool(z_seq)
        z_trend, z_cycle     = self.global_dec(z_global)
        return z_inv_seq, z_osc_seq, z_global, z_trend, z_cycle
    def forward(self, x):
        z_inv_seq, z_osc_seq, z_global, z_trend, _ = self.encode(x)
        z_osc_pred = self.koop_op(z_osc_seq, z_global, self.pred_steps)
        z_inv_mean = z_inv_seq.mean(dim=1)
        return torch.cat([z_inv_mean, z_osc_pred, z_trend], dim=-1)

class PatchKoopDecoder(nn.Module):
    def __init__(self, input_dim, pred_len):
        super().__init__()
        hidden = max(input_dim, pred_len * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, pred_len),
        )
    def forward(self, h): return self.net(h)

# ============================================================
# 数据加载（只加载一次，两个实验共用）
# ============================================================
def load_data():
    full_loader = get_evaluation_loaders(
        cfg.data_path, cfg.batch_size,
        ratio_patches=cfg.num_patches, mask_ratio=0.7,
    )
    dataset  = full_loader.dataset
    train_df = dataset.train_df
    n_full   = len(train_df)
    n_val    = int(n_full * 0.1)
    n_tr     = n_full - n_val
    dataset.train_df = train_df[:n_tr]
    val_df           = train_df[n_tr:]
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=True, num_workers=4)
    val_dataset          = copy.copy(dataset)
    val_dataset.train_df = val_df
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=4)
    return train_loader, val_loader, dataset.test_df

# ============================================================
# 单次实验：加载指定checkpoint，训练+测试
# ============================================================
def run_one_experiment(exp_name, ckpt_path, train_loader, val_loader, test_df):
    print(f"\n{'='*65}")
    print(f"实验: {exp_name}")
    print(f"Checkpoint: {ckpt_path.split('/')[-1]}")
    print(f"{'='*65}")

    # 固定seed保证公平
    set_seed(cfg.seed)

    # 加载encoder
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    official_enc = OfficialEncoder(
        num_patches=cfg.num_patches, dim_in=cfg.context,
        kernel_size=cfg.kernel_size, embed_dim=cfg.embed_dim,
        embed_bias=True, nhead=cfg.encoder_nhead,
        num_layers=cfg.encoder_layers, jepa=True,
    ).to(cfg.device)
    official_enc.load_state_dict(ckpt["encoder"], strict=True)
    for p in official_enc.parameters():
        p.requires_grad_(False)
    official_enc.eval()

    # 构建模型（每次从零初始化，保证公平）
    patch_dec   = PatchDecoupler(cfg.embed_dim, cfg.patch_inv_dim, cfg.patch_osc_dim)
    global_pool = TemporalAttentionPool(cfg.embed_dim)
    global_dec  = GlobalDecoupler(cfg.embed_dim, cfg.global_trend_dim, cfg.global_cycle_dim)
    koop_op     = ConditionedKoopmanOperator(
        koop_dim=cfg.patch_osc_dim, embed_dim=cfg.embed_dim,
        film_hidden_dim=cfg.film_hidden_dim,
        spectral_radius_max=cfg.koop_spectral_radius,
        lyap_lambda=cfg.koop_lyap_lambda, lyap_epsilon=cfg.koop_lyap_epsilon,
    )
    h_enc   = HPatchKoopEncoder(official_enc, patch_dec, global_pool, global_dec, koop_op).to(cfg.device)
    decoder = PatchKoopDecoder(cfg.decoder_input_dim, cfg.pred_len).to(cfg.device)

    # 训练
    trainable = (list(patch_dec.parameters()) + list(global_pool.parameters()) +
                 list(global_dec.parameters()) + list(koop_op.parameters()) +
                 list(decoder.parameters()))
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_epochs, eta_min=1e-7)

    best, pat, best_ckpt = float('inf'), 0, {}
    train_log = []

    for ep in range(1, cfg.num_epochs + 1):
        h_enc.train(); decoder.train(); h_enc.official_enc.eval()
        ep_pred = ep_dyn = 0.0; nb = 0

        for ctx, tgt in train_loader:
            ctx, tgt = ctx.to(cfg.device), tgt.to(cfg.device)
            z_inv_seq, z_osc_seq, z_global, z_trend, _ = h_enc.encode(ctx)
            z_osc_pred = h_enc.koop_op(z_osc_seq, z_global, h_enc.pred_steps)
            z_inv_mean = z_inv_seq.mean(dim=1)
            h_vec      = torch.cat([z_inv_mean, z_osc_pred, z_trend], dim=-1)
            pred       = decoder(h_vec)
            L_pred = F.mse_loss(pred, tgt)
            L_dyn  = h_enc.koop_op.dynamics_loss(z_osc_seq, z_global)
            L_lyap = h_enc.koop_op.lyapunov_loss()
            loss   = L_pred + cfg.lambda_dyn * L_dyn + cfg.lambda_lyap * L_lyap
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            ep_pred += L_pred.item(); ep_dyn += L_dyn.item(); nb += 1
        sch.step()

        # 验证
        h_enc.eval(); decoder.eval()
        vl = 0.0
        with torch.no_grad():
            for ctx, tgt in val_loader:
                h_vec = h_enc(ctx.to(cfg.device))
                vl   += F.mse_loss(decoder(h_vec), tgt.to(cfg.device)).item()
        vl /= max(1, len(val_loader))
        train_log.append(vl)

        if ep % 50 == 0:
            print(f"  ep {ep:4d} | pred={ep_pred/nb:.5f} | "
                  f"dyn={ep_dyn/nb:.5f} | val={vl:.5f}")

        if vl < best - cfg.min_delta:
            best, pat = vl, 0
            best_ckpt = {k: copy.deepcopy(v.state_dict()) for k, v in {
                'patch_dec': patch_dec, 'global_pool': global_pool,
                'global_dec': global_dec, 'koop_op': koop_op, 'decoder': decoder
            }.items()}
        else:
            pat += 1
        if pat >= cfg.patience:
            print(f"  [早停] ep={ep} | best_val={best:.6f}")
            break

    # 恢复最优权重
    if best_ckpt:
        patch_dec.load_state_dict(best_ckpt['patch_dec'])
        global_pool.load_state_dict(best_ckpt['global_pool'])
        global_dec.load_state_dict(best_ckpt['global_dec'])
        koop_op.load_state_dict(best_ckpt['koop_op'])
        decoder.load_state_dict(best_ckpt['decoder'])

    # 滚动测试
    h_enc.eval(); decoder.eval()
    context, num_patches = cfg.context, cfg.num_patches
    num_steps = (len(test_df) - context * num_patches) // context
    l_mse, l_mae = [], []
    for step in range(num_steps):
        ctx = (test_df[context*step : context*num_patches + context*step]
               .reshape(num_patches, context).unsqueeze(0).to(cfg.device))
        tgt = test_df[context*num_patches + step*context :
                      context*num_patches + (step+1)*context]
        with torch.no_grad():
            pred = decoder(h_enc(ctx))
        l_mse.append(mse(pred.flatten().cpu().numpy(), tgt.numpy()))
        l_mae.append(mae(pred.flatten().cpu().numpy(), tgt.numpy()))

    result = {
        "exp_name" : exp_name,
        "MSE"      : float(np.mean(l_mse)),
        # "MSE"      : float(l_mse[-1]),  # 只取最后一个预测的MSE作为最终指标
        "MAE"      : float(np.mean(l_mae)),
        "best_val" : best,
        "converge_ep": len(train_log),
    }
    print(f"\n  ✅ {exp_name}: MSE={result['MSE']:.6f} | MAE={result['MAE']:.6f}")

    return result

# ============================================================
# 主流程
# ============================================================
def main():
    print("="*65)
    print("对比实验: 原始JEPA vs Koopman-JEPA")
    print("下游结构: H_PatchKoop（完全相同）")
    print("唯一变量: 预训练checkpoint")
    print("="*65)

    print("\n[数据加载]")
    train_loader, val_loader, test_df = load_data()
    print(f"  train={len(train_loader.dataset):,} | "
          f"test_df={len(test_df):,}")

    results = []
    for exp_name, ckpt_path in CHECKPOINTS.items():
        r = run_one_experiment(
            exp_name, ckpt_path,
            train_loader, val_loader, test_df
        )
        results.append(r)

    # ── 最终对比表格 ──────────────────────────────────────
    print("\n" + "="*65)
    print("📊 最终对比结果")
    print("="*65)
    print(f"  {'方法':<20} {'MSE':>10} {'MAE':>10} {'best_val':>12} {'收敛epoch':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    for r in results:
        marker = ""
        if len(results) == 2:
            other = [x for x in results if x != r][0]
            if r['MSE'] < other['MSE']:
                marker = " ✅"
        print(f"  {r['exp_name']:<20} {r['MSE']:>10.6f} {r['MAE']:>10.6f} "
              f"{r['best_val']:>12.6f} {r['converge_ep']:>10}{marker}")

    if len(results) == 2:
        mse_diff = results[1]['MSE'] - results[0]['MSE']
        mae_diff = results[1]['MAE'] - results[0]['MAE']
        mse_rel  = mse_diff / results[0]['MSE'] * 100
        print(f"\n  MSE变化: {mse_diff:+.6f} ({mse_rel:+.2f}%)")
        print(f"  MAE变化: {mae_diff:+.6f}")
        if mse_diff < 0:
            print(f"\n  ✅ Koopman-JEPA预训练使MSE降低了{abs(mse_rel):.2f}%")
        else:
            print(f"\n  ⚠️  Koopman-JEPA预训练未带来MSE提升")
            print(f"     原因可能是z_osc坍塌，建议修复后重新预训练")
    print("="*65)

if __name__ == "__main__":
    main()