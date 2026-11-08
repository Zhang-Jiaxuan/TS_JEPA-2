"""
============================================================
Step3 — 完全对齐官方源码版本
============================================================
对齐点：
  1. 使用官方 EvaluationDataLoader（相同split/归一化逻辑）
  2. 聚合方式：sum(dim=1) → (B, 128)，与官方完全一致
  3. LinearDecoder(emb_dim=128, patch_size=32)，与官方一致
  4. encoder 传入 jepa=True（与官方下游脚本一致）
  5. 学习率 lr=1e-5，与官方 config_downstream 一致
  6. 测试方式：滚动预测（step-by-step），与官方一致
  7. MSE/MAE 计算方式与官方 utils.py 一致

实验组：
  A_Scratch  — 随机初始化 encoder（下界）
  B_JEPA     — 官方预训练 checkpoint（基准）
  C_Direct   — B_JEPA + 频率辅助头（对照）
  D_Decoupled— B_JEPA + 频率解耦模块（目标方案）
============================================================
"""

import os, sys, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from scipy.fft import fft, fftfreq
from sklearn.metrics import r2_score

sys.path.insert(0, "/data_p6/zjx/TS_JEPA")
from src.models.encoder import Encoder as OfficialEncoder
from src.models.decoder import LinearDecoder
from src.data_loaders.data_loader import get_evaluation_loaders


# ============================================================
# 0. 配置（完全对齐官方）
# ============================================================

class Config:
    # ── 路径 ──────────────────────────────────────────────
    checkpoint_path = (
        "/data_p6/zjx/TS_JEPA/logs/output_model/weather/"
        "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_ratio_patches_10_"
        "encoder_128_2_1_predictor_128_2_1_epoch_5000.pt"
    )
    data_path = "/data_p6/zjx/TS_JEPA/data/weather/weather.csv"

    # ── 官方固定参数（来自官方下游脚本 hardcode）────────────
    context     = 32          # patch_size
    num_patches = 10          # context_size
    pred_len    = 32          # 预测一个 patch

    # ── encoder 参数（与 checkpoint 文件名对齐）────────────
    embed_dim      = 128
    encoder_nhead  = 2
    encoder_layers = 1
    kernel_size    = 3

    # ── 训练参数（对齐官方 config_downstream）──────────────
    batch_size  = 32
    lr          = 1e-5        # ✅ 官方用 1e-5
    num_epochs  = 500         # ✅ 官方用 500
    patience    = 50          # 早停（官方无早停，但加上更稳健）
    min_delta   = 1e-6

    # ── 频率解耦模块参数 ────────────────────────────────────
    spectral_dim = 16         # 频率子空间维度

    # ── 微调参数 ────────────────────────────────────────────
    finetune_lr         = 1e-4
    finetune_epochs     = 200
    finetune_patience   = 30
    finetune_min_delta  = 1e-6
    finetune_warmup     = 10

    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    seed   = 42

    @property
    def inv_dim(self):
        return self.embed_dim - self.spectral_dim   # 112


cfg = Config()


# ============================================================
# 1. 官方 MSE / MAE（来自 main/utils.py）
# ============================================================

def mse(pred, true):
    """与官方 utils.py 完全一致"""
    return np.mean((pred - true) ** 2)

def mae(pred, true):
    """与官方 utils.py 完全一致"""
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
        jepa        = True,          # ✅ 与官方下游脚本一致
    ).to(cfg.device)

    enc.load_state_dict(ckpt["encoder"], strict=True)
    n = sum(p.numel() for p in enc.parameters())
    print(f"  ✅ 加载成功 | epoch={ckpt.get('epoch','?')} | 参数量={n:,}")

    # 验证输出维度
    x_t = torch.randn(2, cfg.num_patches, cfg.context).to(cfg.device)
    with torch.no_grad():
        z_t = enc(x_t)                          # (2, 10, 128)
        z_s = torch.sum(z_t, dim=1)             # (2, 128) ✅ 官方聚合
    print(f"  ✅ encoder输出: patch_repr={tuple(z_t.shape)}, "
          f"summed={tuple(z_s.shape)}, "
          f"mean={z_s.mean():.4f}, std={z_s.std():.4f}")
    return enc


# ============================================================
# 3. 官方数据加载器
# ============================================================

def load_official_data(cfg: Config):
    """
    完全使用官方 EvaluationDataLoader：
      - 内部做 float 列归一化（mean/std）
      - train/val/test split: 65%/5%/30%
      - __getitem__ 返回 (context_patches[10,32], target_patch[32])
    """
    print(f"\n[数据] 使用官方 EvaluationDataLoader")
    from torch.utils.data import DataLoader, random_split

    # 官方 get_evaluation_loaders 返回的是完整 DataLoader（无 train/val 分离）
    # 我们需要拿到 dataset 后自己分 train/val 用于早停
    full_loader = get_evaluation_loaders(
        cfg.data_path,
        cfg.batch_size,
        ratio_patches=cfg.num_patches,
        mask_ratio=0.7,
    )
    dataset = full_loader.dataset

    # 官方 EvaluationDataLoader 内部已经做了 train/test split
    # dataset.train_df 和 dataset.test_df 已经分好
    # 我们从 train_df 再切出 val（最后 10%）
    train_df = dataset.train_df
    test_df  = dataset.test_df

    n_train_full = len(train_df)
    n_val  = int(n_train_full * 0.1)
    n_tr   = n_train_full - n_val

    dataset.train_df = train_df[:n_tr]
    val_df = train_df[n_tr:]

    # 构建 train DataLoader（用官方 dataset，但只用 train 部分）
    from torch.utils.data import DataLoader as DL
    train_loader = DL(dataset, batch_size=cfg.batch_size,
                      shuffle=True, drop_last=True, num_workers=4)

    # 构建 val DataLoader（临时替换 train_df）
    import copy as _copy
    val_dataset = _copy.copy(dataset)
    val_dataset.train_df = val_df
    val_loader = DL(val_dataset, batch_size=cfg.batch_size,
                    shuffle=False, num_workers=4)

    n_tr_samples  = len(train_df[:n_tr])
    n_val_samples = len(val_df)
    n_te_steps    = (len(test_df) - cfg.context * cfg.num_patches) // cfg.context
    print(f"  train_df 长度: {n_tr_samples:,} | "
          f"val_df 长度: {n_val_samples:,} | "
          f"test 滚动步数: {n_te_steps:,}")
    print(f"  归一化: 官方全列 float mean/std（已在 EvaluationDataLoader 内完成）")

    return train_loader, val_loader, dataset.test_df


# ============================================================
# 4. 频率辅助模块
# ============================================================

class SpectralHead(nn.Module):
    """预测主频标签，输入 embed_dim=128"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)


class DecoupledProjector(nn.Module):
    """
    将 sum(dim=1) 后的 128 维 latent 解耦为：
      z_inv  (112维): 时序不变特征
      z_spec ( 16维): 频率相关特征
    """
    def __init__(self, embed_dim, spectral_dim):
        super().__init__()
        self.proj_inv  = nn.Linear(embed_dim, embed_dim - spectral_dim)
        self.proj_spec = nn.Linear(embed_dim, spectral_dim)

    def forward(self, z):
        return self.proj_inv(z), self.proj_spec(z)


def compute_dominant_freq(patch_np: np.ndarray) -> float:
    """计算一段序列的主频（归一化频率）"""
    spec  = np.abs(fft(patch_np))
    freqs = fftfreq(len(patch_np))
    pos   = freqs > 0
    return float(freqs[pos][np.argmax(spec[pos])])


# ============================================================
# 5. 训练工具
# ============================================================

def warmup_cosine_lr(opt, warmup, total):
    def fn(ep):
        if ep < warmup: return (ep + 1) / max(1, warmup)
        p = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ============================================================
# 6. 下游预测训练（使用官方 DataLoader 格式）
# ============================================================

def train_forecast_head(encoder_fn, decoder, train_loader, val_loader,
                        cfg: Config, tag: str):
    """
    训练下游线性预测头。
    train_loader 每个 batch: (context_patches[B,10,32], target_patch[B,32])
    与官方下游脚本完全一致。
    """
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
                z = encoder_fn(context_patches)          # (B, 128)

            pred = decoder(z)                            # (B, 32)
            loss = F.mse_loss(pred, target_patch)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
        sch.step()

        # 验证
        decoder.eval()
        vl = 0.0
        with torch.no_grad():
            for context_patches, target_patch in val_loader:
                context_patches = context_patches.to(cfg.device)
                target_patch    = target_patch.to(cfg.device)
                z    = encoder_fn(context_patches)
                pred = decoder(z)
                vl  += F.mse_loss(pred, target_patch).item()
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
    return decoder


# ============================================================
# 7. 官方滚动测试（完全对齐官方测试逻辑）
# ============================================================

def rolling_test(encoder_fn, decoder, test_df: torch.Tensor,
                 cfg: Config, tag: str) -> dict:
    """
    完全复刻官方下游脚本的测试逻辑：
      for step in range(num_steps):
          current_context = test_df[step*32 : 10*32 + step*32].reshape(10,32)
          target = test_df[10*32 + step*32 : 10*32 + (step+1)*32]
    """
    print(f"\n  ── 滚动测试: {tag} ──")
    context     = cfg.context       # 32
    num_patches = cfg.num_patches   # 10

    num_steps = (len(test_df) - context * num_patches) // context
    print(f"    test_df 长度={len(test_df):,} | 滚动步数={num_steps:,}")

    decoder.eval()
    l_mse, l_mae = [], []

    for step in range(num_steps):
        # 构造 context（与官方完全一致）
        current_context = (
            test_df[context * step : context * num_patches + context * step]
            .reshape(num_patches, context)
            .unsqueeze(0)                    # (1, 10, 32)
        )
        target_value = test_df[
            context * num_patches + step * context :
            context * num_patches + (step + 1) * context
        ]                                    # (32,)

        current_context = current_context.to(cfg.device)

        with torch.no_grad():
            z    = encoder_fn(current_context)   # (1, 128)
            pred = decoder(z)                    # (1, 32)

        pred_np   = pred.flatten().cpu().numpy()
        target_np = target_value.numpy()

        l_mse.append(mse(pred_np, target_np))
        l_mae.append(mae(pred_np, target_np))

    result = {
        "MSE": float(np.mean(l_mse)),
        "MAE": float(np.mean(l_mae)),
    }
    print(f"    ✅ MSE={result['MSE']:.6f} | MAE={result['MAE']:.6f}")
    return result


# ============================================================
# 8. 频率模块微调
# ============================================================

def finetune_freq_module(encoder_fn, freq_module, train_loader,
                         val_loader, cfg: Config, tag: str):
    print(f"\n  [频率模块微调] {tag}")
    freq_module = freq_module.to(cfg.device)

    # 计算训练集主频标签
    print("    计算频率标签...")
    train_freqs = []
    for context_patches, _ in train_loader:
        # 用 context 的最后一个 patch 计算主频
        last_patch = context_patches[:, -1, :].numpy()   # (B, 32)
        for i in range(len(last_patch)):
            train_freqs.append(compute_dominant_freq(last_patch[i]))
    freq_arr  = np.array(train_freqs)
    freq_mean = float(freq_arr.mean())
    freq_std  = float(freq_arr.std()) + 1e-8
    print(f"    频率统计: mean={freq_mean:.4f}, std={freq_std:.4f}")

    def get_freq_label(context_patches_np):
        """归一化主频标签"""
        last = context_patches_np[:, -1, :]   # (B, 32)
        labels = np.array([compute_dominant_freq(last[i]) for i in range(len(last))])
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
    return freq_module


# ============================================================
# 9. 主流程
# ============================================================

def run_all():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print("=" * 65)
    print("TS-JEPA Step3 — 完全对齐官方源码版本")
    print(f"device={cfg.device} | embed_dim={cfg.embed_dim} | "
          f"sum(dim=1) 聚合 | lr={cfg.lr}")
    print("=" * 65)

    # ── 加载数据（官方 EvaluationDataLoader）──────────────
    train_loader, val_loader, test_df = load_official_data(cfg)

    # ── 加载官方 Encoder ───────────────────────────────────
    official_enc = load_official_encoder(cfg)
    for p in official_enc.parameters():
        p.requires_grad = False
    official_enc.eval()

    def base_enc_fn(x):
        """✅ 官方聚合：sum(dim=1) → (B, 128)"""
        return torch.sum(official_enc(x.to(cfg.device)), dim=1)

    results = {}

    # ══════════════════════════════════════════════════════
    # A_Scratch：随机初始化（下界）
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("【A_Scratch】随机初始化 encoder，冻结（下界）")
    scratch_enc = OfficialEncoder(
        num_patches=cfg.num_patches, dim_in=cfg.context,
        kernel_size=cfg.kernel_size, embed_dim=cfg.embed_dim,
        embed_bias=True, nhead=cfg.encoder_nhead,
        num_layers=cfg.encoder_layers, jepa=True,
    ).to(cfg.device)
    for p in scratch_enc.parameters():
        p.requires_grad = False
    scratch_enc.eval()

    def scratch_enc_fn(x):
        return torch.sum(scratch_enc(x.to(cfg.device)), dim=1)

    dec_A = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_A = train_forecast_head(scratch_enc_fn, dec_A,
                                train_loader, val_loader, cfg, "A_Scratch")
    results["A_Scratch"] = rolling_test(scratch_enc_fn, dec_A,
                                        test_df, cfg, "A_Scratch")

    # ══════════════════════════════════════════════════════
    # B_JEPA：官方预训练（基准）
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("【B_JEPA】官方预训练 checkpoint（基准）")
    dec_B = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_B = train_forecast_head(base_enc_fn, dec_B,
                                train_loader, val_loader, cfg, "B_JEPA")
    results["B_JEPA"] = rolling_test(base_enc_fn, dec_B,
                                     test_df, cfg, "B_JEPA")

    # ══════════════════════════════════════════════════════
    # C_Direct：加频率头但不解耦（对照组）
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("【C_Direct】频率辅助头，不解耦（对照组）")
    sh_C = SpectralHead(cfg.embed_dim)
    finetune_freq_module(base_enc_fn, sh_C,
                         train_loader, val_loader, cfg, "C_Direct")
    # 注意：C_Direct 的预测头输入仍然是原始 128 维 latent
    dec_C = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_C = train_forecast_head(base_enc_fn, dec_C,
                                train_loader, val_loader, cfg, "C_Direct")
    results["C_Direct"] = rolling_test(base_enc_fn, dec_C,
                                       test_df, cfg, "C_Direct")

    # ══════════════════════════════════════════════════════
    # D_Decoupled：频率解耦（目标方案）
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("【D_Decoupled】频率解耦 latent（目标方案）")
    decoupler = DecoupledProjector(cfg.embed_dim, cfg.spectral_dim).to(cfg.device)
    sh_D      = SpectralHead(cfg.spectral_dim).to(cfg.device)

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
        train_loader, val_loader, cfg, "D_Decoupled")
    freq_mod_D.eval()
    for p in freq_mod_D.parameters():
        p.requires_grad = False

    def decoupled_enc_fn(x):
        z = base_enc_fn(x)
        z_inv, z_spec = freq_mod_D.dec(z)
        return torch.cat([z_inv, z_spec], dim=-1)   # (B, 128)

    # D 的 decoder 输入仍是 128 维（inv_dim + spectral_dim = 128）
    dec_D = LinearDecoder(emb_dim=cfg.embed_dim, patch_size=cfg.pred_len)
    dec_D = train_forecast_head(decoupled_enc_fn, dec_D,
                                train_loader, val_loader, cfg, "D_Decoupled")
    results["D_Decoupled"] = rolling_test(decoupled_enc_fn, dec_D,
                                          test_df, cfg, "D_Decoupled")

    # ══════════════════════════════════════════════════════
    # 对比表
    # ══════════════════════════════════════════════════════
    bm = results["B_JEPA"]["MSE"]
    ba = results["B_JEPA"]["MAE"]

    print("\n" + "=" * 65)
    print("📊  最终对比表（滚动测试）—— Weather OT 列")
    print("=" * 65)
    print(f"{'组别':<22}│{'MSE':>10}│{'MAE':>10}│"
          f"{'ΔMSE vs B':>12}│{'ΔMAE vs B':>12}")
    print("─" * 65)
    for name, res in results.items():
        if name == "B_JEPA":
            ds = da = "  (基准)"
        else:
            ds = f"{(res['MSE'] - bm) / bm * 100:+.1f}%"
            da = f"{(res['MAE'] - ba) / ba * 100:+.1f}%"
        print(f"{name:<22}│{res['MSE']:>10.6f}│{res['MAE']:>10.6f}│"
              f"{ds:>12}│{da:>12}")
    print("=" * 65)

    print("\n📌 结果解读：")
    print("  B < A  → 预训练有效，JEPA 表征优于随机初始化 ✅")
    print("  D < B  → 解耦 latent 提升预测性能（Step3 核心结论）✅")
    print("  D < C  → 解耦优于直接加频率头（消融结论）✅")

    return results


if __name__ == "__main__":
    run_all()