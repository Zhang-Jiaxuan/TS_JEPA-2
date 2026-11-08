"""
============================================================
Step3 v7：改用 flatten(1) 聚合，保留所有 patch 信息
核心修改：
  - encoder(x).sum(dim=1)  →  encoder(x).flatten(1)
  - latent_dim: 128  →  1280 (10 patches × 128 dim)
  - DecoupledProjector 输入维度同步更新为 1280
============================================================
"""

import os, sys, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from scipy.fft import fft, fftfreq
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, "/data_p6/zjx/TS_JEPA")
from src.models.encoder import Encoder as OfficialEncoder


# ============================================================
# 0. 配置
# ============================================================

@dataclass
class Config:
    checkpoint_path: str = (
        "/data_p6/zjx/TS_JEPA/logs/output_model/weather/"
        "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_ratio_patches_10_"
        "encoder_128_2_1_predictor_128_2_1_epoch_5000.pt"
    )
    data_path:   str = "/data_p6/zjx/TS_JEPA/data/weather/weather.csv"
    target_col:  str = "OT"

    num_patches: int = 10
    patch_len:   int = 32
    pred_len:    int = 32

    embed_dim:      int = 128
    encoder_nhead:  int = 2
    encoder_layers: int = 1
    kernel_size:    int = 3

    batch_size:          int   = 32
    forecast_lr:         float = 1e-4
    forecast_max_epochs: int   = 500
    forecast_patience:   int   = 50      # ✅ 恢复早停，避免过拟合
    forecast_min_delta:  float = 1e-6

    finetune_lr:         float = 1e-4
    finetune_max_epochs: int   = 200
    finetune_patience:   int   = 30
    finetune_min_delta:  float = 1e-6
    finetune_warmup:     int   = 10

    spectral_dim: int = 32

    train_ratio: float = 0.7
    val_ratio:   float = 0.1
    slide_step:  int   = 1

    device: str = "cuda:7" if torch.cuda.is_available() else "cpu"
    seed:   int = 42

    @property
    def window_size(self) -> int:
        return self.num_patches * self.patch_len       # 320

    @property
    def flat_dim(self) -> int:
        return self.num_patches * self.embed_dim       # ✅ 1280

    @property
    def inv_dim(self) -> int:
        return self.flat_dim - self.spectral_dim       # 1248


# ============================================================
# 1. 加载官方 Encoder
# ============================================================

def load_official_encoder(config: Config) -> OfficialEncoder:
    print(f"\n[加载] {os.path.basename(config.checkpoint_path)}")
    ckpt = torch.load(config.checkpoint_path, map_location=config.device)

    enc = OfficialEncoder(
        num_patches = config.num_patches,
        dim_in      = config.patch_len,
        kernel_size = config.kernel_size,
        embed_dim   = config.embed_dim,
        embed_bias  = True,
        nhead       = config.encoder_nhead,
        num_layers  = config.encoder_layers,
    ).to(config.device)

    enc.load_state_dict(ckpt["encoder"], strict=True)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  ✅ 加载成功 | epoch={ckpt.get('epoch','?')} | 参数量={n_params:,}")

    # 验证输出维度
    x_test = torch.randn(2, config.num_patches, config.patch_len).to(config.device)
    with torch.no_grad():
        z_test = enc(x_test).flatten(1)    # ✅ flatten
    print(f"  ✅ 验证输出: z.shape={tuple(z_test.shape)}, "
          f"mean={z_test.mean():.4f}, std={z_test.std():.4f}")
    return enc


# ============================================================
# 2. 下游预测头（输入维度改为 flat_dim=1280）
# ============================================================

class LinearDecoder(nn.Module):
    def __init__(self, emb_dim: int, patch_size: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, patch_size)

    def forward(self, x):
        return self.fc(x)


# ============================================================
# 3. 数据集（不变）
# ============================================================

class ForecastDataset(Dataset):
    def __init__(self, data: np.ndarray, config: Config,
                 freq_mean: float = 0.0, freq_std: float = 1.0):
        self.samples: List[Tuple] = []
        W = config.window_size
        P = config.pred_len
        S = config.slide_step

        i = 0
        while i + W + P <= len(data):
            x_raw = data[i : i + W]
            y     = data[i + W : i + W + P]
            x_pat = x_raw.reshape(config.num_patches, config.patch_len)
            freq_n = (self._dominant_freq(x_raw) - freq_mean) / (freq_std + 1e-8)
            self.samples.append((x_pat.copy(), y.copy(), float(freq_n)))
            i += S

    @staticmethod
    def _dominant_freq(w: np.ndarray) -> float:
        spec  = np.abs(fft(w))
        freqs = fftfreq(len(w))
        pos   = freqs > 0
        return float(freqs[pos][np.argmax(spec[pos])])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x, y, f = self.samples[idx]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(f, dtype=torch.float32))


def build_dataloaders(config: Config):
    df   = pd.read_csv(config.data_path)
    assert config.target_col in df.columns
    data = df[config.target_col].values.astype(np.float32)
    data = data[~np.isnan(data)]
    print(f"[数据] 总行数={len(data)} | 目标列='{config.target_col}'")

    n    = len(data)
    n_tr = int(n * config.train_ratio)
    n_va = int(n * config.val_ratio)
    mu   = data[:n_tr].mean()
    sg   = data[:n_tr].std() + 1e-8
    data = (data - mu) / sg

    tr_d = data[:n_tr]
    va_d = data[n_tr : n_tr + n_va]
    te_d = data[n_tr + n_va :]

    _tmp  = ForecastDataset(tr_d, config)
    raw_f = [s[2] for s in _tmp.samples]
    fm, fs = float(np.mean(raw_f)), float(np.std(raw_f)) + 1e-8
    del _tmp

    tr_ds = ForecastDataset(tr_d, config, fm, fs)
    va_ds = ForecastDataset(va_d, config, fm, fs)
    te_ds = ForecastDataset(te_d, config, fm, fs)

    print(f"[数据] 训练:{len(tr_ds):,} | 验证:{len(va_ds):,} | 测试:{len(te_ds):,}")

    kw = dict(batch_size=config.batch_size, num_workers=4, pin_memory=True)
    return (DataLoader(tr_ds, shuffle=True,  drop_last=True,  **kw),
            DataLoader(va_ds, shuffle=False, drop_last=False, **kw),
            DataLoader(te_ds, shuffle=False, drop_last=False, **kw))


# ============================================================
# 4. 频率辅助模块（输入维度改为 flat_dim=1280）
# ============================================================

class SpectralHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
    def forward(self, z): return self.net(z).squeeze(-1)


class DecoupledProjector(nn.Module):
    def __init__(self, flat_dim, spectral_dim):
        super().__init__()
        self.proj_inv  = nn.Linear(flat_dim, flat_dim - spectral_dim)
        self.proj_spec = nn.Linear(flat_dim, spectral_dim)

    def forward(self, z):
        return self.proj_inv(z), self.proj_spec(z)


# ============================================================
# 5. 频率模块微调（不变）
# ============================================================

def warmup_cosine_scheduler(opt, warmup, total):
    def fn(ep):
        if ep < warmup: return (ep + 1) / max(1, warmup)
        p = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def finetune_freq_module(encoder_fn, freq_module, train_loader,
                         val_loader, config, tag):
    print(f"\n  [频率模块微调] {tag}")
    freq_module = freq_module.to(config.device)
    opt = torch.optim.AdamW(freq_module.parameters(),
                            lr=config.finetune_lr, weight_decay=1e-4)
    sch = warmup_cosine_scheduler(opt, config.finetune_warmup,
                                  config.finetune_max_epochs)
    best, pat, best_st = float('inf'), 0, None

    for ep in range(1, config.finetune_max_epochs + 1):
        freq_module.train()
        for x, _, fl in train_loader:
            x, fl = x.to(config.device), fl.to(config.device)
            with torch.no_grad(): z = encoder_fn(x)
            loss = F.mse_loss(freq_module(z), fl)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(freq_module.parameters(), 1.0)
            opt.step()
        sch.step()

        freq_module.eval()
        vl = 0.0
        with torch.no_grad():
            for x, _, fl in val_loader:
                x, fl = x.to(config.device), fl.to(config.device)
                vl += F.mse_loss(freq_module(encoder_fn(x)), fl).item()
        vl /= max(1, len(val_loader))

        if ep % 30 == 0 or ep <= 5:
            print(f"    ep {ep:3d} | val_freq_mse={vl:.6f}")

        if vl < best - config.finetune_min_delta:
            best, pat, best_st = vl, 0, copy.deepcopy(freq_module.state_dict())
        else:
            pat += 1
        if pat >= config.finetune_patience:
            print(f"    [早停] ep={ep} | best={best:.6f}")
            break

    if best_st: freq_module.load_state_dict(best_st)
    return freq_module


# ============================================================
# 6. 下游预测训练
# ============================================================

def train_eval_forecast(encoder_fn, latent_dim, train_loader,
                        val_loader, test_loader, config, tag):
    print(f"\n  ── 下游预测: {tag} (latent_dim={latent_dim}) ──")

    head = LinearDecoder(latent_dim, config.pred_len).to(config.device)
    opt  = torch.optim.Adam(head.parameters(), lr=config.forecast_lr)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.forecast_max_epochs, eta_min=1e-6)

    best, pat, best_st = float('inf'), 0, None

    for ep in range(1, config.forecast_max_epochs + 1):
        head.train()
        for x, y, _ in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            with torch.no_grad(): z = encoder_fn(x)
            loss = F.mse_loss(head(z), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        sch.step()

        head.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                vl += F.mse_loss(head(encoder_fn(x)), y).item()
        vl /= max(1, len(val_loader))

        if ep % 50 == 0 or ep <= 3:
            print(f"    ep {ep:3d} | val_mse={vl:.6f}")

        if vl < best - config.forecast_min_delta:
            best, pat, best_st = vl, 0, copy.deepcopy(head.state_dict())
        else:
            pat += 1
        if pat >= config.forecast_patience:
            print(f"    [早停] ep={ep} | best_val_mse={best:.6f}")
            break

    if best_st: head.load_state_dict(best_st)

    head.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            preds.append(head(encoder_fn(x.to(config.device))).cpu().numpy())
            trues.append(y.numpy())

    p = np.concatenate(preds).flatten()
    t = np.concatenate(trues).flatten()
    res = {
        "MSE": float(mean_squared_error(t, p)),
        "MAE": float(mean_absolute_error(t, p)),
        "R²":  float(r2_score(t, p)),
    }
    print(f"    测试集 → MSE={res['MSE']:.4f} | MAE={res['MAE']:.4f} | R²={res['R²']:.4f}")
    return res


# ============================================================
# 7. 主流程
# ============================================================

def run_all():
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    cfg = Config()
    print(f"[配置] device={cfg.device} | window={cfg.window_size} "
          f"({cfg.num_patches}×{cfg.patch_len}) | pred={cfg.pred_len} "
          f"| flat_dim={cfg.flat_dim} | batch={cfg.batch_size} | lr={cfg.forecast_lr}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    official_enc = load_official_encoder(cfg)
    for p in official_enc.parameters(): p.requires_grad = False
    official_enc.eval()

    @torch.no_grad()
    def base_enc_fn(x):
        """✅ flatten 聚合：保留所有 patch 信息 → (B, 1280)"""
        return official_enc(x.to(cfg.device)).flatten(1)

    results = {}

    # ── A_Scratch ─────────────────────────────────────────
    print("\n" + "="*60)
    print("【A_Scratch】随机初始化，冻结（下界）")
    scratch = OfficialEncoder(
        num_patches=cfg.num_patches, dim_in=cfg.patch_len,
        kernel_size=cfg.kernel_size, embed_dim=cfg.embed_dim,
        embed_bias=True, nhead=cfg.encoder_nhead,
        num_layers=cfg.encoder_layers,
    ).to(cfg.device)
    for p in scratch.parameters(): p.requires_grad = False
    scratch.eval()
    results["A_Scratch"] = train_eval_forecast(
        lambda x: scratch(x.to(cfg.device)).flatten(1),   # ✅ flatten
        cfg.flat_dim, train_loader, val_loader, test_loader, cfg, "A_Scratch")

    # ── B_JEPA ────────────────────────────────────────────
    print("\n" + "="*60)
    print("【B_JEPA】官方 checkpoint，直接用（基准）")
    results["B_JEPA"] = train_eval_forecast(
        base_enc_fn, cfg.flat_dim,
        train_loader, val_loader, test_loader, cfg, "B_JEPA")

    # ── C_Direct ──────────────────────────────────────────
    print("\n" + "="*60)
    print("【C_Direct】加频率头，未解耦（对照组）")
    finetune_freq_module(
        base_enc_fn, SpectralHead(cfg.flat_dim),
        train_loader, val_loader, cfg, "C_Direct")
    results["C_Direct"] = train_eval_forecast(
        base_enc_fn, cfg.flat_dim,
        train_loader, val_loader, test_loader, cfg, "C_Direct")

    # ── D_Decoupled ───────────────────────────────────────
    print("\n" + "="*60)
    print("【D_Decoupled】解耦 latent（目标方案）")

    decoupler = DecoupledProjector(cfg.flat_dim, cfg.spectral_dim).to(cfg.device)
    sh_d      = SpectralHead(cfg.spectral_dim).to(cfg.device)

    class _FreqModule(nn.Module):
        def __init__(self, dec, sh):
            super().__init__()
            self.dec = dec; self.sh = sh
        def forward(self, z):
            _, z_spec = self.dec(z)
            return self.sh(z_spec)

    freq_mod_d = finetune_freq_module(
        base_enc_fn, _FreqModule(decoupler, sh_d),
        train_loader, val_loader, cfg, "D_Decoupled")
    freq_mod_d.eval()
    for p in freq_mod_d.parameters(): p.requires_grad = False

    @torch.no_grad()
    def decoupled_enc_fn(x):
        z = base_enc_fn(x)
        z_inv, z_spec = freq_mod_d.dec(z)
        return torch.cat([z_inv, z_spec], dim=-1)   # (B, 1280)

    results["D_Decoupled"] = train_eval_forecast(
        decoupled_enc_fn, cfg.flat_dim,
        train_loader, val_loader, test_loader, cfg, "D_Decoupled")

    # ── 对比表 ────────────────────────────────────────────
    bm, ba = results["B_JEPA"]["MSE"], results["B_JEPA"]["MAE"]
    print("\n" + "="*75)
    print("📊  最终对比表（测试集）—— Weather OT 列")
    print("="*75)
    print(f"{'组别':<22}│{'MSE':>8}│{'MAE':>8}│{'R²':>8}│"
          f"{'ΔMSE vs B':>12}│{'ΔMAE vs B':>12}")
    print("─"*75)
    for name, res in results.items():
        if name == "B_JEPA":
            ds, da = "  (基准)", "  (基准)"
        else:
            ds = f"{(res['MSE']-bm)/bm*100:+.1f}%"
            da = f"{(res['MAE']-ba)/ba*100:+.1f}%"
        print(f"{name:<22}│{res['MSE']:>8.4f}│{res['MAE']:>8.4f}│"
              f"{res['R²']:>8.4f}│{ds:>12}│{da:>12}")
    print("="*75)

    print("\n📌 结果解读：")
    print("  B > A  → 预训练有效，JEPA encoder 学到了有用表征 ✅")
    print("  D < B  → 解耦 latent 提升预测性能（Step3 核心结论）✅")
    print("  D < C  → 解耦优于直接加频率头（消融结论）✅")

    return results


if __name__ == "__main__":
    run_all()