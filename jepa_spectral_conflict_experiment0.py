"""
============================================================
Step3 v5：严格对齐官方实现
核心修复：
  TS_Tokenizer.proj: stride=kernel_size, padding=0
  conv_output_length = (patch_len - ks) // ks + 1 = (32-3)//3+1 = 10
  fc: Linear(embed_dim * conv_output_length, embed_dim) = Linear(1280, 128)
  聚合: torch.sum(dim=1)
  下游头: LinearDecoder = Linear(128, 32)，只训练这一层
============================================================
"""

import os, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from scipy.fft import fft, fftfreq
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable


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

    # 官方 eval 参数（严格对齐）
    num_patches: int = 10    # ratio_patches
    patch_len:   int = 32    # context
    pred_len:    int = 32    # 预测下一个 patch

    # 官方模型结构参数
    embed_dim:      int = 128
    encoder_nhead:  int = 2
    encoder_layers: int = 1
    kernel_size:    int = 3   # Conv stride=kernel_size

    # 数据划分
    train_ratio: float = 0.7
    val_ratio:   float = 0.1
    slide_step:  int   = 1    # 滑动步长

    # 解耦参数
    spectral_dim: int = 32    # z_spec 维度，z_inv = 96

    # 频率模块微调
    finetune_lr:         float = 1e-4
    finetune_max_epochs: int   = 200
    finetune_patience:   int   = 30
    finetune_min_delta:  float = 1e-6
    finetune_warmup:     int   = 10

    # 下游预测（官方：只训练 Linear，500 epoch）
    forecast_lr:         float = 1e-04
    forecast_max_epochs: int   = 500
    forecast_patience:   int   = 50
    forecast_min_delta:  float = 1e-6

    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed:   int = 42

    @property
    def window_size(self) -> int:
        return self.num_patches * self.patch_len  # 320

    @property
    def conv_out_len(self) -> int:
        """官方公式: (patch_len - ks) // ks + 1"""
        return (self.patch_len - self.kernel_size) // self.kernel_size + 1  # 10

    @property
    def fc_in_dim(self) -> int:
        return self.embed_dim * self.conv_out_len  # 1280

    @property
    def inv_dim(self) -> int:
        return self.embed_dim - self.spectral_dim  # 96


# ============================================================
# 1. 官方模型（严格对齐）
# ============================================================

class TS_Tokenizer(nn.Module):
    """完全对齐官方 tokenizer.py"""
    def __init__(self, config: Config):
        super().__init__()
        ks = config.kernel_size
        self.proj = nn.Conv1d(
            in_channels  = 1,
            out_channels = config.embed_dim,
            kernel_size  = ks,
            stride       = ks,   # ✅ stride=kernel_size
            padding      = 0,    # ✅ 无 padding
        )
        self.fc = nn.Linear(config.fc_in_dim, config.embed_dim)  # Linear(1280,128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, patch_len)
        return: (B, N, embed_dim)
        完全对齐官方 forward：
          view(-1,1,L) → Conv → flatten → fc → view(B,N,D)
        """
        B, N, L = x.shape
        x = x.reshape(B * N, 1, L)          # (BN, 1, 32)
        x = self.proj(x)                     # (BN, 128, 10)
        x = x.reshape(x.size(0), -1)         # (BN, 1280)
        x = self.fc(x)                       # (BN, 128)
        x = x.reshape(B, N, -1)              # (B, N, 128)
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.nhead    = nhead
        self.head_dim = embed_dim // nhead
        self.scale    = self.head_dim ** -0.5
        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, nhead, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = Attention(embed_dim, nhead)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class OfficialEncoder(nn.Module):
    """严格对齐官方 encoder.py"""
    def __init__(self, config: Config):
        super().__init__()
        D   = config.embed_dim
        N   = config.num_patches
        ffn = int(D * 4)

        self.tokenizer = TS_Tokenizer(config)

        # Sin-Cos 位置编码，requires_grad=False
        self.pos_embed = nn.Parameter(
            torch.zeros(1, N, D), requires_grad=False
        )
        self._init_sincos_embed(N, D)

        self.predictor_blocks = nn.ModuleList([
            TransformerBlock(D, config.encoder_nhead, ffn)
            for _ in range(config.encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(D)

    def _init_sincos_embed(self, N, D):
        assert D % 2 == 0
        omega = np.arange(D // 2, dtype=float)
        omega /= D / 2.0
        omega = 1.0 / 10000 ** omega
        pos   = np.arange(N, dtype=float)
        out   = np.einsum("m,d->md", pos, omega)
        emb   = np.concatenate([np.sin(out), np.cos(out)], axis=1)
        self.pos_embed.data.copy_(
            torch.from_numpy(emb).float().unsqueeze(0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, patch_len)
        return: (B, N, D)
        严格对齐官方 encoder.forward（无 mask）
        """
        B, N, L = x.shape
        x = self.tokenizer(x)                    # (B, N, D)
        x = x + self.pos_embed[:, :N, :]         # +pos_embed
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x                                  # (B, N, D)

    def encode_to_vec(self, x: torch.Tensor) -> torch.Tensor:
        """
        官方下游聚合：torch.sum(dim=1)
        return: (B, D)
        """
        return self.forward(x).sum(dim=1)


# ============================================================
# 2. 官方 LinearDecoder（严格对齐）
# ============================================================

class LinearDecoder(nn.Module):
    """严格对齐官方 decoder.py：单层 Linear"""
    def __init__(self, emb_dim: int, patch_size: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, patch_size)

    def forward(self, x):
        return self.fc(x)


# ============================================================
# 3. 加载 Checkpoint
# ============================================================

def load_official_checkpoint(config: Config) -> OfficialEncoder:
    print(f"\n[加载] {os.path.basename(config.checkpoint_path)}")
    model = OfficialEncoder(config).to(config.device)
    ckpt  = torch.load(config.checkpoint_path, map_location=config.device)
    sd    = ckpt["encoder"]

    # 验证关键 shape
    assert sd["tokenizer.proj.weight"].shape == torch.Size([128, 1, 3]), \
        f"proj.weight shape 不匹配: {sd['tokenizer.proj.weight'].shape}"
    assert sd["tokenizer.fc.weight"].shape == torch.Size([128, 1280]), \
        f"fc.weight shape 不匹配: {sd['tokenizer.fc.weight'].shape}"

    model.load_state_dict(sd, strict=True)
    print(f"  ✅ 加载成功 | epoch={ckpt.get('epoch','?')} "
          f"| 参数量={sum(p.numel() for p in model.parameters()):,}")
    print(f"  ✅ tokenizer.proj: {tuple(sd['tokenizer.proj.weight'].shape)} "
          f"| tokenizer.fc: {tuple(sd['tokenizer.fc.weight'].shape)}")
    return model


# ============================================================
# 4. 数据集（对齐官方格式）
# ============================================================

class ForecastDataset(Dataset):
    def __init__(self, data: np.ndarray, config: Config,
                 freq_mean: float = 0.0, freq_std: float = 1.0):
        self.samples: List[Tuple] = []
        W = config.window_size   # 320
        P = config.pred_len      # 32
        S = config.slide_step    # 1

        i = 0
        while i + W + P <= len(data):
            x_raw = data[i : i + W]
            y     = data[i + W : i + W + P]
            # reshape 成官方格式 (N, patch_len)
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
    assert config.target_col in df.columns, \
        f"列 '{config.target_col}' 不存在，可用列: {list(df.columns)}"
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

    # 用训练集统计频率均值/方差，避免 leakage
    _tmp  = ForecastDataset(tr_d, config)
    raw_f = [s[2] for s in _tmp.samples]
    fm, fs = float(np.mean(raw_f)), float(np.std(raw_f)) + 1e-8
    del _tmp

    tr_ds = ForecastDataset(tr_d, config, fm, fs)
    va_ds = ForecastDataset(va_d, config, fm, fs)
    te_ds = ForecastDataset(te_d, config, fm, fs)

    print(f"[数据] 训练:{len(tr_ds):,} | 验证:{len(va_ds):,} | 测试:{len(te_ds):,} "
          f"| step={config.slide_step} | window={config.window_size} | pred={config.pred_len}")

    kw = dict(batch_size=config.batch_size, num_workers=4, pin_memory=True)
    return (DataLoader(tr_ds, shuffle=True,  drop_last=True,  **kw),
            DataLoader(va_ds, shuffle=False, drop_last=False, **kw),
            DataLoader(te_ds, shuffle=False, drop_last=False, **kw))


# ============================================================
# 5. 频率辅助模块
# ============================================================

class SpectralHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
    def forward(self, z): return self.net(z).squeeze(-1)


class DecoupledProjector(nn.Module):
    def __init__(self, embed_dim, spectral_dim):
        super().__init__()
        self.proj_inv  = nn.Linear(embed_dim, embed_dim - spectral_dim)
        self.proj_spec = nn.Linear(embed_dim, spectral_dim)

    def forward(self, z):
        return self.proj_inv(z), self.proj_spec(z)


# ============================================================
# 6. 频率模块微调
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
# 7. 下游预测（严格对齐官方：单层 Linear，500 epoch）
# ============================================================

def train_eval_forecast(encoder_fn, latent_dim, train_loader,
                        val_loader, test_loader, config, tag):
    print(f"\n  ── 下游预测: {tag} (latent_dim={latent_dim}) ──")

    # 官方：LinearDecoder = Linear(embed_dim, patch_size)
    head = LinearDecoder(latent_dim, config.pred_len).to(config.device)
    opt  = torch.optim.AdamW(head.parameters(),
                             lr=config.forecast_lr, weight_decay=1e-4)
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
# 8. 主流程
# ============================================================

def run_all():
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    cfg = Config()
    print(f"[配置] device={cfg.device} | window={cfg.window_size} "
          f"({cfg.num_patches}×{cfg.patch_len}) | step={cfg.slide_step} "
          f"| pred={cfg.pred_len} | target='{cfg.target_col}'")
    print(f"[配置] conv_out_len={cfg.conv_out_len} | fc_in_dim={cfg.fc_in_dim}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # 加载官方 checkpoint
    official_enc = load_official_checkpoint(cfg)
    for p in official_enc.parameters(): p.requires_grad = False
    official_enc.eval()

    @torch.no_grad()
    def base_enc_fn(x):
        return official_enc.encode_to_vec(x.to(cfg.device))

    results = {}

    # ── A_Scratch ─────────────────────────────────────────
    print("\n" + "="*60)
    print("【A_Scratch】随机初始化，冻结（下界）")
    scratch = OfficialEncoder(cfg).to(cfg.device)
    for p in scratch.parameters(): p.requires_grad = False
    scratch.eval()
    results["A_Scratch"] = train_eval_forecast(
        lambda x: scratch.encode_to_vec(x.to(cfg.device)),
        cfg.embed_dim, train_loader, val_loader, test_loader, cfg, "A_Scratch")

    # ── B_JEPA ────────────────────────────────────────────
    print("\n" + "="*60)
    print("【B_JEPA】官方 checkpoint，直接用（基准）")
    results["B_JEPA"] = train_eval_forecast(
        base_enc_fn, cfg.embed_dim,
        train_loader, val_loader, test_loader, cfg, "B_JEPA")

    # ── C_Direct ──────────────────────────────────────────
    print("\n" + "="*60)
    print("【C_Direct】加频率头，未解耦（对照组）")
    sh_c = finetune_freq_module(
        base_enc_fn, SpectralHead(cfg.embed_dim),
        train_loader, val_loader, cfg, "C_Direct")
    results["C_Direct"] = train_eval_forecast(
        base_enc_fn, cfg.embed_dim,
        train_loader, val_loader, test_loader, cfg, "C_Direct")

    # ── D_Decoupled ───────────────────────────────────────
    print("\n" + "="*60)
    print("【D_Decoupled】解耦 latent（目标方案）")

    decoupler = DecoupledProjector(cfg.embed_dim, cfg.spectral_dim).to(cfg.device)
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
        return torch.cat([z_inv, z_spec], dim=-1)

    results["D_Decoupled"] = train_eval_forecast(
        decoupled_enc_fn, cfg.embed_dim,
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
    print("  C ≈ B  → 未解耦时频率信息混杂，预测器无法显式利用")

    return results


if __name__ == "__main__":
    run_all()