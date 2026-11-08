"""
============================================================
Step3: 解耦 Latent 在下游预测任务上的性能验证

完整故事链：
  Step1（已完成）：发现频率信息张力
  Step2（已完成）：解耦方案消除梯度冲突，Val-JEPA 稳定收敛
  Step3（本文件）：解耦 latent 在真实预测任务上是否更好？

四组对比（控制变量：只有 latent 来源不同）：
  A_Scratch    ← 随机初始化编码器（下界，证明预训练有没有用）
  B_JEPA       ← 纯 JEPA 预训练的 latent（基准）
  C_Direct     ← JEPA + 频率辅助 loss，未解耦（梯度冲突对照）
  D_Decoupled  ← JEPA + 解耦 latent（目标方案）

关键对比：
  D vs B：解耦是否带来预测提升？
  D vs C：解耦是否比直接加频率头更好？
  B vs A：预训练本身是否有效？

架构对齐原则（严格对齐 TS-JEPA 原论文）：
  - 1D-CNN Tokenizer 将时序切成 patches
  - Transformer Encoder 只处理非 mask patches
  - Transformer Predictor 预测 mask patches 的表征
  - EMA-Encoder 编码 mask patches 作为 target
  - 掩码策略：uniform random masking on patches
  - 下游评估：冻结 encoder，只训练线性/MLP 预测头（linear probing）

数据集：Weather（与之前实验保持一致）
预测设置：输入 window_size=210，预测未来 pred_len=96 步

依赖：torch >= 1.12, numpy, pandas, sklearn, scipy
============================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from scipy.fft import fft, fftfreq
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple


# ============================================================
# 0. 全局配置
# ============================================================

@dataclass
class Config:
    # ---------- 数据 ----------
    data_path:   str   = "weather.csv"
    target_col:  int   = 0          # 使用第 0 列（单变量）
    window_size: int   = 210        # 历史输入长度
    pred_len:    int   = 96         # 预测未来步数
    train_ratio: float = 0.7
    val_ratio:   float = 0.1
    # test_ratio 自动 = 0.2

    # ---------- Patch / Tokenizer ----------
    patch_len:   int = 15           # 每个 patch 的长度（210 / 15 = 14 个 patches）
    # num_patches = window_size // patch_len = 14
    patch_stride: int = 15          # 非重叠 patches（stride == patch_len）
    tokenizer_channels: int = 64    # 1D-CNN 输出通道数（patch embedding 维度）

    # ---------- Encoder（Transformer）----------
    embed_dim:        int = 128     # Transformer d_model
    encoder_nhead:    int = 2
    encoder_layers:   int = 2
    encoder_ff_dim:   int = 256
    encoder_dropout:  float = 0.1

    # ---------- Predictor（Transformer）----------
    predictor_embed:  int = 64      # Predictor 内部 d_model（可以比 encoder 小）
    predictor_nhead:  int = 2
    predictor_layers: int = 1
    predictor_ff_dim: int = 128
    predictor_dropout: float = 0.1

    # ---------- EMA ----------
    ema_momentum: float = 0.998     # 对齐原论文命令行参数

    # ---------- 掩码 ----------
    mask_ratio: float = 0.7         # 对齐原论文命令行参数

    # ---------- 解耦专用 ----------
    spectral_dim: int = 32          # z_spectral 维度
    # z_inv 维度 = embed_dim - spectral_dim = 96
    # get_representation 输出 = embed_dim（拼接后保持一致）

    # ---------- 预训练超参数 ----------
    pretrain_lr:         float = 1e-4
    pretrain_max_epochs: int   = 200
    pretrain_patience:   int   = 20
    pretrain_min_delta:  float = 1e-5
    pretrain_warmup:     int   = 10
    lambda_spectral:     float = 0.5   # 频率辅助 loss 权重

    # ---------- 下游预测器 ----------
    freeze_encoder:       bool  = True   # 冻结预训练 encoder（linear probing）
    forecast_hidden:      int   = 256
    forecast_lr:          float = 1e-3
    forecast_max_epochs:  int   = 100
    forecast_patience:    int   = 15
    forecast_min_delta:   float = 1e-5

    # ---------- 其他 ----------
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed:   int = 42

    @property
    def num_patches(self) -> int:
        return (self.window_size - self.patch_len) // self.patch_stride + 1

    @property
    def inv_dim(self) -> int:
        return self.embed_dim - self.spectral_dim


# ============================================================
# 1. 数据集
# ============================================================

class ForecastDataset(Dataset):
    """
    每个样本：
      x          : (window_size,)  历史输入（已归一化）
      y          : (pred_len,)     未来目标
      freq_label : float           主频（归一化后），用于预训练频率辅助 loss
    """
    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        pred_len: int,
        freq_mean: float = 0.0,
        freq_std:  float = 1.0,
    ):
        self.samples: List[Tuple] = []
        total = len(data)
        i = 0
        while i + window_size + pred_len <= total:
            x = data[i : i + window_size]
            y = data[i + window_size : i + window_size + pred_len]
            freq_raw  = self._dominant_freq(x)
            freq_norm = (freq_raw - freq_mean) / (freq_std + 1e-8)
            self.samples.append((x.copy(), y.copy(), float(freq_norm)))
            i += 1   # stride=1，最大化样本量

    @staticmethod
    def _dominant_freq(window: np.ndarray) -> float:
        """
        提取主频（归一化频率，范围 (0, 0.5]）。
        与预训练实验的标签计算方式严格一致。
        """
        spectrum = np.abs(fft(window))
        freqs    = fftfreq(len(window))
        pos_mask = freqs > 0
        idx      = np.argmax(spectrum[pos_mask])
        return float(freqs[pos_mask][idx])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, freq = self.samples[idx]
        return (
            torch.tensor(x,    dtype=torch.float32),
            torch.tensor(y,    dtype=torch.float32),
            torch.tensor(freq, dtype=torch.float32),
        )


def build_dataloaders(config: Config):
    """
    读取 CSV，归一化，按 7:1:2 切分，构建 DataLoader。
    返回 (train_loader, val_loader, test_loader, freq_mean, freq_std)
    """
    df   = pd.read_csv(config.data_path)
    data = df.iloc[:, config.target_col].values.astype(np.float32)

    # 全局归一化（用训练集统计量）
    n_train = int(len(data) * config.train_ratio)
    mu      = data[:n_train].mean()
    sigma   = data[:n_train].std() + 1e-8
    data    = (data - mu) / sigma

    n_val  = int(len(data) * config.val_ratio)
    n_test = len(data) - n_train - n_val

    train_data = data[:n_train]
    val_data   = data[n_train : n_train + n_val]
    test_data  = data[n_train + n_val :]

    # 用训练集计算频率标签统计量（避免 data leakage）
    _tmp = ForecastDataset(train_data, config.window_size, config.pred_len)
    raw_freqs = [s[2] for s in _tmp.samples]   # 此时 mean=0, std=1，s[2] = raw freq
    freq_mean = float(np.mean(raw_freqs))
    freq_std  = float(np.std(raw_freqs)) + 1e-8
    del _tmp

    train_ds = ForecastDataset(train_data, config.window_size, config.pred_len, freq_mean, freq_std)
    val_ds   = ForecastDataset(val_data,   config.window_size, config.pred_len, freq_mean, freq_std)
    test_ds  = ForecastDataset(test_data,  config.window_size, config.pred_len, freq_mean, freq_std)

    print(f"[数据] 训练:{len(train_ds)} | 验证:{len(val_ds)} | 测试:{len(test_ds)}")
    print(f"[数据] 频率标签 mean={freq_mean:.5f}, std={freq_std:.5f}")

    kw = dict(batch_size=config.batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **kw)

    return train_loader, val_loader, test_loader


# ============================================================
# 2. 模型组件（严格对齐 TS-JEPA 原论文架构）
# ============================================================

# ----------------------------------------------------------
# 2.1 Tokenizer：1D-CNN，将时序切成 patch embeddings
# ----------------------------------------------------------

class PatchTokenizer(nn.Module):
    """
    对齐原论文：1D-CNN Tokenizer + 绝对 sin-cos 位置编码。
    输入：(B, window_size)
    输出：(B, num_patches, embed_dim)
    """
    def __init__(self, config: Config):
        super().__init__()
        self.patch_len    = config.patch_len
        self.patch_stride = config.patch_stride
        self.embed_dim    = config.embed_dim

        # 1D-CNN：patch_len → tokenizer_channels
        self.conv = nn.Conv1d(
            in_channels  = 1,
            out_channels = config.tokenizer_channels,
            kernel_size  = config.patch_len,
            stride       = config.patch_stride,
        )
        # 投影到 embed_dim
        self.proj = nn.Linear(config.tokenizer_channels, config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)

        # 位置编码（固定 sin-cos，对齐原论文）
        num_patches = config.num_patches
        pe = self._build_sincos_pe(num_patches, config.embed_dim)
        self.register_buffer("pe", pe)   # (1, num_patches, embed_dim)

    @staticmethod
    def _build_sincos_pe(seq_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W) → (B, num_patches, embed_dim)"""
        B = x.size(0)
        x = x.unsqueeze(1)                    # (B, 1, W)
        x = self.conv(x)                      # (B, channels, num_patches)
        x = x.transpose(1, 2)                 # (B, num_patches, channels)
        x = self.norm(self.proj(x))            # (B, num_patches, embed_dim)
        x = x + self.pe                        # 加位置编码
        return x


# ----------------------------------------------------------
# 2.2 掩码工具函数
# ----------------------------------------------------------

def random_mask_patches(
    num_patches: int,
    mask_ratio:  float,
    batch_size:  int,
    device:      torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对每个样本独立随机生成 patch 掩码。
    返回：
      mask         : (B, num_patches) bool，True = 被 mask（target）
      visible_idx  : (B, num_visible) long，可见 patch 的索引
    """
    num_mask    = max(1, int(num_patches * mask_ratio))
    num_visible = num_patches - num_mask

    # 对每个样本独立随机打乱
    noise = torch.rand(batch_size, num_patches, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)

    visible_idx = ids_shuffle[:, :num_visible]   # (B, num_visible)
    mask_idx    = ids_shuffle[:, num_visible:]   # (B, num_mask)

    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, mask_idx, True)

    return mask, visible_idx


def gather_patches(
    patches: torch.Tensor,
    idx:     torch.Tensor,
) -> torch.Tensor:
    """
    从 patches (B, N, D) 中按 idx (B, K) 收集子集。
    返回：(B, K, D)
    """
    B, K = idx.shape
    D    = patches.size(-1)
    idx_expanded = idx.unsqueeze(-1).expand(B, K, D)
    return torch.gather(patches, dim=1, index=idx_expanded)


# ----------------------------------------------------------
# 2.3 Transformer Encoder（只处理可见 patches）
# ----------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = config.embed_dim,
            nhead           = config.encoder_nhead,
            dim_feedforward = config.encoder_ff_dim,
            dropout         = config.encoder_dropout,
            batch_first     = True,
            norm_first      = True,   # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_visible, embed_dim) → (B, num_visible, embed_dim)"""
        return self.transformer(x)


# ----------------------------------------------------------
# 2.4 Transformer Predictor（预测 mask patches 的表征）
# ----------------------------------------------------------

class TransformerPredictor(nn.Module):
    """
    接收 encoder 输出的可见 patch 表征，
    加上 mask token（可学习），预测所有 mask patches 的表征。
    """
    def __init__(self, config: Config):
        super().__init__()
        self.embed_dim      = config.embed_dim
        self.predictor_embed = config.predictor_embed

        # 投影到 predictor 内部维度
        self.input_proj  = nn.Linear(config.embed_dim, config.predictor_embed)
        self.output_proj = nn.Linear(config.predictor_embed, config.embed_dim)

        # 可学习的 mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.predictor_embed))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Predictor Transformer
        pred_layer = nn.TransformerEncoderLayer(
            d_model         = config.predictor_embed,
            nhead           = config.predictor_nhead,
            dim_feedforward = config.predictor_ff_dim,
            dropout         = config.predictor_dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            pred_layer, num_layers=config.predictor_layers
        )

        # 位置编码（与 tokenizer 共享结构，但独立参数）
        self.register_buffer(
            "pe",
            PatchTokenizer._build_sincos_pe(
                config.num_patches, config.predictor_embed
            )
        )

    def forward(
        self,
        z_visible:   torch.Tensor,   # (B, num_visible, embed_dim)
        visible_idx: torch.Tensor,   # (B, num_visible)
        mask_idx:    torch.Tensor,   # (B, num_mask)
    ) -> torch.Tensor:
        """
        返回：z_pred_mask (B, num_mask, embed_dim)，对 mask patches 的预测表征
        """
        B, num_visible, _ = z_visible.shape
        num_mask = mask_idx.size(1)

        # 投影到 predictor 维度
        z_vis = self.input_proj(z_visible)   # (B, num_visible, pred_embed)

        # 加位置编码（可见部分）
        vis_pe = gather_patches(
            self.pe.expand(B, -1, -1), visible_idx
        )   # (B, num_visible, pred_embed)
        z_vis = z_vis + vis_pe

        # 构造 mask tokens + 位置编码（mask 部分）
        mask_tokens = self.mask_token.expand(B, num_mask, -1)   # (B, num_mask, pred_embed)
        mask_pe     = gather_patches(
            self.pe.expand(B, -1, -1), mask_idx
        )   # (B, num_mask, pred_embed)
        z_mask = mask_tokens + mask_pe

        # 拼接可见 + mask tokens，送入 Transformer
        z_all  = torch.cat([z_vis, z_mask], dim=1)   # (B, num_visible+num_mask, pred_embed)
        z_all  = self.transformer(z_all)

        # 取出 mask 部分的预测结果
        z_pred_mask = z_all[:, num_visible:, :]       # (B, num_mask, pred_embed)
        z_pred_mask = self.output_proj(z_pred_mask)   # (B, num_mask, embed_dim)

        return z_pred_mask


# ----------------------------------------------------------
# 2.5 频率辅助模块
# ----------------------------------------------------------

class SpectralHead(nn.Module):
    """从 patch 表征的均值池化预测主频标签（回归）"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, num_patches, D) → (B,)"""
        pooled = z.mean(dim=1)           # 均值池化
        return self.head(pooled).squeeze(-1)


class DecoupledProjector(nn.Module):
    """
    将 encoder 输出的 patch 表征解耦为：
      z_inv      : 时域不变表征（用于 JEPA loss）
      z_spectral : 频域表征（用于 spectral loss）
    """
    def __init__(self, embed_dim: int, spectral_dim: int):
        super().__init__()
        inv_dim = embed_dim - spectral_dim
        self.proj_inv      = nn.Linear(embed_dim, inv_dim)
        self.proj_spectral = nn.Linear(embed_dim, spectral_dim)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """z: (B, N, D) → z_inv (B, N, inv_dim), z_spec (B, N, spec_dim)"""
        return self.proj_inv(z), self.proj_spectral(z)


# ----------------------------------------------------------
# 2.6 完整预训练模型（统一三种 mode）
# ----------------------------------------------------------

class PretrainModel(nn.Module):
    """
    统一预训练模型，通过 mode 控制三种变体：
      'jepa'      : 纯 JEPA，无频率辅助
      'direct'    : JEPA + 频率头，共享 latent（未解耦）
      'decoupled' : JEPA + 解耦 latent + 频率头
    """
    def __init__(self, config: Config, mode: str):
        super().__init__()
        assert mode in ('jepa', 'direct', 'decoupled'), f"未知 mode: {mode}"
        self.mode   = mode
        self.config = config

        # 共享 Tokenizer
        self.tokenizer = PatchTokenizer(config)

        # Online Encoder（通过梯度更新）
        self.online_encoder = TransformerEncoder(config)

        # Target Encoder（EMA 更新，不参与梯度）
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = TransformerPredictor(config)

        # 频率辅助模块（按 mode 决定是否创建）
        if mode == 'decoupled':
            self.decoupler     = DecoupledProjector(config.embed_dim, config.spectral_dim)
            self.spectral_head = SpectralHead(config.spectral_dim)
        elif mode == 'direct':
            self.decoupler     = None
            self.spectral_head = SpectralHead(config.embed_dim)
        else:
            self.decoupler     = None
            self.spectral_head = None

    @torch.no_grad()
    def update_ema(self):
        m = self.config.ema_momentum
        for op, tp in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            tp.data.mul_(m).add_(op.data, alpha=1.0 - m)

    def forward(
        self,
        x:          torch.Tensor,   # (B, window_size)
        mask:       torch.Tensor,   # (B, num_patches) bool，True=mask
        visible_idx: torch.Tensor,  # (B, num_visible)
        mask_idx:    torch.Tensor,  # (B, num_mask)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        返回：
          z_pred   : (B, num_mask, embed_dim)  predictor 对 mask patches 的预测
          z_target : (B, num_mask, embed_dim)  EMA encoder 对 mask patches 的编码
          freq_pred: (B,) or None              频率预测（仅 direct/decoupled 模式）
        """
        # Tokenize 整个序列
        patches = self.tokenizer(x)   # (B, num_patches, embed_dim)

        # 取可见 patches → Online Encoder
        visible_patches = gather_patches(patches, visible_idx)   # (B, num_visible, D)
        z_visible       = self.online_encoder(visible_patches)   # (B, num_visible, D)

        # Predictor：从可见表征预测 mask patches
        z_pred_mask = self.predictor(z_visible, visible_idx, mask_idx)   # (B, num_mask, D)

        # EMA Encoder：编码 mask patches 作为 target
        with torch.no_grad():
            mask_patches = gather_patches(patches, mask_idx)       # (B, num_mask, D)
            z_target     = self.target_encoder(mask_patches)       # (B, num_mask, D)
            z_target     = z_target.detach()

        # 频率辅助 loss 的预测
        freq_pred = None
        if self.mode == 'decoupled':
            # 解耦：JEPA loss 只作用于 z_inv 部分
            z_pred_inv,   z_pred_spec   = self.decoupler(z_pred_mask)
            z_target_inv, _             = self.decoupler(z_target)
            # 用 z_spectral 预测频率
            # 注意：这里用 z_visible 的 spectral 投影来预测（全局频率信息）
            _, z_vis_spec = self.decoupler(z_visible)
            freq_pred     = self.spectral_head(z_vis_spec)
            # 覆盖 z_pred / z_target 为解耦后的 inv 部分
            z_pred_mask = z_pred_inv
            z_target    = z_target_inv

        elif self.mode == 'direct':
            # 直接加频率头，不解耦
            freq_pred = self.spectral_head(z_visible)

        return z_pred_mask, z_target, freq_pred

    @torch.no_grad()
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        下游任务：提取输入 x 的全局表征（均值池化）。
        
        jepa / direct  → mean_pool(z_visible)          shape: (B, embed_dim)
        decoupled      → cat(mean_pool(z_inv),
                             mean_pool(z_spec))         shape: (B, embed_dim)
                         维度与其他组相同，保证下游预测器公平对比
        """
        patches = self.tokenizer(x)   # (B, num_patches, embed_dim)
        # 下游推理时：用全部 patches（不做 mask）
        z = self.online_encoder(patches)   # (B, num_patches, embed_dim)

        if self.mode == 'decoupled':
            z_inv, z_spec = self.decoupler(z)
            z_inv_pool  = z_inv.mean(dim=1)    # (B, inv_dim)
            z_spec_pool = z_spec.mean(dim=1)   # (B, spectral_dim)
            return torch.cat([z_inv_pool, z_spec_pool], dim=-1)   # (B, embed_dim)
        else:
            return z.mean(dim=1)   # (B, embed_dim)


# ============================================================
# 3. 预训练流程
# ============================================================

def build_lr_scheduler(optimizer, warmup_epochs: int, max_epochs: int):
    """Warmup + Cosine Annealing"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_one_model(
    model:        PretrainModel,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    config:       Config,
    tag:          str,
) -> PretrainModel:
    """
    预训练单个模型，返回 best checkpoint。
    JEPA loss = 1 - cosine_similarity(z_pred, z_target)，在 patch 维度上平均。
    """
    print(f"\n{'='*55}")
    print(f"  预训练: {tag}")
    print(f"  mode={model.mode} | mask_ratio={config.mask_ratio} "
          f"| num_patches={config.num_patches}")
    print(f"{'='*55}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.pretrain_lr, weight_decay=1e-4
    )
    scheduler = build_lr_scheduler(
        optimizer, config.pretrain_warmup, config.pretrain_max_epochs
    )

    best_val_loss    = float('inf')
    patience_counter = 0
    best_state       = None

    for epoch in range(1, config.pretrain_max_epochs + 1):
        # ---- 训练 ----
        model.train()
        train_jepa_total = 0.0
        train_spec_total = 0.0
        n_batches        = 0

        for x, y, freq_label in train_loader:
            x          = x.to(config.device)
            freq_label = freq_label.to(config.device)
            B          = x.size(0)

            # 生成掩码（每个 batch 独立随机）
            mask, visible_idx = random_mask_patches(
                config.num_patches, config.mask_ratio, B, config.device
            )
            # mask_idx = 被 mask 的 patch 索引
            num_mask = mask.sum(dim=1)[0].item()   # 假设每个样本 mask 数量相同
            mask_idx = torch.where(mask)[1].view(B, -1)   # (B, num_mask)

            z_pred, z_target, freq_pred = model(x, mask, visible_idx, mask_idx)

            # JEPA loss：在 patch 维度上取均值
            # z_pred, z_target: (B, num_mask, D) 或 (B, num_mask, inv_dim)
            cos_sim  = F.cosine_similarity(z_pred, z_target, dim=-1)   # (B, num_mask)
            l_jepa   = (1.0 - cos_sim).mean()

            if freq_pred is not None:
                l_spec = F.mse_loss(freq_pred, freq_label)
                loss   = l_jepa + config.lambda_spectral * l_spec
                train_spec_total += l_spec.item()
            else:
                loss = l_jepa

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_ema()

            train_jepa_total += l_jepa.item()
            n_batches        += 1

        scheduler.step()

        # ---- 验证 ----
        model.eval()
        val_jepa_total = 0.0
        val_n          = 0

        with torch.no_grad():
            for x, y, freq_label in val_loader:
                x = x.to(config.device)
                B = x.size(0)

                mask, visible_idx = random_mask_patches(
                    config.num_patches, config.mask_ratio, B, config.device
                )
                mask_idx = torch.where(mask)[1].view(B, -1)

                z_pred, z_target, _ = model(x, mask, visible_idx, mask_idx)
                cos_sim  = F.cosine_similarity(z_pred, z_target, dim=-1)
                val_jepa_total += (1.0 - cos_sim).mean().item()
                val_n          += 1

        val_jepa = val_jepa_total / max(1, val_n)
        avg_train_jepa = train_jepa_total / max(1, n_batches)
        avg_train_spec = train_spec_total / max(1, n_batches)

        if epoch % 10 == 0 or epoch <= 5:
            spec_str = f" | train_spec={avg_train_spec:.4f}" if model.mode != 'jepa' else ""
            print(f"  Epoch {epoch:3d} | train_jepa={avg_train_jepa:.4f}"
                  f"{spec_str} | val_jepa={val_jepa:.4f}")

        # 早停（基于 val_jepa）
        if val_jepa < best_val_loss - config.pretrain_min_delta:
            best_val_loss    = val_jepa
            patience_counter = 0
            best_state       = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= config.pretrain_patience:
            print(f"  [早停] epoch={epoch} | best_val_jepa={best_val_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [完成] 加载 best checkpoint，val_jepa={best_val_loss:.4f}")

    return model


# ============================================================
# 4. 下游预测器
# ============================================================

class ForecastHead(nn.Module):
    """
    接在冻结 encoder 后的 MLP 预测头。
    输入：latent z (B, latent_dim)
    输出：未来 pred_len 步 (B, pred_len)
    """
    def __init__(self, latent_dim: int, pred_len: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, pred_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ============================================================
# 5. 下游预测训练 & 评估
# ============================================================

def train_and_eval_forecast(
    encoder_fn:   Optional[Callable],   # None → Scratch（随机 encoder）
    latent_dim:   int,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    config:       Config,
    tag:          str,
) -> Dict[str, float]:
    """
    训练下游预测头，在测试集上评估。
    
    所有组（包括 Scratch）都使用相同的策略：
      - 固定 encoder（随机初始化 or 预训练），只训练 ForecastHead
      - 这样能最干净地反映 latent 质量，而不是预测器容量
      - 参数量完全一致，对比公平
    """
    print(f"\n  ── 下游预测: {tag} ──")

    # 构建 encoder（Scratch 组使用随机初始化的 encoder，不训练）
    if encoder_fn is None:
        _scratch_model = PretrainModel(config, mode='jepa').to(config.device)
        for p in _scratch_model.parameters():
            p.requires_grad = False
        _enc_fn = lambda x: _scratch_model.get_representation(x)
    else:
        _enc_fn = encoder_fn

    head = ForecastHead(latent_dim, config.pred_len,
                        config.forecast_hidden).to(config.device)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.forecast_lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.forecast_max_epochs, eta_min=1e-6
    )

    best_val_loss    = float('inf')
    patience_counter = 0
    best_state       = None

    for epoch in range(1, config.forecast_max_epochs + 1):
        head.train()
        for x, y, _ in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            with torch.no_grad():
                z = _enc_fn(x)
            pred = head(z)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # 验证
        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                z    = _enc_fn(x)
                pred = head(z)
                val_loss += F.mse_loss(pred, y).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val_loss - config.forecast_min_delta:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state       = copy.deepcopy(head.state_dict())
        else:
            patience_counter += 1
        if patience_counter >= config.forecast_patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    # 测试集评估
    head.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(config.device)
            z = _enc_fn(x)
            pred = head(z)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.numpy())

    pred_flat = np.concatenate(all_pred).flatten()
    true_flat = np.concatenate(all_true).flatten()

    results = {
        "MSE": float(mean_squared_error(true_flat, pred_flat)),
        "MAE": float(mean_absolute_error(true_flat, pred_flat)),
        "R²":  float(r2_score(true_flat, pred_flat)),
    }
    print(f"    测试集 → MSE={results['MSE']:.4f} | "
          f"MAE={results['MAE']:.4f} | R²={results['R²']:.4f}")
    return results


# ============================================================
# 6. 主实验流程
# ============================================================

def run_all():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    config = Config()
    print(f"[配置] device={config.device} | window={config.window_size} "
          f"| pred_len={config.pred_len} | num_patches={config.num_patches}")

    train_loader, val_loader, test_loader = build_dataloaders(config)
    all_results: Dict[str, Dict] = {}

    # --------------------------------------------------------
    # 组别 A：Scratch（随机 encoder，不预训练）→ 下界
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("【组别 A】Scratch — 随机初始化 encoder，不预训练（下界）")
    all_results["A_Scratch"] = train_and_eval_forecast(
        encoder_fn   = None,
        latent_dim   = config.embed_dim,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        config       = config,
        tag          = "A_Scratch",
    )

    # --------------------------------------------------------
    # 组别 B：纯 JEPA 预训练 → 基准
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("【组别 B】纯 JEPA 预训练（基准）")
    model_b = PretrainModel(config, mode='jepa').to(config.device)
    model_b = pretrain_one_model(model_b, train_loader, val_loader, config, "B_JEPA")
    if config.freeze_encoder:
        for p in model_b.parameters():
            p.requires_grad = False

    all_results["B_JEPA"] = train_and_eval_forecast(
        encoder_fn   = lambda x: model_b.get_representation(x),
        latent_dim   = config.embed_dim,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        config       = config,
        tag          = "B_JEPA",
    )

    # --------------------------------------------------------
    # 组别 C：JEPA + 直接加频率头（未解耦）→ 梯度冲突对照
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("【组别 C】JEPA + Direct 频率头（未解耦，梯度冲突对照）")
    model_c = PretrainModel(config, mode='direct').to(config.device)
    model_c = pretrain_one_model(model_c, train_loader, val_loader, config, "C_Direct")
    if config.freeze_encoder:
        for p in model_c.parameters():
            p.requires_grad = False

    all_results["C_Direct"] = train_and_eval_forecast(
        encoder_fn   = lambda x: model_c.get_representation(x),
        latent_dim   = config.embed_dim,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        config       = config,
        tag          = "C_Direct",
    )

    # --------------------------------------------------------
    # 组别 D：JEPA + 解耦 latent + 频率头 → 目标方案
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("【组别 D】JEPA + 解耦 latent + 频率头（目标方案）")
    model_d = PretrainModel(config, mode='decoupled').to(config.device)
    model_d = pretrain_one_model(model_d, train_loader, val_loader, config, "D_Decoupled")
    if config.freeze_encoder:
        for p in model_d.parameters():
            p.requires_grad = False

    all_results["D_Decoupled"] = train_and_eval_forecast(
        encoder_fn   = lambda x: model_d.get_representation(x),
        latent_dim   = config.embed_dim,   # inv_dim + spectral_dim = embed_dim
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        config       = config,
        tag          = "D_Decoupled",
    )

    # --------------------------------------------------------
    # 打印最终对比表
    # --------------------------------------------------------
    jepa_mse = all_results["B_JEPA"]["MSE"]
    jepa_mae = all_results["B_JEPA"]["MAE"]

    print("\n" + "="*75)
    print("📊  最终对比表（测试集）")
    print("="*75)
    header = f"{'组别':<22} | {'MSE':>8} | {'MAE':>8} | {'R²':>8} | {'ΔMSE vs B':>12} | {'ΔMAE vs B':>12}"
    print(header)
    print("-"*75)

    for name, res in all_results.items():
        if name == "B_JEPA":
            delta_mse = "(基准)"
            delta_mae = "(基准)"
        else:
            dm = (res["MSE"] - jepa_mse) / jepa_mse * 100
            da = (res["MAE"] - jepa_mae) / jepa_mae * 100
            delta_mse = f"{dm:+.1f}%"
            delta_mae = f"{da:+.1f}%"
        print(f"{name:<22} | {res['MSE']:>8.4f} | {res['MAE']:>8.4f} | "
              f"{res['R²']:>8.4f} | {delta_mse:>12} | {delta_mae:>12}")

    print("="*75)
    print("\n📌 结果解读指南：")
    print("  B vs A  : ΔMSE < 0 → 预训练有效，encoder 学到了有用的表征")
    print("  D vs B  : ΔMSE < 0 → 解耦 latent 对预测有实质帮助（核心结论）")
    print("  D vs C  : ΔMSE < 0 → 解耦比直接加频率头更好（消融结论）")
    print("  C vs B  : ΔMSE > 0 → 梯度冲突损害了 latent 质量（Step2 的预测任务验证）")

    return all_results


# ============================================================
# 7. 入口
# ============================================================

if __name__ == "__main__":
    results = run_all()