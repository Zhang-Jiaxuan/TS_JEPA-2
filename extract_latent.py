#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract TS-JEPA latent representations from a pretrained checkpoint.

Usage example:
  python extract_latent.py \
    --save_dir outputs/latents/weather_ckpt5000 \
    --batch_size 32 \
    --window_size 210 \
    --num_patches 10 \
    --dim_in 21 \
    --kernel_size 3 \
    --embed_dim 128 \
    --nhead 2 \
    --num_layers 1 \
    --jepa \
    --pooling mean \
    --l2_norm
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.encoder import Encoder


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WeatherCSVDataset(Dataset):
    def __init__(self, csv_path, window_size=210, stride=1,
                 drop_cols=None, normalize=True):
        df = pd.read_csv(csv_path)

        if drop_cols is not None and len(drop_cols) > 0:
            df = df.drop(columns=drop_cols, errors="ignore")

        # 只保留数值列
        x = df.select_dtypes(include=[np.number]).values.astype(np.float32)  # [T, C]

        # 按列标准化
        if normalize:
            mean = x.mean(axis=0, keepdims=True)
            std  = x.std(axis=0,  keepdims=True) + 1e-6
            x = (x - mean) / std

        xs = []
        for i in range(0, len(x) - window_size + 1, stride):
            xs.append(x[i : i + window_size])   # [window_size, C]

        self.x = np.stack(xs, axis=0)            # [N, window_size, C]
        print(f"[Dataset] raw shape: {x.shape}, "
              f"window_size={window_size}, stride={stride}, "
              f"num_samples={len(self.x)}, C={self.x.shape[2]}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), idx   # [window_size, C], scalar


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Extract TS-JEPA latent with Encoder only")

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("--x_path", type=str,
                   default="/data_p6/zjx/TS_JEPA/data/weather/weather.csv",
                   help="输入CSV路径")
    p.add_argument("--ckpt_path", type=str,
                   default="/data_p6/zjx/TS_JEPA/logs/output_model/weather/"
                           "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_"
                           "ratio_patches_10_encoder_128_2_1_predictor_128_2_1_epoch_5000.pt",
                   help="预训练checkpoint路径")
    p.add_argument("--save_dir", type=str, default="outputs/latents",
                   help="输出目录")

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--window_size",  type=int, default=210,
                   help="滑窗长度 = num_patches * dim_in，必须与 dim_in * num_patches 对齐")
    p.add_argument("--stride",       type=int, default=1,
                   help="滑窗步长")

    # ── Model hyperparams（必须和 pretrain 一致）────────────────────────────
    p.add_argument("--num_patches",    type=int,   required=True,
                   help="patch 数量，与 pretrain 的 ratio_patches 一致")
    p.add_argument("--dim_in",         type=int,   required=True,
                   help="每个 patch 的长度 = window_size // num_patches")
    p.add_argument("--kernel_size",    type=int,   required=True,
                   help="Tokenizer Conv1d kernel size，与 pretrain 一致")
    p.add_argument("--embed_dim",      type=int,   required=True)
    p.add_argument("--embed_bias",     action="store_true")
    p.add_argument("--nhead",          type=int,   required=True)
    p.add_argument("--num_layers",     type=int,   required=True)
    p.add_argument("--mlp_ratio",      type=float, default=4.0)
    p.add_argument("--qkv_bias",       action="store_true")
    p.add_argument("--drop_rate",      type=float, default=0.0)
    p.add_argument("--attn_drop_rate", type=float, default=0.0)
    p.add_argument("--jepa",           action="store_true",
                   help="是否启用 jepa 模式（与预训练保持一致）")

    # ── Feature settings ─────────────────────────────────────────────────────
    p.add_argument("--pooling", type=str, default="mean",
                   choices=["mean", "cls", "flatten"],
                   help="patch 维度的 pooling 方式")
    p.add_argument("--channel_pooling", type=str, default="mean",
                   choices=["mean", "flatten"],
                   help="channel 维度的 pooling 方式（channel-independent 后还原）")
    p.add_argument("--l2_norm", action="store_true")

    # ── Runtime ──────────────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_config(args):
    """
    验证参数对齐：
      window_size 必须 == num_patches * dim_in
      这样每个 patch 的长度恰好是 dim_in，与 Tokenizer 初始化一致
    """
    print("\n" + "=" * 60)
    print("[Config Verification]")
    print(f"  window_size       = {args.window_size}")
    print(f"  num_patches       = {args.num_patches}")
    print(f"  dim_in (patch_len)= {args.dim_in}")
    print(f"  kernel_size       = {args.kernel_size}")
    print(f"  embed_dim         = {args.embed_dim}")

    expected_window = args.num_patches * args.dim_in
    if args.window_size != expected_window:
        print(f"\n  ⚠️  WARNING: window_size={args.window_size} ≠ "
              f"num_patches({args.num_patches}) * dim_in({args.dim_in}) "
              f"= {expected_window}")
        print(f"  ✏️  自动修正 window_size → {expected_window}")
        args.window_size = expected_window
    else:
        print(f"  ✅ window_size == num_patches * dim_in = {expected_window}")

    # 推算 Tokenizer fc in_features
    conv_out_len = (args.dim_in - args.kernel_size) // args.kernel_size + 1
    fc_in = args.embed_dim * conv_out_len
    print(f"\n  [Tokenizer]")
    print(f"  conv_output_length= {conv_out_len}  "
          f"= (dim_in({args.dim_in}) - kernel_size({args.kernel_size})) "
          f"// kernel_size + 1")
    print(f"  fc.in_features    = embed_dim({args.embed_dim}) "
          f"* conv_out_len({conv_out_len}) = {fc_in}")
    print(f"  fc.out_features   = {args.embed_dim}")
    print("=" * 60 + "\n")

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_encoder(args, device):
    model = Encoder(
        num_patches     = args.num_patches,
        dim_in          = args.dim_in,
        kernel_size     = args.kernel_size,
        embed_dim       = args.embed_dim,
        embed_bias      = args.embed_bias,
        nhead           = args.nhead,
        num_layers      = args.num_layers,
        mlp_ratio       = args.mlp_ratio,
        qkv_bias        = args.qkv_bias,
        drop_rate       = args.drop_rate,
        attn_drop_rate  = args.attn_drop_rate,
        jepa            = args.jepa,
    ).to(device)
    model.eval()
    return model


def pick_state_dict(ckpt_obj):
    """从各种格式的 checkpoint 中提取 state_dict"""
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "network"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k], k
        return ckpt_obj, "root"
    raise ValueError("无法解析 checkpoint 中的 state_dict 结构")


def extract_encoder_substate(state_dict):
    """
    从总 state_dict 中提取 encoder 权重，兼容前缀：
      - encoder.xxx
      - module.encoder.xxx
      - target_encoder.xxx
      - context_encoder.xxx
      - 兜底：直接整包加载
    """
    prefixes = [
        ("module.encoder.", "module.encoder"),
        ("encoder.",        "encoder"),
        ("target_encoder.", "target_encoder"),
        ("context_encoder.","context_encoder"),
    ]
    for prefix, label in prefixes:
        sub = {k[len(prefix):]: v
               for k, v in state_dict.items()
               if k.startswith(prefix)}
        if sub:
            print(f"[Checkpoint] 使用前缀 '{label}' 提取到 {len(sub)} 个参数")
            return sub, label

    print("[Checkpoint] 未找到已知前缀，尝试直接加载整包参数")
    return state_dict, "direct_or_unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Patch helper
# ─────────────────────────────────────────────────────────────────────────────

def to_patches(xb, num_patches, patch_len):
    """
    将 [B, T, C] 的窗口数据转为 TS-JEPA 所需的 patch 格式。

    TS-JEPA 采用 channel-independent 策略：
      每个变量独立切成 num_patches 段，每段长度 patch_len。

    流程：
      [B, T, C]
        -> permute -> [B, C, T]
        -> reshape  -> [B*C, T]
        -> reshape  -> [B*C, num_patches, patch_len]   ← 送入 Encoder

    Args:
        xb         : Tensor [B, T, C]
        num_patches: int，patch 数量
        patch_len  : int，每个 patch 长度 = dim_in

    Returns:
        patches    : Tensor [B*C, num_patches, patch_len]
        B, C       : 原始 batch size 和 channel 数（用于还原）
    """
    B, T, C = xb.shape
    total_len = num_patches * patch_len

    # 截断到整除长度（防止余数导致 reshape 失败）
    xb = xb[:, :total_len, :]                          # [B, total_len, C]

    # channel-independent: 每个变量独立
    xb = xb.permute(0, 2, 1).contiguous()             # [B, C, total_len]
    xb = xb.reshape(B * C, total_len)                  # [B*C, total_len]
    xb = xb.reshape(B * C, num_patches, patch_len)     # [B*C, num_patches, patch_len]

    return xb, B, C


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = (args.device
              if (args.device == "cpu" or torch.cuda.is_available())
              else "cpu")
    print(f"[Device] using: {device}")

    # ── 0) 参数验证与自动修正 ────────────────────────────────────────────────
    args = verify_config(args)

    # ── 1) Build model ───────────────────────────────────────────────────────
    encoder = build_encoder(args, device)
    print(f"[Model] Encoder built: "
          f"num_patches={args.num_patches}, dim_in={args.dim_in}, "
          f"kernel_size={args.kernel_size}, embed_dim={args.embed_dim}, "
          f"nhead={args.nhead}, num_layers={args.num_layers}")

    # ── 2) Load checkpoint ───────────────────────────────────────────────────
    print(f"[Checkpoint] Loading: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        print(f"[Checkpoint] Top-level keys: {list(ckpt.keys())[:10]}")

    raw_state, state_from = pick_state_dict(ckpt)
    enc_state, load_mode  = extract_encoder_substate(raw_state)

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"[Checkpoint] load_mode={load_mode}, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing keys (head):    {list(missing)[:10]}")
    if unexpected:
        print(f"  unexpected keys (head): {list(unexpected)[:10]}")

    # ── 3) Dataset & DataLoader ──────────────────────────────────────────────
    ds = WeatherCSVDataset(
        csv_path    = args.x_path,
        window_size = args.window_size,   # = num_patches * dim_in
        stride      = args.stride,
    )
    dl = DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device != "cpu"),
        drop_last   = False,
    )
    print(f"[Data] {len(ds)} samples, {len(dl)} batches, "
          f"window_size={args.window_size}")

    # ── 4) Extract ───────────────────────────────────────────────────────────
    all_z, all_ids = [], []

    for batch_idx, (xb, idx) in enumerate(dl):
        xb = xb.to(device, non_blocking=True)   # [B, window_size, C]

        # ── 切分 patches（channel-independent）──────────────────────────────
        # [B, T, C] -> [B*C, num_patches, patch_len=dim_in]
        patches, B, C = to_patches(xb, args.num_patches, args.dim_in)

        if batch_idx == 0:
            print(f"[Forward] xb shape          : {tuple(xb.shape)}")
            print(f"[Forward] patches shape      : {tuple(patches.shape)}"
                  f"  = [B({B})*C({C}), num_patches({args.num_patches}), "
                  f"patch_len({args.dim_in})]")

        # ── Encoder forward ──────────────────────────────────────────────────
        out = encoder(patches)                   # [B*C, num_patches, embed_dim]

        if batch_idx == 0:
            print(f"[Forward] encoder out shape  : {tuple(out.shape)}")

        # ── Patch-level pooling -> [B*C, embed_dim] ──────────────────────────
        if out.dim() == 3:
            if args.pooling == "mean":
                z = out.mean(dim=1)              # [B*C, embed_dim]
            elif args.pooling == "cls":
                z = out[:, 0, :]                 # [B*C, embed_dim]
            elif args.pooling == "flatten":
                z = out.reshape(out.size(0), -1) # [B*C, num_patches*embed_dim]
        elif out.dim() == 2:
            z = out                              # [B*C, embed_dim]
        else:
            raise ValueError(f"不支持的 Encoder 输出维度: {out.shape}")

        if batch_idx == 0:
            print(f"[Forward] after patch pool   : {tuple(z.shape)}")

        # ── Channel-level pooling -> [B, embed_dim] ──────────────────────────
        z = z.reshape(B, C, -1)                  # [B, C, embed_dim]
        if args.channel_pooling == "mean":
            z = z.mean(dim=1)                    # [B, embed_dim]
        elif args.channel_pooling == "flatten":
            z = z.reshape(B, -1)                 # [B, C*embed_dim]

        if batch_idx == 0:
            print(f"[Forward] after channel pool : {tuple(z.shape)}")

        # ── L2 normalize ─────────────────────────────────────────────────────
        if args.l2_norm:
            z = F.normalize(z, p=2, dim=1)

        all_z.append(z.cpu().numpy().astype(np.float32))
        all_ids.append(idx.numpy())

    z          = np.concatenate(all_z,   axis=0)   # [N, embed_dim]
    sample_ids = np.concatenate(all_ids, axis=0)   # [N]

    # ── 5) Save ──────────────────────────────────────────────────────────────
    z_path    = os.path.join(args.save_dir, "z_tsjepa.npy")
    id_path   = os.path.join(args.save_dir, "sample_ids.npy")
    meta_path = os.path.join(args.save_dir, "meta.json")

    np.save(z_path,  z)
    np.save(id_path, sample_ids)

    meta = {
        "x_path"               : args.x_path,
        "ckpt_path"            : args.ckpt_path,
        "device"               : device,
        "window_size"          : args.window_size,
        "stride"               : args.stride,
        "pooling"              : args.pooling,
        "channel_pooling"      : args.channel_pooling,
        "l2_norm"              : args.l2_norm,
        "z_shape"              : list(z.shape),
        "state_dict_from"      : state_from,
        "load_mode"            : load_mode,
        "missing_keys_count"   : len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_head"    : list(missing[:20]),
        "unexpected_keys_head" : list(unexpected[:20]),
        "model_args": {
            "num_patches"   : args.num_patches,
            "dim_in"        : args.dim_in,
            "kernel_size"   : args.kernel_size,
            "embed_dim"     : args.embed_dim,
            "embed_bias"    : args.embed_bias,
            "nhead"         : args.nhead,
            "num_layers"    : args.num_layers,
            "mlp_ratio"     : args.mlp_ratio,
            "qkv_bias"      : args.qkv_bias,
            "drop_rate"     : args.drop_rate,
            "attn_drop_rate": args.attn_drop_rate,
            "jepa"          : args.jepa,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("✅  Latent extraction finished.")
    print(f"   z        → {z_path}   shape={z.shape}")
    print(f"   ids      → {id_path}")
    print(f"   meta     → {meta_path}")
    print(f"   missing={len(missing)}, unexpected={len(unexpected)}")
    print("=" * 80)


if __name__ == "__main__":
    main()