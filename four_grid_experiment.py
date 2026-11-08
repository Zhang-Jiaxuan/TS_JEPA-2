"""
四格实验：
    {原始 JEPA Encoder, Koopman-JEPA Encoder}
    × {原始求和 + LinearDecoder, H_PatchKoop}

放置位置：TS_JEPA 仓库根目录。
不依赖 compare_experiment.py；不修改、不重新训练预训练模型。

示例：
python four_grid_experiment.py \
    --data data/weather/weather.csv \
    --jepa-ckpt logs/output_model/weather/原始检查点.pt \
    --koop-ckpt logs/output_model/weather/Koopman检查点.pt \
    --device cuda:7

注意：
1. 两个 checkpoint 必须包含键 "encoder"，并与下列 Encoder 配置兼容。
2. 默认沿用仓库“全序列标准化”的做法，以匹配已有 checkpoint；
   这不是严格的无泄漏标准化。另可用 --normalization train。
3. 指标在标准化后的 OT 序列上计算。
"""

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.encoder import Encoder
from src.models.decoder import LinearDecoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--jepa-ckpt", required=True)
    p.add_argument("--koop-ckpt", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--output", default="four_grid_results")

    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--num-patches", type=int, default=10)
    p.add_argument("--pred-len", type=int, default=32)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)

    # 必须与两个预训练 checkpoint 的 Encoder 架构一致。
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--encoder-nhead", type=int, default=2)
    p.add_argument("--encoder-layers", type=int, default=1)
    p.add_argument("--kernel-size", type=int, default=3)

    p.add_argument("--inv-dim", type=int, default=112)
    p.add_argument("--dyn-dim", type=int, default=16)
    p.add_argument("--trend-dim", type=int, default=112)
    p.add_argument("--cycle-dim", type=int, default=16)
    p.add_argument("--film-hidden-dim", type=int, default=64)
    p.add_argument("--spectral-radius-max", type=float, default=0.99)
    p.add_argument("--lyap-epsilon", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-dyn", type=float, default=0.1)
    # 本脚本只在这里乘一次 Lyapunov 损失权重。
    p.add_argument("--lambda-lyap", type=float, default=1e-3)

    p.add_argument(
        "--normalization", choices=["repository", "train"],
        default="repository",
        help="repository: 使用完整 OT 序列统计量，贴近原仓库；"
             "train: 只用训练段统计量，避免该步骤的未来信息泄漏。",
    )
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_series_splits(args):
    df = pd.read_csv(args.data, parse_dates=["date"], low_memory=False)
    if "OT" not in df.columns or "date" not in df.columns:
        raise ValueError("数据 CSV 必须含有 date 和 OT 列")
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    values = pd.to_numeric(df["OT"], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("OT 含 NaN 或无穷大，请先处理缺失值")

    n = len(values)
    n_val = int(n * 0.05)
    n_test = int(n * 0.30)
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("序列太短，不能按 65% / 5% / 30% 划分")

    reference = values if args.normalization == "repository" else values[:n_train]
    # pandas.Series.std() 默认 ddof=1，与仓库 pandas 标准差约定一致。
    mean = float(np.mean(reference, dtype=np.float64))
    std = float(np.std(reference, ddof=1, dtype=np.float64))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("标准差为零或无效，无法标准化")
    standardized = (values - mean) / std

    return (
        standardized[:n_train].copy(),
        standardized[n_train:n_train + n_val].copy(),
        standardized[n_train + n_val:].copy(),
        {"mean": mean, "std": std, "train_size": n_train,
         "val_size": n_val, "test_size": n_test},
    )


class NextPatchDataset(Dataset):
    """
    每个数据分段内部独立构造窗口，不允许窗口跨越 train/val/test 边界。
    x: (num_patches, patch_size)
    y: (pred_len,)
    """
    def __init__(self, series, num_patches, patch_size, pred_len, stride):
        self.values = torch.as_tensor(series, dtype=torch.float32)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.pred_len = pred_len
        self.input_len = num_patches * patch_size
        last_start = len(self.values) - self.input_len - pred_len
        self.starts = list(range(0, last_start + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        start = self.starts[i]
        end = start + self.input_len
        x = self.values[start:end].reshape(
            self.num_patches, self.patch_size
        )
        y = self.values[end:end + self.pred_len]
        return x, y


def make_loaders(splits, args, seed):
    datasets = [
        NextPatchDataset(
            part, args.num_patches, args.patch_size,
            args.pred_len, args.stride
        )
        for part in splits
    ]
    sizes = [len(ds) for ds in datasets]
    if min(sizes) == 0:
        raise ValueError(
            f"窗口数 train/val/test={sizes}；请缩短输入窗口或预测长度"
        )
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(
        datasets[0], batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0
    )
    val = DataLoader(
        datasets[1], batch_size=args.batch_size, shuffle=False,
        num_workers=0
    )
    test = DataLoader(
        datasets[2], batch_size=args.batch_size, shuffle=False,
        num_workers=0
    )
    return train, val, test, sizes


def load_encoder(ckpt_path, args, device):
    try:
        checkpoint = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"读取 checkpoint 失败：{ckpt_path}。如果它包含自定义对象，"
            "请先确认来源可信，再按本机 PyTorch 版本处理加载选项。"
        ) from exc
    if "encoder" not in checkpoint:
        raise KeyError(f"{ckpt_path} 不包含 'encoder' 键")

    encoder = Encoder(
        num_patches=args.num_patches,
        dim_in=args.patch_size,
        kernel_size=args.kernel_size,
        embed_dim=args.embed_dim,
        embed_bias=True,
        nhead=args.encoder_nhead,
        num_layers=args.encoder_layers,
        jepa=True,
    )
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


class OriginalHead(nn.Module):
    """原仓库下游方式：Patch 表示求和，再接 LinearDecoder。"""
    def __init__(self, args):
        super().__init__()
        self.decoder = LinearDecoder(
            emb_dim=args.embed_dim,
            patch_size=args.pred_len,
        )

    def forward(self, z_seq, return_aux=False):
        pred = self.decoder(z_seq.sum(dim=1))
        return (pred, pred.new_zeros(())) if return_aux else pred


def cayley_transform(matrix):
    d = matrix.shape[-1]
    eye = torch.eye(d, dtype=matrix.dtype, device=matrix.device)
    skew = (matrix - matrix.transpose(-1, -2)) / 2.0
    return torch.linalg.solve(eye + skew, eye - skew)


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.val_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, z_seq):
        batch = z_seq.shape[0]
        q = self.query.expand(batch, -1, -1)
        keys = self.key_proj(z_seq)
        values = self.val_proj(z_seq)
        attn = torch.softmax(
            torch.bmm(q, keys.transpose(1, 2)) * self.scale, dim=-1
        )
        return torch.bmm(attn, values).squeeze(1)


class PatchDecoupler(nn.Module):
    def __init__(self, dim, inv_dim, dyn_dim):
        super().__init__()
        self.proj_inv = nn.Linear(dim, inv_dim)
        self.proj_dyn = nn.Linear(dim, dyn_dim)

    def forward(self, z_seq):
        return self.proj_inv(z_seq), self.proj_dyn(z_seq)


class GlobalDecoupler(nn.Module):
    def __init__(self, dim, trend_dim, cycle_dim):
        super().__init__()
        self.proj_trend = nn.Linear(dim, trend_dim)
        self.proj_cycle = nn.Linear(dim, cycle_dim)

    def forward(self, z_global):
        return self.proj_trend(z_global), self.proj_cycle(z_global)


class FiLMHyperNetwork(nn.Module):
    def __init__(self, input_dim, koop_dim, hidden_dim):
        super().__init__()
        self.koop_dim = koop_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        out_dim = koop_dim * koop_dim
        self.gamma_head = nn.Linear(hidden_dim, out_dim)
        self.beta_head = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
        with torch.no_grad():
            self.gamma_head.bias.copy_(torch.eye(koop_dim).flatten())

    def forward(self, z_global):
        h = self.net(z_global)
        batch = z_global.shape[0]
        gamma = self.gamma_head(h).reshape(
            batch, self.koop_dim, self.koop_dim
        )
        beta = self.beta_head(h).reshape(
            batch, self.koop_dim, self.koop_dim
        )
        return gamma, beta


class ConditionedKoopmanOperator(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.dyn_dim
        self.d = d
        self.rho_max = args.spectral_radius_max
        self.lyap_epsilon = args.lyap_epsilon
        self.A_U = nn.Parameter(torch.randn(d, d) * 0.01)
        self.A_V = nn.Parameter(torch.randn(d, d) * 0.01)
        self.log_sigma = nn.Parameter(torch.zeros(d))
        self.film = FiLMHyperNetwork(
            args.embed_dim, d, args.film_hidden_dim
        )

    def get_K_base(self):
        u = cayley_transform(self.A_U)
        v = cayley_transform(self.A_V)
        sigma = F.softplus(self.log_sigma).clamp(max=self.rho_max)
        return u @ torch.diag(sigma) @ v.transpose(0, 1)

    def get_K(self, z_global):
        base = self.get_K_base()
        gamma, beta = self.film(z_global)
        return gamma * base.unsqueeze(0) + beta

    def forward(self, z_dyn_seq, z_global, steps=1):
        matrix = self.get_K(z_global)
        z = z_dyn_seq[:, -1, :]
        for _ in range(steps):
            z = torch.bmm(matrix, z.unsqueeze(-1)).squeeze(-1)
        return z

    def dynamics_loss(self, z_dyn_seq, z_global):
        matrix = self.get_K(z_global)
        source = z_dyn_seq[:, :-1, :]
        target = z_dyn_seq[:, 1:, :]
        predicted = torch.bmm(source, matrix.transpose(1, 2))
        return F.mse_loss(predicted, target)

    def lyapunov_penalty(self):
        """返回未加权惩罚；权重只在训练 loss 中乘一次。"""
        base = self.get_K_base()
        eye = torch.eye(self.d, dtype=base.dtype, device=base.device)
        return F.relu(
            torch.linalg.matrix_norm(
                base.transpose(0, 1) @ base - eye, ord="fro"
            ) - self.lyap_epsilon
        )


class HPatchKoopHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.inv_dim + args.dyn_dim != args.embed_dim:
            raise ValueError("inv_dim + dyn_dim 必须等于 embed_dim")
        self.patch_dec = PatchDecoupler(
            args.embed_dim, args.inv_dim, args.dyn_dim
        )
        self.global_pool = TemporalAttentionPool(args.embed_dim)
        self.global_dec = GlobalDecoupler(
            args.embed_dim, args.trend_dim, args.cycle_dim
        )
        self.koop_op = ConditionedKoopmanOperator(args)

        decoder_input_dim = (
            args.inv_dim + args.dyn_dim + args.trend_dim
        )
        hidden = max(decoder_input_dim, args.pred_len * 2)
        self.decoder = nn.Sequential(
            nn.LayerNorm(decoder_input_dim),
            nn.Linear(decoder_input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, args.pred_len),
        )

    def forward(self, z_seq, return_aux=False):
        z_inv, z_dyn = self.patch_dec(z_seq)
        z_global = self.global_pool(z_seq)
        z_trend, _ = self.global_dec(z_global)
        z_dyn_next = self.koop_op(z_dyn, z_global, steps=1)
        representation = torch.cat(
            [z_inv.mean(dim=1), z_dyn_next, z_trend], dim=-1
        )
        pred = self.decoder(representation)
        if return_aux:
            return pred, self.koop_op.dynamics_loss(z_dyn, z_global)
        return pred


@torch.no_grad()
def evaluate(encoder, head, loader, device):
    encoder.eval()
    head.eval()
    squared_sum = 0.0
    absolute_sum = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        prediction = head(encoder(x))
        squared_sum += (prediction - y).square().sum().item()
        absolute_sum += (prediction - y).abs().sum().item()
        count += y.numel()
    return squared_sum / count, absolute_sum / count

# 用本函数替换上一版 four_grid_experiment.py 中的 run_cell()。
# 其他函数、类及 main() 保持不变。

def run_cell(name, ckpt_path, head_type, seed, splits, args):
    device = torch.device(args.device)

    # 相同 seed；同一类 head 在两个 Encoder 下从相同初始化开始。
    set_seed(seed)
    train_loader, val_loader, test_loader, sizes = make_loaders(
        splits, args, seed
    )
    encoder = load_encoder(ckpt_path, args, device)

    # Encoder 的构建可能消耗随机数，因此在初始化 head 前重设 seed。
    set_seed(seed)
    head = (
        OriginalHead(args) if head_type == "original"
        else HPatchKoopHead(args)
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    val_history = []

    # 不早停、不保存 best_state：所有实验训练相同数量的 epoch。
    for epoch in range(1, args.epochs + 1):
        encoder.eval()
        head.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                z_seq = encoder(x)

            prediction, dyn_loss = head(
                z_seq, return_aux=True
            )
            loss = F.mse_loss(prediction, y)

            if head_type == "koop":
                loss = (
                    loss
                    + args.lambda_dyn * dyn_loss
                    + args.lambda_lyap
                    * head.koop_op.lyapunov_penalty()
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=1.0
            )
            optimizer.step()

        scheduler.step()

        # 仅记录验证指标，不用它早停或挑选 checkpoint。
        val_mse, _ = evaluate(
            encoder, head, val_loader, device
        )
        val_history.append(val_mse)

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(
                f"{name:13s} {head_type:8s} seed={seed} "
                f"epoch={epoch:4d}/{args.epochs} "
                f"val_mse={val_mse:.6f}",
                flush=True,
            )

    # 测试最后一个 epoch 的权重，而非历史最佳验证权重。
    test_mse, test_mae = evaluate(
        encoder, head, test_loader, device
    )

    row = {
        "pretrain": name,
        "head": head_type,
        "seed": seed,
        "trained_epochs": args.epochs,
        "final_val_mse": val_history[-1],
        "best_val_mse_diagnostic": min(val_history),
        "test_mse": test_mse,
        "test_mae": test_mae,
        "train_windows": sizes[0],
        "val_windows": sizes[1],
        "test_windows": sizes[2],
    }
    print(f"完成：{row}", flush=True)
    return row

def main():
    args = parse_args()
    if args.pred_len <= 0 or args.stride <= 0:
        raise ValueError("pred_len 和 stride 必须为正整数")
    if args.num_patches <= 0 or args.patch_size <= 0:
        raise ValueError("num_patches 和 patch_size 必须为正整数")

    for path in (args.data, args.jepa_ckpt, args.koop_ckpt):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    train, val, test, scale = load_series_splits(args)
    print(
        f"标准化方式={args.normalization}；"
        f"分段长度 train/val/test="
        f"{len(train)}/{len(val)}/{len(test)}；"
        f"mean={scale['mean']:.6f}, std={scale['std']:.6f}"
    )

    checkpoints = {
        "JEPA": args.jepa_ckpt,
        "Koopman-JEPA": args.koop_ckpt,
    }
    rows = []
    for seed in args.seeds:
        for name, ckpt_path in checkpoints.items():
            for head_type in ("original", "koop"):
                rows.append(
                    run_cell(
                        name, ckpt_path, head_type, seed,
                        (train, val, test), args
                    )
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    csv_path = output.with_suffix(".csv")
    json_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "scale": scale,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n四格实验：test MSE / MAE，均值 ± 样本标准差")
    for name in checkpoints:
        for head_type in ("original", "koop"):
            subset = [
                row for row in rows
                if row["pretrain"] == name
                and row["head"] == head_type
            ]
            mse = np.array(
                [row["test_mse"] for row in subset], dtype=float
            )
            mae = np.array(
                [row["test_mae"] for row in subset], dtype=float
            )
            ddof = 1 if len(subset) > 1 else 0
            print(
                f"{name:13s} | {head_type:8s} | "
                f"MSE {mse.mean():.6f} ± {mse.std(ddof=ddof):.6f} | "
                f"MAE {mae.mean():.6f} ± {mae.std(ddof=ddof):.6f}"
            )

    print(f"\n逐次结果已保存：{csv_path}、{json_path}")


if __name__ == "__main__":
    main()