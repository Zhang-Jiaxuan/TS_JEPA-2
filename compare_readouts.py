"""
冻结 Encoder 的读取器实验：
    2 个预训练 checkpoint × 3 种下游读取器 × 3 种训练数据比例

放在 compare_experiment.py 同一目录下运行。
复用其中的 checkpoint、Encoder、数据加载以及 H_PatchKoop 训练代码。

注意：
1. 此脚本的测试窗口协议与当前 compare_experiment.py 保持一致，
   便于直接比较；正式报告前仍需核查数据加载与测试协议。
2. 1% / 10% 采用训练集时间顺序上的前段样本，不使用验证或测试样本。
"""

import copy
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import compare_experiment as base


FRACTIONS = (0.01, 0.10, 1.00)
SEEDS = (42, 43, 44)
HEADS = ("linear", "mlp", "patchkoop")
OUTPUT_CSV = "readout_results.csv"


class FrozenReadout(nn.Module):
    def __init__(self, encoder, head_type):
        super().__init__()
        self.encoder = encoder
        self.head_type = head_type

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        input_dim = base.cfg.num_patches * base.cfg.embed_dim

        if head_type == "linear":
            self.head = nn.Linear(input_dim, base.cfg.pred_len)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(input_dim, base.cfg.embed_dim),
                nn.GELU(),
                nn.Linear(base.cfg.embed_dim, base.cfg.pred_len),
            )
        else:
            raise ValueError(f"Unknown head: {head_type}")

    def forward(self, ctx):
        # 即使外层调用 model.train()，Encoder 仍冻结且处于 eval 模式。
        with torch.no_grad():
            z_seq = self.encoder(ctx)

        if z_seq.ndim != 3:
            raise RuntimeError(
                f"Expected Encoder output (B, patches, D), got {z_seq.shape}"
            )

        expected = (base.cfg.num_patches, base.cfg.embed_dim)
        if tuple(z_seq.shape[1:]) != expected:
            raise RuntimeError(
                f"Expected Encoder output (*, {expected[0]}, {expected[1]}), "
                f"got {tuple(z_seq.shape)}"
            )

        return self.head(z_seq.flatten(start_dim=1))


def load_frozen_encoder(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=base.cfg.device)

    encoder = base.OfficialEncoder(
        num_patches=base.cfg.num_patches,
        dim_in=base.cfg.context,
        kernel_size=base.cfg.kernel_size,
        embed_dim=base.cfg.embed_dim,
        embed_bias=True,
        nhead=base.cfg.encoder_nhead,
        num_layers=base.cfg.encoder_layers,
        jepa=True,
    ).to(base.cfg.device)

    encoder.load_state_dict(ckpt["encoder"], strict=True)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad_(False)

    return encoder


def make_train_loader(full_train_loader, fraction, seed):
    dataset = full_train_loader.dataset
    n = max(1, int(len(dataset) * fraction))

    # 时间序列实验：只从训练集选样本，不碰验证集和测试集。
    subset = Subset(dataset, range(n))
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        subset,
        batch_size=base.cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        generator=generator,
    )


def evaluate_rolling(model, test_df):
    """
    与当前 compare_experiment.py 的测试协议保持一致：
    每次前进一个 patch（context 个点），预测下一个 patch。
    """
    model.eval()

    context = base.cfg.context
    input_len = context * base.cfg.num_patches

    # 当前配置中 pred_len == context；不相等时必须重新定义评测协议。
    if base.cfg.pred_len != context:
        raise ValueError(
            "当前滚动评测假设 cfg.pred_len == cfg.context；"
            "请先统一测试窗口协议。"
        )

    if len(test_df) < input_len + base.cfg.pred_len:
        raise ValueError("测试数据短于一个输入窗口加一个预测窗口")

    all_mse, all_mae = [], []

    with torch.no_grad():
        for start in range(
            0,
            len(test_df) - input_len - base.cfg.pred_len + 1,
            context,
        ):
            end = start + input_len

            ctx = (
                test_df[start:end]
                .reshape(base.cfg.num_patches, context)
                .unsqueeze(0)
                .to(base.cfg.device)
            )
            tgt = test_df[end:end + base.cfg.pred_len]
            pred = model(ctx).flatten().cpu().numpy()
            truth = tgt.numpy()

            all_mse.append(base.mse(pred, truth))
            all_mae.append(base.mae(pred, truth))

    return float(np.mean(all_mse)), float(np.mean(all_mae))


def run_simple_head(
    exp_name, ckpt_path, head_type, train_loader, val_loader, test_df, seed
):
    base.set_seed(seed)

    encoder = load_frozen_encoder(ckpt_path)
    model = FrozenReadout(encoder, head_type).to(base.cfg.device)

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=base.cfg.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=base.cfg.num_epochs,
        eta_min=1e-7,
    )

    best_val = float("inf")
    best_head = None
    patience_count = 0
    epochs_run = 0

    for epoch in range(1, base.cfg.num_epochs + 1):
        epochs_run = epoch
        model.train()
        model.encoder.eval()

        for ctx, tgt in train_loader:
            ctx = ctx.to(base.cfg.device)
            tgt = tgt.to(base.cfg.device)

            pred = model(ctx)
            loss = F.mse_loss(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        total_val = 0.0

        with torch.no_grad():
            for ctx, tgt in val_loader:
                pred = model(ctx.to(base.cfg.device))
                total_val += F.mse_loss(
                    pred, tgt.to(base.cfg.device)
                ).item()

        val_loss = total_val / max(1, len(val_loader))

        if val_loss < best_val - base.cfg.min_delta:
            best_val = val_loss
            best_head = copy.deepcopy(model.head.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= base.cfg.patience:
            break

    if best_head is not None:
        model.head.load_state_dict(best_head)

    test_mse, test_mae = evaluate_rolling(model, test_df)

    print(
        f"{exp_name:15s} | {head_type:9s} | seed={seed} | "
        f"MSE={test_mse:.6f} | MAE={test_mae:.6f}"
    )

    return {
        "checkpoint": exp_name,
        "head": head_type,
        "seed": seed,
        "MSE": test_mse,
        "MAE": test_mae,
        "best_val": best_val,
        "epochs_run": epochs_run,
    }


def main():
    full_train_loader, val_loader, test_df = base.load_data()
    rows = []

    for fraction in FRACTIONS:
        for seed in SEEDS:
            # 同一比例、同一 seed 的两种 Encoder 使用同一批训练样本，
            # DataLoader 的 generator 在每个实验前重新创建。
            for head_type in HEADS:
                for exp_name, ckpt_path in base.CHECKPOINTS.items():
                    train_loader = make_train_loader(
                        full_train_loader, fraction, seed
                    )

                    if head_type == "patchkoop":
                        base.cfg.seed = seed
                        result = base.run_one_experiment(
                            exp_name,
                            ckpt_path,
                            train_loader,
                            val_loader,
                            test_df,
                        )

                        row = {
                            "checkpoint": exp_name,
                            "head": head_type,
                            "seed": seed,
                            "MSE": result["MSE"],
                            "MAE": result["MAE"],
                            "best_val": result["best_val"],
                            "epochs_run": result["converge_ep"],
                        }
                    else:
                        row = run_simple_head(
                            exp_name,
                            ckpt_path,
                            head_type,
                            train_loader,
                            val_loader,
                            test_df,
                            seed,
                        )

                    row["fraction"] = fraction
                    rows.append(row)

                    with open(OUTPUT_CSV, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)

    print("\n汇总：MSE 均值 ± 标准差")
    for fraction in FRACTIONS:
        for head_type in HEADS:
            for exp_name in base.CHECKPOINTS:
                vals = [
                    r["MSE"]
                    for r in rows
                    if r["fraction"] == fraction
                    and r["head"] == head_type
                    and r["checkpoint"] == exp_name
                ]

                print(
                    f"fraction={fraction:.0%} | "
                    f"head={head_type:9s} | "
                    f"{exp_name:15s} | "
                    f"{np.mean(vals):.6f} ± {np.std(vals):.6f}"
                )

    print(f"\n逐次结果已保存到 {OUTPUT_CSV}")


if __name__ == "__main__":
    main()