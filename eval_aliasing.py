"""
Q3 混叠实验：eval_aliasing.py
目标：分析 TS-JEPA Encoder 对不同频率成分的表征分离能力
输出：
  - aliasing_tsne.png       (t-SNE 按频率类别着色)
  - aliasing_cosine_sim.png (Cosine Similarity 热力图)
  - aliasing_silhouette.png (Silhouette Score 对比柱状图)
  - aliasing_results.txt    (数值摘要)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, silhouette_samples
import argparse
import os
import sys
from datetime import datetime

# ── 路径配置 ──────────────────────────────────────────────
sys.path.append("/data_p6/zjx/TS_JEPA")
from src.models.encoder import Encoder

# ─────────────────────────────────────────────────────────
# 1. 参数
# ─────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim",   type=int,   default=128)
    parser.add_argument("--nhead",       type=int,   default=2)
    parser.add_argument("--num_layers",  type=int,   default=1)
    parser.add_argument("--kernel_size", type=int,   default=3)
    parser.add_argument("--num_patches", type=int,   default=10)
    parser.add_argument("--patch_size",  type=int,   default=32)
    parser.add_argument("--checkpoint",  type=str,
        default="/data_p6/zjx/TS_JEPA/logs/output_model/weather/lr_1e-07_ema_momentum_0.998_mask_ratio_0.75_ratio_patches_10_encoder_128_2_1_predictor_128_2_1_epoch_5000.pt")
    parser.add_argument("--n_samples",   type=int,   default=200)
    parser.add_argument("--mask_ratio",  type=float, default=0.75)
    parser.add_argument("--save_dir",    type=str,
        default="/data_p6/zjx/TS_JEPA/results/q3_aliasing")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
# 2. 生成受控频率信号
# ─────────────────────────────────────────────────────────
def generate_signals(n_samples, num_patches, patch_size, noise_std=0.05):
    T   = num_patches * patch_size
    t   = np.arange(T, dtype=np.float32)
    rng = np.random.default_rng(42)
    signals, labels = [], []

    for cls in range(3):
        for _ in range(n_samples):
            f_low  = rng.uniform(0.02, 0.08)
            f_high = rng.uniform(0.35, 0.49)
            phase  = rng.uniform(0, 2 * np.pi)

            if cls == 0:
                sig = np.sin(2 * np.pi * f_low * t + phase)
            elif cls == 1:
                sig = np.sin(2 * np.pi * f_high * t + phase)
            else:
                a   = rng.uniform(0.4, 0.6)
                sig = (a * np.sin(2 * np.pi * f_low  * t + phase) +
                       (1-a) * np.sin(2 * np.pi * f_high * t + rng.uniform(0, 2*np.pi)))

            sig += rng.normal(0, noise_std, T)
            sig  = sig / (np.abs(sig).max() + 1e-8)
            signals.append(sig.reshape(num_patches, patch_size))
            labels.append(cls)

    return (np.stack(signals, axis=0).astype(np.float32),
            np.array(labels, dtype=np.int64))


# ─────────────────────────────────────────────────────────
# 3. 加载 Encoder
# ─────────────────────────────────────────────────────────
def load_encoder(args, device):
    encoder = Encoder(
        num_patches=args.num_patches, dim_in=args.patch_size,
        kernel_size=args.kernel_size, embed_dim=args.embed_dim,
        embed_bias=True, nhead=args.nhead, num_layers=args.num_layers,
        jepa=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = (ckpt.get("encoder") or ckpt.get("model") or ckpt)
    encoder_state = {
        k.replace("encoder.", "").replace("module.", ""): v
        for k, v in state_dict.items()
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    print(f"[Encoder] missing={len(missing)}  unexpected={len(unexpected)}")
    encoder.eval()
    return encoder


# ─────────────────────────────────────────────────────────
# 4. 提取潜空间表征
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_representations(encoder, signals, device, batch_size=64):
    reps = []
    for i in range(0, len(signals), batch_size):
        batch = torch.tensor(signals[i:i+batch_size]).to(device)
        z     = encoder(batch)           # (B, P, D)
        reps.append(z.mean(dim=1).cpu().numpy())
    reps = np.concatenate(reps, axis=0)

    # ── 诊断信息 ──────────────────────────────────────────
    print(f"[表征诊断] shape          : {reps.shape}")
    print(f"[表征诊断] 各维度std均值  : {reps.std(axis=0).mean():.6f}")
    print(f"[表征诊断] 各维度std最大  : {reps.std(axis=0).max():.6f}")
    norms = np.linalg.norm(reps, axis=1)
    print(f"[表征诊断] L2范数 mean={norms.mean():.4f}  std={norms.std():.4f}")
    return reps


# ─────────────────────────────────────────────────────────
# 5. Silhouette Score 计算
# ─────────────────────────────────────────────────────────
def compute_silhouette(reps, labels):
    """
    计算三种度量下的 Silhouette Score：
      - euclidean  : 欧氏距离（主要指标）
      - cosine     : 余弦距离（= 1 - cosine_sim）
      - manhattan  : 曼哈顿距离（鲁棒性验证）

    同时计算每个样本的 per-sample silhouette，用于绘制蝴蝶图。
    返回 dict。
    """
    results = {}
    for metric in ["euclidean", "cosine", "manhattan"]:
        score = silhouette_score(reps, labels, metric=metric)
        per_sample = silhouette_samples(reps, labels, metric=metric)
        results[metric] = {"score": score, "per_sample": per_sample}
        print(f"[Silhouette] {metric:12s}: {score:.4f}")
    return results


# ─────────────────────────────────────────────────────────
# 6. 可视化：Silhouette 汇总图（柱状 + per-sample 分布）
# ─────────────────────────────────────────────────────────
def plot_silhouette(sil_results, labels, save_path, mask_ratio):
    names   = ["低频", "高频", "混合"]
    colors  = ["#2196F3", "#F44336", "#4CAF50"]
    metrics = ["euclidean", "cosine", "manhattan"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Silhouette Analysis  (mask_ratio={mask_ratio})", fontsize=14)

    for ax, metric in zip(axes, metrics):
        per_sample = sil_results[metric]["per_sample"]
        overall    = sil_results[metric]["score"]

        y_lower = 10
        for cls in range(3):
            cls_vals = np.sort(per_sample[labels == cls])
            size     = len(cls_vals)
            y_upper  = y_lower + size
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cls_vals,
                             alpha=0.7, color=colors[cls],
                             label=names[cls])
            ax.text(-0.05, y_lower + size / 2, names[cls], fontsize=9)
            y_lower  = y_upper + 10

        ax.axvline(overall, color="red", linestyle="--", linewidth=1.5,
                   label=f"avg={overall:.3f}")
        ax.set_title(f"{metric}\nOverall={overall:.4f}", fontsize=11)
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("Samples")
        ax.set_xlim([-0.3, 1.0])
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIZ-Sil] 保存至 {save_path}")


# ─────────────────────────────────────────────────────────
# 7. 可视化：t-SNE（着色 + 标注 silhouette）
# ─────────────────────────────────────────────────────────
def plot_tsne(reps, labels, sil_score, save_path, mask_ratio):
    print("[VIZ-5] 计算 t-SNE ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb  = tsne.fit_transform(reps)

    colors  = ["#2196F3", "#F44336", "#4CAF50"]
    markers = ["o", "s", "^"]
    names   = ["低频 (Low-freq)", "高频 (High-freq)", "混合 (Mixed)"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in range(3):
        idx = labels == cls
        ax.scatter(emb[idx, 0], emb[idx, 1],
                   c=colors[cls], marker=markers[cls],
                   label=names[cls], alpha=0.6, s=25, edgecolors="none")

    ax.set_title(
        f"t-SNE of Encoder Representations\n"
        f"mask_ratio={mask_ratio}  |  Silhouette(euclidean)={sil_score:.4f}",
        fontsize=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIZ-5] 保存至 {save_path}")


# ─────────────────────────────────────────────────────────
# 8. 可视化：Cosine Similarity 热力图
# ─────────────────────────────────────────────────────────
def plot_cosine_heatmap(reps, labels, save_path, mask_ratio, n_per_class=50):
    print("[VIZ-6] 计算 Cosine Similarity 热力图 ...")
    names = ["低频", "高频", "混合"]
    selected = np.concatenate([
        reps[labels == cls][:n_per_class] for cls in range(3)
    ], axis=0)
    normed      = normalize(selected, norm="l2")
    sim_matrix  = normed @ normed.T
    n           = n_per_class

    block_means = np.array([
        [sim_matrix[i*n:(i+1)*n, j*n:(j+1)*n].mean() for j in range(3)]
        for i in range(3)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(sim_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im0, ax=axes[0])
    for pos in [n, 2*n]:
        axes[0].axhline(pos-0.5, color="black", linewidth=1.5)
        axes[0].axvline(pos-0.5, color="black", linewidth=1.5)
    tick_pos = [n//2, n+n//2, 2*n+n//2]
    axes[0].set_xticks(tick_pos); axes[0].set_xticklabels(names)
    axes[0].set_yticks(tick_pos); axes[0].set_yticklabels(names)
    axes[0].set_title(f"Cosine Similarity Matrix\n(mask_ratio={mask_ratio})", fontsize=12)

    im1 = axes[1].imshow(block_means, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    plt.colorbar(im1, ax=axes[1])
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{block_means[i,j]:.3f}",
                         ha="center", va="center", fontsize=13,
                         color="black" if block_means[i,j] > 0.4 else "white")
    axes[1].set_xticks([0,1,2]); axes[1].set_xticklabels(names)
    axes[1].set_yticks([0,1,2]); axes[1].set_yticklabels(names)
    axes[1].set_title(f"Block-wise Mean Cosine Similarity\n(mask_ratio={mask_ratio})", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIZ-6] 保存至 {save_path}")
    return block_means


# ─────────────────────────────────────────────────────────
# 9. 数值摘要（含 Silhouette）
# ─────────────────────────────────────────────────────────
def save_summary(block_means, sil_results, save_path, mask_ratio):
    names   = ["低频", "高频", "混合"]
    metrics = ["euclidean", "cosine", "manhattan"]
    lines   = []

    lines.append("=" * 55)
    lines.append(f"  Q3 混叠实验结果摘要   mask_ratio={mask_ratio}")
    lines.append(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 55)

    # Silhouette
    lines.append("\n[Silhouette Score]  范围 [-1, 1]，越大分离越好")
    lines.append(f"  {'度量':12s}  {'全局':>8}  {'低频':>8}  {'高频':>8}  {'混合':>8}")
    lines.append("  " + "-" * 48)
    for metric in metrics:
        s   = sil_results[metric]
        per = s["per_sample"]
        cls_means = [per[np.where(
            np.concatenate([np.zeros(200), np.ones(200), np.full(200,2)]) == c
        )].mean() for c in range(3)]
        # 用 labels 计算各类均值
        all_labels = np.array([0]*200 + [1]*200 + [2]*200)
        cls_means  = [per[all_labels == c].mean() for c in range(3)]
        lines.append(
            f"  {metric:12s}  {s['score']:>8.4f}  "
            + "  ".join(f"{cls_means[c]:>8.4f}" for c in range(3))
        )

    # 混叠程度判断（基于 euclidean silhouette）
    euc_score = sil_results["euclidean"]["score"]
    lines.append(f"\n[混叠程度评估]  基于 Silhouette(euclidean)={euc_score:.4f}")
    if euc_score > 0.5:
        verdict = "✓  低混叠：编码器能有效区分不同频率成分"
    elif euc_score > 0.2:
        verdict = "⚡  中等混叠：存在一定程度的频率混淆"
    elif euc_score > 0.0:
        verdict = "⚠️  较强混叠：频率区分能力弱"
    else:
        verdict = "✗  严重混叠：高频/低频表征高度重叠"
    lines.append(f"  → {verdict}")

    # Cosine Sim 块均值
    lines.append("\n[Cosine Similarity 块均值]")
    lines.append(f"  {'':8}" + "  ".join(f"{n:>6}" for n in names))
    for i, ni in enumerate(names):
        lines.append(f"  {ni:8}" + "  ".join(f"{block_means[i,j]:>6.3f}" for j in range(3)))
    lines.append("\n  ⚠️  注意：高维空间中 cosine sim 普遍偏高（维度诅咒），")
    lines.append("      请以 Silhouette(euclidean) 为主要判断指标。")

    text = "\n".join(lines)
    print("\n" + text)
    with open(save_path, "w") as f:
        f.write(text)
    print(f"\n[摘要] 保存至 {save_path}")

    return euc_score


# ─────────────────────────────────────────────────────────
# 10. 主流程
# ─────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"{'='*55}")
    print(f"  Q3 混叠实验   mask_ratio={args.mask_ratio}   device={device}")
    print(f"{'='*55}\n")

    # Step 1: 生成信号
    print("[Step 1] 生成受控频率信号 ...")
    signals, labels = generate_signals(args.n_samples, args.num_patches, args.patch_size)
    print(f"  signals: {signals.shape}   labels: {labels.shape}\n")

    # Step 2: 加载 Encoder
    print("[Step 2] 加载 Encoder ...")
    encoder = load_encoder(args, device)
    print()

    # Step 3: 提取表征
    print("[Step 3] 提取潜空间表征 ...")
    reps = extract_representations(encoder, signals, device)
    print()

    # Step 4: Silhouette Score
    print("[Step 4] 计算 Silhouette Score ...")
    sil_results = compute_silhouette(reps, labels)
    print()

    # Step 5: 可视化
    print("[Step 5] 可视化 ...")
    prefix = os.path.join(args.save_dir, f"mask{args.mask_ratio}_{ts}")

    plot_tsne(reps, labels,
              sil_score=sil_results["euclidean"]["score"],
              save_path=f"{prefix}_tsne.png",
              mask_ratio=args.mask_ratio)

    block_means = plot_cosine_heatmap(reps, labels,
              save_path=f"{prefix}_cosine_sim.png",
              mask_ratio=args.mask_ratio)

    plot_silhouette(sil_results, labels,
              save_path=f"{prefix}_silhouette.png",
              mask_ratio=args.mask_ratio)

    save_summary(block_means, sil_results,
              save_path=f"{prefix}_results.txt",
              mask_ratio=args.mask_ratio)

    print(f"\n{'='*55}")
    print("  全部完成！")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()