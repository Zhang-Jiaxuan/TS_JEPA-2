#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
频率信息线性探针实验 + 下游任务性能验证（纯回归版）

Step 1: 回归线性探针 —— TS-JEPA latent 是否保留频率信息
Step 2: 下游任务    —— 频率信息丢失是否造成性能损害

Usage:
  python freq_probe_experiment.py \
    --z_path outputs/latents/weather_ckpt5000/z_tsjepa.npy \
    --x_path data/weather/weather.csv \
    --save_dir outputs/freq_probe \
    --window_size 210 \
    --num_patches 10 \
    --top_k_freqs 5
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("TS-JEPA Frequency Linear Probe (Regression Only)")

    p.add_argument("--z_path",      type=str, required=True,
                   help="extract_latent.py 输出的 z_tsjepa.npy 路径")
    p.add_argument("--x_path",      type=str, required=True,
                   help="原始 CSV 数据路径")
    p.add_argument("--save_dir",    type=str, default="outputs/freq_probe")
    p.add_argument("--window_size", type=int, default=210,
                   help="与 extract_latent 一致")
    p.add_argument("--stride",      type=int, default=1)
    p.add_argument("--num_patches", type=int, default=10)
    p.add_argument("--top_k_freqs", type=int, default=5,
                   help="提取前 K 个主频功率作为回归目标")
    p.add_argument("--test_size",   type=float, default=0.2)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 频率特征提取
# ─────────────────────────────────────────────────────────────────────────────

def extract_frequency_features(x_windows, top_k=5):
    """
    从时间序列窗口中提取频率特征作为探针回归目标。

    Args:
        x_windows : np.ndarray [N, T, C]
        top_k     : int  提取前 K 个主频的功率占比

    Returns:
        dict of np.ndarray，每项 shape 均为 [N] （已对 channel 取均值）
          - dominant_freq    : 主频（归一化，0~0.5）
          - spectral_entropy : 频谱熵
          - mean_freq        : 功率加权平均频率
          - top{k}_power     : 第 k 强频率的功率占比  (k=1..top_k)
    """
    N, T, C = x_windows.shape
    freqs = rfftfreq(T)   # 归一化频率 [0, 0.5]

    dominant_freq    = np.zeros((N, C), dtype=np.float32)
    spectral_entropy = np.zeros((N, C), dtype=np.float32)
    mean_freq        = np.zeros((N, C), dtype=np.float32)
    top_k_power      = np.zeros((N, C, top_k), dtype=np.float32)

    for i in range(N):
        for c in range(C):
            seg       = x_windows[i, :, c]
            fft_vals  = np.abs(rfft(seg))
            power     = fft_vals ** 2
            power_sum = power.sum() + 1e-8

            # 主频（跳过直流分量 index=0）
            dominant_idx        = np.argmax(power[1:]) + 1
            dominant_freq[i, c] = freqs[dominant_idx]

            # 频谱熵
            p_norm = power / power_sum
            p_norm_pos = p_norm[p_norm > 0]
            spectral_entropy[i, c] = -np.sum(p_norm_pos * np.log(p_norm_pos + 1e-8))

            # 功率加权平均频率
            mean_freq[i, c] = np.sum(freqs * p_norm)

            # 前 K 个频率的功率占比（跳过直流）
            sorted_idx = np.argsort(power[1:])[::-1][:top_k] + 1
            for ki, idx in enumerate(sorted_idx):
                top_k_power[i, c, ki] = power[idx] / power_sum

    # 对 channel 取均值，得到样本级标量特征
    features = {
        "dominant_freq"   : dominant_freq.mean(axis=1),       # [N]
        "spectral_entropy": spectral_entropy.mean(axis=1),    # [N]
        "mean_freq"       : mean_freq.mean(axis=1),           # [N]
    }
    for k in range(top_k):
        features[f"top{k+1}_power"] = top_k_power[:, :, k].mean(axis=1)  # [N]

    print(f"[FreqFeatures] N={N}, T={T}, C={C}, top_k={top_k}")
    for name, arr in features.items():
        print(f"  {name:20s}: range=[{arr.min():.4f}, {arr.max():.4f}]  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 回归线性探针
# ─────────────────────────────────────────────────────────────────────────────

def run_regression_probe(Z, y, label_name, test_size=0.2, seed=42):
    """
    用 Ridge 回归线性探针预测连续频率特征 y。

    Args:
        Z          : [N, D]  latent 表示
        y          : [N]     回归目标
        label_name : str

    Returns:
        dict: R², MSE, baseline_R²（均值预测）
    """
    Z_tr, Z_te, y_tr, y_te = train_test_split(
        Z, y, test_size=test_size, random_state=seed)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    pipe.fit(Z_tr, y_tr)
    y_pred = pipe.predict(Z_te)

    r2  = r2_score(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)

    # Baseline：用训练集均值预测测试集
    baseline_pred = np.full_like(y_te, y_tr.mean())
    baseline_r2   = r2_score(y_te, baseline_pred)

    print(f"  [{label_name:25s}]  R²={r2:+.4f}  "
          f"MSE={mse:.6f}  Baseline_R²={baseline_r2:+.4f}")

    return {
        "label"      : label_name,
        "R2"         : float(r2),
        "MSE"        : float(mse),
        "baseline_R2": float(baseline_r2),
        "n_train"    : int(len(Z_tr)),
        "n_test"     : int(len(Z_te)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1：频率信息线性探针
# ─────────────────────────────────────────────────────────────────────────────

def run_step1(Z, x_windows, args):
    """
    用多种频率特征作为回归目标，测试 latent Z 对频率信息的保留程度。

    判断标准（R² 越高 = 保留越多）：
      R² > 0.5  → 该频率特征保留较好
      R² < 0.3  → 该频率特征大量丢失
    """
    print("\n" + "=" * 70)
    print("STEP 1: 频率信息线性探针（回归）")
    print("=" * 70)

    freq_feats = extract_frequency_features(x_windows, top_k=args.top_k_freqs)

    print(f"\n[回归探针] Z shape={Z.shape}，"
          f"预测各频率特征（R² 越高 = 保留越多频率信息）")

    results = []
    for name, y in freq_feats.items():
        res = run_regression_probe(Z, y, name, args.test_size, args.seed)
        results.append(res)

    # 汇总
    r2_values = [r["R2"] for r in results]
    avg_r2    = float(np.mean(r2_values))
    good      = [r["label"] for r in results if r["R2"] > 0.5]
    poor      = [r["label"] for r in results if r["R2"] < 0.3]

    print(f"\n[Step 1 Summary]")
    print(f"  平均 R²         = {avg_r2:.4f}")
    print(f"  保留较好 (>0.5) : {good if good else '无'}")
    print(f"  丢失严重 (<0.3) : {poor if poor else '无'}")

    freq_info_lost = len(poor) > 0
    print(f"\n  结论: 频率信息"
          f"{'❌ 存在丢失（' + str(len(poor)) + ' 项 R²<0.3）' if freq_info_lost else '✅ 整体保留较好'}")

    return results, freq_feats, freq_info_lost


# ─────────────────────────────────────────────────────────────────────────────
# Step 2：下游任务 —— 频率信息丢失的性能影响
# ─────────────────────────────────────────────────────────────────────────────

def run_step2(Z, x_windows, freq_feats, args):
    """
    下游任务：预测窗口后 1/4 段的均值（模拟短期预测）。

    对比三种特征：
      (a) TS-JEPA latent Z          ← 被评估对象
      (b) 手工频率特征（上界参考）    ← 频率信息充分时的性能上界
      (c) 时域统计特征（baseline）   ← 不含频率信息的基线

    若 gap(b - a) 显著，说明频率信息丢失造成了性能损害。
    """
    print("\n" + "=" * 70)
    print("STEP 2: 下游任务 —— 短期预测（频率信息丢失的性能影响）")
    print("=" * 70)

    N, T, C = x_windows.shape
    quarter = T // 4

    # 预测目标：窗口后 1/4 段各 channel 的均值，再对 channel 取均值
    y_target = x_windows[:, -quarter:, :].mean(axis=(1, 2))   # [N]
    print(f"  预测目标: 窗口后 {quarter} 步均值，shape={y_target.shape}")

    results = {}

    # ── (a) TS-JEPA latent Z ─────────────────────────────────────────────────
    print("\n[方案 a] TS-JEPA latent Z → 线性预测")
    res_a = run_regression_probe(Z, y_target, "TSJEPA_latent",
                                 args.test_size, args.seed)
    results["tsjepa_latent"] = res_a

    # ── (b) 手工频率特征（上界）──────────────────────────────────────────────
    print("\n[方案 b] 手工频率特征 → 线性预测（频率信息充分时的上界）")
    freq_vec = np.stack(list(freq_feats.values()), axis=1)   # [N, n_freq_feats]
    res_b = run_regression_probe(freq_vec, y_target,
                                 "HandcraftedFreqFeats",
                                 args.test_size, args.seed)
    results["handcrafted_freq"] = res_b

    # ── (c) 时域统计特征（baseline）──────────────────────────────────────────
    print("\n[方案 c] 时域统计特征 → 线性预测（不含频率信息的 baseline）")
    time_vec = np.concatenate([
        x_windows.mean(axis=1),                   # [N, C] 均值
        x_windows.std(axis=1),                    # [N, C] 标准差
        x_windows.max(axis=1) - x_windows.min(axis=1),  # [N, C] 极差
    ], axis=1)                                     # [N, 3C]
    res_c = run_regression_probe(time_vec, y_target,
                                 "TimeDomainStats",
                                 args.test_size, args.seed)
    results["time_domain_stats"] = res_c

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    gap_vs_freq = res_b["R2"] - res_a["R2"]
    gap_vs_time = res_a["R2"] - res_c["R2"]

    print(f"\n[Step 2 Summary] 预测任务 R² 对比：")
    print(f"  (a) TS-JEPA latent Z  R² = {res_a['R2']:+.4f}")
    print(f"  (b) 手工频率特征       R² = {res_b['R2']:+.4f}")
    print(f"  (c) 时域统计特征       R² = {res_c['R2']:+.4f}")
    print(f"\n  gap(b - a) = {gap_vs_freq:+.4f}  "
          f"({'⚠️  频率信息丢失造成显著性能损害' if gap_vs_freq > 0.05 else '差距不显著'})")
    print(f"  gap(a - c) = {gap_vs_time:+.4f}  "
          f"({'✅ TS-JEPA 优于纯时域特征' if gap_vs_time > 0.02 else '⚠️  TS-JEPA 未超越时域 baseline'})")

    results["gap_vs_freq"] = float(gap_vs_freq)
    results["gap_vs_time"] = float(gap_vs_time)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 加载原始数据
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_windows(x_path, window_size, stride=1):
    df = pd.read_csv(x_path)
    x  = df.select_dtypes(include=[np.number]).values.astype(np.float32)

    mean = x.mean(axis=0, keepdims=True)
    std  = x.std(axis=0,  keepdims=True) + 1e-6
    x    = (x - mean) / std

    xs = []
    for i in range(0, len(x) - window_size + 1, stride):
        xs.append(x[i : i + window_size])
    x_windows = np.stack(xs, axis=0)
    print(f"[RawData] x_windows shape: {x_windows.shape}")
    return x_windows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)

    # ── 加载 latent Z ─────────────────────────────────────────────────────────
    print(f"[Load] latent Z: {args.z_path}")
    Z = np.load(args.z_path)                         # [N, D]
    print(f"[Load] Z shape : {Z.shape}")

    # ── 加载原始窗口数据 ──────────────────────────────────────────────────────
    x_windows = load_raw_windows(args.x_path, args.window_size, args.stride)

    # 对齐样本数
    N = min(len(Z), len(x_windows))
    Z, x_windows = Z[:N], x_windows[:N]
    print(f"[Align] 使用 {N} 个样本，Z={Z.shape}，x_windows={x_windows.shape}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1
    # ════════════════════════════════════════════════════════════════════════
    step1_results, freq_feats, freq_info_lost = run_step1(Z, x_windows, args)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2
    # ════════════════════════════════════════════════════════════════════════
    step2_results = run_step2(Z, x_windows, freq_feats, args)

    # ════════════════════════════════════════════════════════════════════════
    # 最终结论
    # ════════════════════════════════════════════════════════════════════════
    avg_r2_step1  = float(np.mean([r["R2"] for r in step1_results]))
    gap_vs_freq   = step2_results["gap_vs_freq"]
    gap_vs_time   = step2_results["gap_vs_time"]
    conclusion    = freq_info_lost and (gap_vs_freq > 0.05)

    print("\n" + "=" * 70)
    print("📊 最终研究结论")
    print("=" * 70)
    print(f"  Step 1 | 平均频率回归 R²        = {avg_r2_step1:.4f}")
    print(f"  Step 2 | 预测 gap(频率 - JEPA)  = {gap_vs_freq:+.4f}")
    print(f"  Step 2 | 预测 gap(JEPA - 时域)  = {gap_vs_time:+.4f}")
    print(f"\n  🔬 研究问题是否成立: "
          f"{'✅ 是 —— TS-JEPA latent 丢失频率信息，且造成下游任务性能损害' if conclusion else '⚠️  证据尚不充分，见 Summary 细节'}")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    output = {
        "step1_probe_results": step1_results,
        "step2_forecast_results": {
            k: v for k, v in step2_results.items()
            if isinstance(v, dict)
        },
        "summary": {
            "avg_freq_R2_step1"      : avg_r2_step1,
            "freq_info_lost"         : bool(freq_info_lost),
            "forecast_gap_freq_jepa" : gap_vs_freq,
            "forecast_gap_jepa_time" : gap_vs_time,
            "research_question_valid": bool(conclusion),
        }
    }
    out_path = os.path.join(args.save_dir, "freq_probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存 → {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()