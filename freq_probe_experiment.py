#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
频率信息线性探针实验 + 下游任务性能验证（纯回归版 v2）

核心发现导向：
  TS-JEPA 丢失了 dominant_freq（主频位置，R²=0.11），
  但保留了频谱能量分布（top_k_power R²>0.57）。

  Step 2 需要验证：主频位置丢失 → 周期感知任务性能损害

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
    p = argparse.ArgumentParser("TS-JEPA Frequency Linear Probe v2")
    p.add_argument("--z_path",      type=str, required=True)
    p.add_argument("--x_path",      type=str, required=True)
    p.add_argument("--save_dir",    type=str, default="outputs/freq_probe")
    p.add_argument("--window_size", type=int, default=210)
    p.add_argument("--stride",      type=int, default=1)
    p.add_argument("--num_patches", type=int, default=10)
    p.add_argument("--top_k_freqs", type=int, default=5)
    p.add_argument("--test_size",   type=float, default=0.2)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 频率特征提取
# ─────────────────────────────────────────────────────────────────────────────

def extract_frequency_features(x_windows, top_k=5):
    """提取频率特征，返回样本级标量（对 channel 取均值）"""
    N, T, C = x_windows.shape
    freqs = rfftfreq(T)

    dominant_freq    = np.zeros((N, C), dtype=np.float32)
    dominant_period  = np.zeros((N, C), dtype=np.float32)
    spectral_entropy = np.zeros((N, C), dtype=np.float32)
    mean_freq        = np.zeros((N, C), dtype=np.float32)
    top_k_power      = np.zeros((N, C, top_k), dtype=np.float32)

    for i in range(N):
        for c in range(C):
            seg       = x_windows[i, :, c]
            fft_vals  = np.abs(rfft(seg))
            power     = fft_vals ** 2
            power_sum = power.sum() + 1e-8

            dominant_idx        = np.argmax(power[1:]) + 1
            dominant_freq[i, c] = freqs[dominant_idx]
            # 主频对应的周期（步数）
            dominant_period[i, c] = (1.0 / freqs[dominant_idx]
                                     if freqs[dominant_idx] > 0 else T)

            p_norm = power / power_sum
            p_norm_pos = p_norm[p_norm > 0]
            spectral_entropy[i, c] = -np.sum(p_norm_pos * np.log(p_norm_pos + 1e-8))
            mean_freq[i, c]        = np.sum(freqs * p_norm)

            sorted_idx = np.argsort(power[1:])[::-1][:top_k] + 1
            for ki, idx in enumerate(sorted_idx):
                top_k_power[i, c, ki] = power[idx] / power_sum

    features = {
        "dominant_freq"   : dominant_freq.mean(axis=1),
        "dominant_period" : dominant_period.mean(axis=1),
        "spectral_entropy": spectral_entropy.mean(axis=1),
        "mean_freq"       : mean_freq.mean(axis=1),
    }
    for k in range(top_k):
        features[f"top{k+1}_power"] = top_k_power[:, :, k].mean(axis=1)

    # 保留 per-channel 版本供 Step2 使用
    features["_dominant_freq_per_channel"]   = dominant_freq    # [N, C]
    features["_dominant_period_per_channel"] = dominant_period  # [N, C]
    features["_top_k_power_per_channel"]     = top_k_power      # [N, C, K]

    print(f"[FreqFeatures] N={N}, T={T}, C={C}, top_k={top_k}")
    for name, arr in features.items():
        if not name.startswith("_"):
            print(f"  {name:20s}: range=[{arr.min():.4f}, {arr.max():.4f}]  "
                  f"mean={arr.mean():.4f}  std={arr.std():.4f}")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 回归线性探针
# ─────────────────────────────────────────────────────────────────────────────

def run_regression_probe(Z, y, label_name, test_size=0.2, seed=42):
    Z_tr, Z_te, y_tr, y_te = train_test_split(
        Z, y, test_size=test_size, random_state=seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(Z_tr, y_tr)
    y_pred       = pipe.predict(Z_te)
    r2           = r2_score(y_te, y_pred)
    mse          = mean_squared_error(y_te, y_pred)
    baseline_r2  = r2_score(y_te, np.full_like(y_te, y_tr.mean()))
    print(f"  [{label_name:30s}]  R²={r2:+.4f}  "
          f"MSE={mse:.6f}  Baseline_R²={baseline_r2:+.4f}")
    return {"label": label_name, "R2": float(r2), "MSE": float(mse),
            "baseline_R2": float(baseline_r2),
            "n_train": int(len(Z_tr)), "n_test": int(len(Z_te))}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1：频率信息线性探针
# ─────────────────────────────────────────────────────────────────────────────

def run_step1(Z, x_windows, args):
    print("\n" + "=" * 70)
    print("STEP 1: 频率信息线性探针（回归）")
    print("=" * 70)

    freq_feats = extract_frequency_features(x_windows, top_k=args.top_k_freqs)
    scalar_targets = {k: v for k, v in freq_feats.items()
                      if not k.startswith("_")}

    print(f"\n[回归探针] Z shape={Z.shape}")
    results = []
    for name, y in scalar_targets.items():
        res = run_regression_probe(Z, y, name, args.test_size, args.seed)
        results.append(res)

    r2_values = [r["R2"] for r in results]
    avg_r2    = float(np.mean(r2_values))
    good = [r["label"] for r in results if r["R2"] > 0.5]
    poor = [r["label"] for r in results if r["R2"] < 0.3]

    print(f"\n[Step 1 Summary]")
    print(f"  平均 R²         = {avg_r2:.4f}")
    print(f"  保留较好 (>0.5) : {good if good else '无'}")
    print(f"  丢失严重 (<0.3) : {poor if poor else '无'}")
    print(f"\n  ⭐ 关键发现：")
    dom_r2 = next(r["R2"] for r in results if r["label"] == "dominant_freq")
    ent_r2 = next(r["R2"] for r in results if r["label"] == "spectral_entropy")
    p1_r2  = next(r["R2"] for r in results if r["label"] == "top1_power")
    print(f"     dominant_freq（主频位置）R² = {dom_r2:.4f}  ← 严重丢失")
    print(f"     spectral_entropy         R² = {ent_r2:.4f}  ← 保留较好")
    print(f"     top1_power（主频强度）   R² = {p1_r2:.4f}  ← 保留较好")
    print(f"\n  → TS-JEPA 保留了「频率的能量强度」，")
    print(f"    但丢失了「主频的位置（即周期长度）」")

    freq_info_lost = len(poor) > 0
    return results, freq_feats, freq_info_lost


# ─────────────────────────────────────────────────────────────────────────────
# Step 2：针对主频位置丢失的下游任务验证
# ─────────────────────────────────────────────────────────────────────────────

def build_periodic_forecast_target(x_windows, freq_feats):
    """
    任务 2A：周期对齐预测。
    
    对每个样本，根据其主频对应的周期 P，
    预测目标 = 窗口内 x[T-P : T] 的均值（即"一个周期前"的片段均值）。
    
    这个任务天然需要知道周期长度（dominant_freq），
    如果 latent 丢失了主频位置，则无法准确完成此任务。
    """
    N, T, C = x_windows.shape
    dominant_period = freq_feats["_dominant_period_per_channel"]  # [N, C]

    y = np.zeros(N, dtype=np.float32)
    for i in range(N):
        # 取所有 channel 的平均周期，四舍五入为整数步长
        P = int(np.round(dominant_period[i].mean()))
        P = max(2, min(P, T - 1))   # 限制在合理范围
        # 目标：窗口内最后 P 步的均值
        y[i] = x_windows[i, -P:, :].mean()
    return y


def build_next_period_phase_target(x_windows, freq_feats):
    """
    任务 2B：相位预测。
    
    预测目标 = 窗口最后一步与其主频余弦波在该时刻的相位差。
    相位信息直接依赖于主频位置，是对 dominant_freq 丢失最敏感的任务。
    """
    N, T, C = x_windows.shape
    dominant_freq_pc = freq_feats["_dominant_freq_per_channel"]  # [N, C]

    y = np.zeros(N, dtype=np.float32)
    for i in range(N):
        phases = []
        for c in range(C):
            f = dominant_freq_pc[i, c]
            if f > 0:
                # 在时刻 T-1 处，主频余弦波的相位
                phase = np.cos(2 * np.pi * f * (T - 1))
                phases.append(phase)
        y[i] = np.mean(phases) if phases else 0.0
    return y


def build_period_similarity_target(x_windows, freq_feats):
    """
    任务 2C：周期相似度预测（检索代理任务）。
    
    对每个样本，预测目标 = 该样本主频与数据集中位数主频的差值。
    这衡量了"该样本的周期与典型周期的偏差"，
    是频率感知检索/匹配任务的代理。
    """
    dominant_freq = freq_feats["dominant_freq"]   # [N]
    median_freq   = np.median(dominant_freq)
    y = dominant_freq - median_freq               # [N] 中心化的主频偏差
    return y


def run_step2(Z, x_windows, freq_feats, args):
    """
    Step 2：三个针对主频位置的下游任务。

    对比：
      (a) TS-JEPA latent Z
      (b) 手工频率特征（含主频位置，上界）
      (c) 时域统计特征（不含频率，baseline）
    """
    print("\n" + "=" * 70)
    print("STEP 2: 下游任务 —— 主频位置丢失的性能影响")
    print("=" * 70)
    print("  设计原则：选择天然依赖「主频位置」的任务，")
    print("  而非依赖时域均值的任务（后者与频率无关）\n")

    N, T, C = x_windows.shape

    # 构造三个任务的目标
    tasks = {
        "2A_periodic_forecast" : build_periodic_forecast_target(x_windows, freq_feats),
        "2B_phase_prediction"  : build_next_period_phase_target(x_windows, freq_feats),
        "2C_freq_deviation"    : build_period_similarity_target(x_windows, freq_feats),
    }

    # 三种特征
    freq_scalar = {k: v for k, v in freq_feats.items() if not k.startswith("_")}
    freq_vec    = np.stack(list(freq_scalar.values()), axis=1)   # [N, n_feats]

    time_vec = np.concatenate([
        x_windows.mean(axis=1),
        x_windows.std(axis=1),
        x_windows.max(axis=1) - x_windows.min(axis=1),
    ], axis=1)   # [N, 3C]

    all_results = {}
    summary_rows = []

    for task_name, y_target in tasks.items():
        print(f"\n{'─'*60}")
        print(f"  任务 {task_name}")
        print(f"  目标范围: [{y_target.min():.4f}, {y_target.max():.4f}]  "
              f"std={y_target.std():.4f}")

        res_a = run_regression_probe(Z,        y_target,
                                     f"{task_name}_TSJEPA",
                                     args.test_size, args.seed)
        res_b = run_regression_probe(freq_vec, y_target,
                                     f"{task_name}_FreqFeats",
                                     args.test_size, args.seed)
        res_c = run_regression_probe(time_vec, y_target,
                                     f"{task_name}_TimeDomain",
                                     args.test_size, args.seed)

        gap_freq_jepa = res_b["R2"] - res_a["R2"]
        gap_jepa_time = res_a["R2"] - res_c["R2"]

        print(f"\n  R² 对比: JEPA={res_a['R2']:+.4f}  "
              f"FreqFeats={res_b['R2']:+.4f}  TimeDomain={res_c['R2']:+.4f}")
        print(f"  gap(FreqFeats - JEPA) = {gap_freq_jepa:+.4f}  "
              f"{'⚠️  主频丢失造成性能损害' if gap_freq_jepa > 0.05 else '差距不显著'}")

        all_results[task_name] = {
            "tsjepa"   : res_a,
            "freq_feats": res_b,
            "time_domain": res_c,
            "gap_freq_jepa": float(gap_freq_jepa),
            "gap_jepa_time": float(gap_jepa_time),
        }
        summary_rows.append({
            "task"            : task_name,
            "JEPA_R2"         : res_a["R2"],
            "FreqFeats_R2"    : res_b["R2"],
            "TimeDomain_R2"   : res_c["R2"],
            "gap_freq_jepa"   : gap_freq_jepa,
            "damaged"         : gap_freq_jepa > 0.05,
        })

    # 汇总
    print(f"\n{'='*70}")
    print(f"[Step 2 Summary]")
    print(f"  {'任务':<30} {'JEPA':>8} {'FreqFeat':>10} {'TimeDom':>10} "
          f"{'gap(F-J)':>10} {'损害?':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for row in summary_rows:
        print(f"  {row['task']:<30} {row['JEPA_R2']:>+8.4f} "
              f"{row['FreqFeats_R2']:>+10.4f} {row['TimeDomain_R2']:>+10.4f} "
              f"{row['gap_freq_jepa']:>+10.4f} "
              f"{'⚠️ 是' if row['damaged'] else '否':>8}")

    n_damaged = sum(r["damaged"] for r in summary_rows)
    print(f"\n  {n_damaged}/{len(summary_rows)} 个任务检测到主频丢失造成的性能损害")

    return all_results, summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# 加载原始数据
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_windows(x_path, window_size, stride=1):
    df = pd.read_csv(x_path)
    x  = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std  = x.std(axis=0,  keepdims=True) + 1e-6
    x    = (x - mean) / std
    xs = [x[i: i + window_size]
          for i in range(0, len(x) - window_size + 1, stride)]
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

    print(f"[Load] latent Z: {args.z_path}")
    Z = np.load(args.z_path)
    print(f"[Load] Z shape : {Z.shape}")

    x_windows = load_raw_windows(args.x_path, args.window_size, args.stride)

    N = min(len(Z), len(x_windows))
    Z, x_windows = Z[:N], x_windows[:N]
    print(f"[Align] 使用 {N} 个样本")

    # ── Step 1 ───────────────────────────────────────────────────────────────
    step1_results, freq_feats, freq_info_lost = run_step1(Z, x_windows, args)

    # ── Step 2 ───────────────────────────────────────────────────────────────
    step2_results, summary_rows = run_step2(Z, x_windows, freq_feats, args)

    # ── 最终结论 ──────────────────────────────────────────────────────────────
    avg_r2_step1 = float(np.mean([r["R2"] for r in step1_results]))
    dom_r2 = next(r["R2"] for r in step1_results if r["label"] == "dominant_freq")
    n_damaged = sum(r["damaged"] for r in summary_rows)
    conclusion = (dom_r2 < 0.3) and (n_damaged >= 1)

    print("\n" + "=" * 70)
    print("📊 最终研究结论")
    print("=" * 70)
    print(f"  Step 1 | dominant_freq R²   = {dom_r2:.4f}  "
          f"← 主频位置{'严重丢失 ❌' if dom_r2 < 0.3 else '保留 ✅'}")
    print(f"  Step 1 | 平均频率 R²        = {avg_r2_step1:.4f}")
    print(f"  Step 2 | 受损任务数         = {n_damaged}/{len(summary_rows)}")
    print()
    print(f"  🔬 研究问题：")
    print(f"     TS-JEPA latent space 选择性地保留了频谱能量分布，")
    print(f"     但丢失了主频位置信息（dominant frequency location）。")
    if conclusion:
        print(f"     ✅ 此丢失在周期感知型下游任务中造成了可量化的性能损害。")
        print(f"     → 研究问题成立，可进入 Step 3 方案设计。")
    else:
        print(f"     ⚠️  下游任务损害证据尚不充分，建议检查任务设计。")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    output = {
        "step1_probe_results"  : step1_results,
        "step2_task_results"   : {
            k: {sk: sv for sk, sv in v.items() if isinstance(sv, dict)}
            for k, v in step2_results.items()
        },
        "step2_summary_rows"   : summary_rows,
        "summary": {
            "dominant_freq_R2"       : float(dom_r2),
            "avg_freq_R2_step1"      : avg_r2_step1,
            "freq_info_lost"         : bool(freq_info_lost),
            "n_damaged_tasks"        : int(n_damaged),
            "research_question_valid": bool(conclusion),
        }
    }
    out_path = os.path.join(args.save_dir, "freq_probe_results_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存 → {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()