
"""
诊断脚本：对比原始JEPA和Koopman-JEPA的表示质量
运行时间：约30秒
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys

sys.path.insert(0, "/data_p6/zjx/TS_JEPA")
from src.models.encoder import Encoder

device = "cuda:7" if torch.cuda.is_available() else "cpu"

CKPT_ORIG = (
    "/data_p6/zjx/TS_JEPA/logs/output_model/weather/"
    "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_ratio_patches_10_"
    "encoder_128_2_1_predictor_128_2_1_epoch_5000.pt"
)
CKPT_KOOP = (
    "/data_p6/zjx/TS_JEPA/logs/output_model/weather/"
    "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_ratio_patches_10_"
    "encoder_128_2_1_predictor_128_2_1_epoch_5000try2.pt"
)

# ── 加载encoder ──────────────────────────────────────────
def make_encoder(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    enc = Encoder(
        num_patches=10, dim_in=32, kernel_size=3,
        embed_dim=128, embed_bias=True,
        nhead=2, num_layers=1, jepa=True,
    ).to(device)
    enc.load_state_dict(ckpt["encoder"], strict=True)
    enc.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  ✅ 加载成功: {ckpt_path.split('/')[-1]} | epoch={epoch}")
    return enc

print("\n" + "="*60)
print("加载 Checkpoint")
print("="*60)
enc_orig = make_encoder(CKPT_ORIG)
enc_koop = make_encoder(CKPT_KOOP)

# ── 检查encoder权重是否真的不同 ──────────────────────────
print("\n" + "="*60)
print("权重差异检查")
print("="*60)
total_diff = 0.0
total_norm = 0.0
for (n1, p1), (n2, p2) in zip(
    enc_orig.named_parameters(), enc_koop.named_parameters()
):
    diff = (p1 - p2).norm().item()
    norm = p1.norm().item()
    total_diff += diff
    total_norm += norm
rel_diff = total_diff / (total_norm + 1e-8)
print(f"  参数总差异(L2):  {total_diff:.6f}")
print(f"  相对差异:        {rel_diff:.6f}")
if rel_diff < 1e-4:
    print("  ⚠️  WARNING: 两个checkpoint几乎相同！预训练可能没有生效。")
elif rel_diff < 0.01:
    print("  ⚠️  差异较小，Koopman-JEPA对encoder的改变有限。")
else:
    print("  ✅ 两个checkpoint有显著差异，预训练生效。")

# ── 用同一批输入比较表示质量 ──────────────────────────────
print("\n" + "="*60)
print("表示质量诊断")
print("="*60)
torch.manual_seed(42)
x = torch.randn(64, 10, 32).to(device)

with torch.no_grad():
    z_orig = enc_orig(x)   # (64, 10, 128)
    z_koop = enc_koop(x)   # (64, 10, 128)

def report(name, z):
    z_flat = z.reshape(-1, 128)                          # (640, 128)
    var    = z.var(dim=0).mean().item()
    mean   = z.mean().item()

    # 各向同性：特征维度方差的min/max比（越接近1越均匀）
    feat_var  = z_flat.var(dim=0)
    isotropy  = (feat_var.min() / feat_var.max().clamp(min=1e-8)).item()

    # 相邻patch的表示距离（时序结构丰富度）
    temp_dist = (z[:, 1:] - z[:, :-1]).norm(dim=-1).mean().item()

    # 表示坍塌检测：rank of representation matrix
    _, s, _ = torch.svd(z_flat - z_flat.mean(dim=0))
    effective_rank = (s / s.sum()).entropy().exp().item()

    print(f"\n  [{name}]")
    print(f"    Var(z):          {var:.5f}   {'✅' if var > 0.05 else '❌ 偏低'}")
    print(f"    Mean(z):         {mean:.5f}  {'✅' if abs(mean) < 0.1 else '⚠️ 偏移'}")
    print(f"    各向同性:         {isotropy:.4f}  {'✅' if isotropy > 0.1 else '❌ 各向异性严重'}")
    print(f"    相邻patch距离:    {temp_dist:.5f}")
    print(f"    有效秩(满=128):   {effective_rank:.1f}  {'✅' if effective_rank > 20 else '❌ 表示坍塌'}")
    return {
        "var": var, "mean": mean, "isotropy": isotropy,
        "temp_dist": temp_dist, "eff_rank": effective_rank
    }

r_orig = report("原始 JEPA", z_orig)
r_koop = report("Koopman-JEPA", z_koop)

# ── 时序一致性对比 ────────────────────────────────────────
print("\n" + "="*60)
print("时序结构对比")
print("="*60)
with torch.no_grad():
    for i, j in [(0, 1), (0, 4), (0, 9)]:
        s_orig = F.cosine_similarity(z_orig[:, i], z_orig[:, j], dim=-1).mean().item()
        s_koop = F.cosine_similarity(z_koop[:, i], z_koop[:, j], dim=-1).mean().item()
        print(f"  patch[{i}] vs patch[{j}] 余弦相似度: "
              f"原始={s_orig:.4f} | Koopman={s_koop:.4f} "
              f"{'↑ Koopman更一致' if s_koop > s_orig else '↓ Koopman更分散'}")

# ── 总结 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("总结")
print("="*60)
print(f"  有效秩:  原始={r_orig['eff_rank']:.1f} | Koopman={r_koop['eff_rank']:.1f}")
print(f"  各向同性: 原始={r_orig['isotropy']:.4f} | Koopman={r_koop['isotropy']:.4f}")
print(f"  时序距离: 原始={r_orig['temp_dist']:.5f} | Koopman={r_koop['temp_dist']:.5f}")
print()
if r_koop['eff_rank'] > r_orig['eff_rank']:
    print("  ✅ Koopman-JEPA表示更丰富（有效秩更高）")
else:
    print("  ⚠️  Koopman-JEPA表示有效秩未提升，z_osc坍塌影响了表示质量")