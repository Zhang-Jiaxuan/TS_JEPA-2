"""
============================================================
TS-JEPA 完整消融实验
============================================================
基准对齐（B_JEPA 应复现）：
  官方口径 MSE (last step) ≈ 0.003331
  正确均值 MSE (all steps) ≈ 0.016022

消融链：
  A_Scratch   — 随机初始化 encoder（下界）
  B_JEPA      — 官方预训练 checkpoint（基准）
  C_Direct    — B + 频率辅助损失（不解耦，对照）
  D_Decoupled — B + 频率解耦投影（Step3 核心）
  E_KoopProp  — D + ODO Koopman 传播器（Step4 新增）

评估双口径：
  官方口径：最后一步 MSE（对齐论文数字）
  正确均值：全部步 MSE 均值（消融结论用）
============================================================
"""

import os, sys, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fft import fft, fftfreq

TSJEPA_ROOT = "/data_p6/zjx/TS_JEPA"
sys.path.insert(0, TSJEPA_ROOT)

from src.models.encoder import Encoder
from src.models.decoder import LinearDecoder
from src.data_loaders.data_loader import get_evaluation_loaders

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 官方 mse / mae
# ============================================================

def mse(pred, true):
    return np.mean((pred - true) ** 2)

def mae(pred, true):
    return np.mean(np.abs(pred - true))


# ============================================================
# 配置
# ============================================================

context     = 32
num_patches = 10
num_epochs  = 500

lr         = 1e-05
batch_size = 32

# checkpoint: encoder_128_2_1
EMBED_DIM   = 128
NHEAD       = 2
NUM_LAYERS  = 1
KERNEL_SIZE = 3

# 频率解耦参数
SPECTRAL_DIM = 16   # z_osc 维度；z_inv = 128-16 = 112

# Koopman 参数
KOOP_RHO    = 0.99
KOOP_LAMBDA = 1e-3
KOOP_EPS    = 0.1

DATA_PATH = f"{TSJEPA_ROOT}/data/weather/weather.csv"
CKPT_PATH = (
    f"{TSJEPA_ROOT}/logs/output_model/weather/"
    "lr_1e-07_ema_momentum_0.998_mask_ratio_0.7_ratio_patches_10_"
    "encoder_128_2_1_predictor_128_2_1_epoch_5000.pt"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# ============================================================
# 数据加载（全局共享，只加载一次）
# ============================================================

print("Load data...")
loader = get_evaluation_loaders(
    DATA_PATH, batch_size,
    ratio_patches=num_patches, mask_ratio=0.9,
)
input_dim          = len(loader.dataset[0][0][0])
num_patches_actual = len(loader.dataset[0][0])
test_df            = loader.dataset.test_df
num_steps = (len(test_df[context * num_patches:])) // context
print(f"train={len(loader.dataset)} | test_df={len(test_df)} | steps={num_steps}")


# ============================================================
# 工具函数
# ============================================================

def make_encoder(random_init=False):
    """构造 Encoder，可选随机初始化或加载预训练权重"""
    enc = Encoder(
        num_patches=num_patches_actual, dim_in=input_dim,
        kernel_size=KERNEL_SIZE, embed_dim=EMBED_DIM,
        embed_bias=True, nhead=NHEAD,
        num_layers=NUM_LAYERS, jepa=True,
    )
    if not random_init:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        enc.load_state_dict(ckpt["encoder"])
        print(f"  预训练权重加载 | epoch={ckpt.get('epoch','?')}")
    else:
        print("  随机初始化 encoder")
    return enc.to(device)


def make_decoder():
    return LinearDecoder(emb_dim=EMBED_DIM, patch_size=32).to(device)


def train_head(enc_fn, decoder, tag, extra_loss_fn=None):
    """
    完全对齐官方训练循环：
      - 固定500epoch，无scheduler，无grad clip，无早停
      - lr=1e-05，AdamW，只训练 decoder（+ extra_loss_fn 中的参数）
      - total_loss += loss / batch_size（官方写法）
    extra_loss_fn: callable(context_patches) -> (额外loss, 额外参数列表)
    """
    print(f"\n  [训练] {tag}")
    trainable = list(decoder.parameters())
    if extra_loss_fn is not None and hasattr(extra_loss_fn, 'parameters'):
        trainable += list(extra_loss_fn.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=lr)

    for epoch in range(num_epochs):
        decoder.train()
        total_loss = 0

        for context_patches, target_patch in loader:
            optimizer.zero_grad()
            context_patches = context_patches.to(device)
            target_patch    = target_patch.to(device)

            z    = enc_fn(context_patches)
            pred = decoder(z)
            loss = F.mse_loss(pred, target_patch, reduction="mean")

            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(context_patches, z)

            loss.backward()
            optimizer.step()
            total_loss += loss / batch_size

        if epoch % 10 == 0:
            print("  Epoch: {} - Total loss: {}".format(epoch, total_loss))

    print(f"  ✅ {tag} 训练完成")
    return decoder


def rolling_test(enc_fn, decoder, tag):
    """
    完全对齐官方测试循环，输出双口径结果
    """
    print(f"\n  [测试] {tag}")
    l_val_mse, l_val_mae = [], []

    for step in range(num_steps):
        current_context = (
            test_df[context*step : context*num_patches + context*step]
            .reshape(num_patches, context)
            .unsqueeze(0)
            .to(device)
        )
        target_value = test_df[
            context*num_patches + step*context :
            context*num_patches + (step+1)*context
        ]

        encoded   = enc_fn(current_context)
        predicted = decoder(encoded)

        v_mse = mse(predicted.flatten().detach().cpu().numpy(), target_value.numpy())
        v_mae = mae(predicted.flatten().detach().cpu().numpy(), target_value.numpy())
        l_val_mse.append(v_mse)
        l_val_mae.append(v_mae)

    # 官方口径（论文数字）
    official_mse = float(np.mean(v_mse))        # 最后一步标量
    official_mae = float(np.mean(l_val_mae))     # 全部步均值
    # 正确均值
    correct_mse  = float(np.mean(l_val_mse))
    correct_mae  = float(np.mean(l_val_mae))

    print(f"  [官方口径] MSE(last)={official_mse:.6f} | MAE(mean)={official_mae:.6f}")
    print(f"  [正确均值] MSE(mean)={correct_mse:.6f}  | MAE(mean)={correct_mae:.6f}")

    return {
        "official_mse": official_mse,
        "official_mae": official_mae,
        "correct_mse":  correct_mse,
        "correct_mae":  correct_mae,
    }


# ============================================================
# 频率解耦模块
# ============================================================

class DecoupledProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_inv  = nn.Linear(EMBED_DIM, EMBED_DIM - SPECTRAL_DIM)
        self.proj_spec = nn.Linear(EMBED_DIM, SPECTRAL_DIM)
    def forward(self, z):
        return self.proj_inv(z), self.proj_spec(z)
    def fused(self, z):
        z_inv, z_osc = self.forward(z)
        return torch.cat([z_inv, z_osc], dim=-1)


def compute_dominant_freq(patch_np):
    spec  = np.abs(fft(patch_np))
    freqs = fftfreq(len(patch_np))
    pos   = freqs > 0
    return float(freqs[pos][np.argmax(spec[pos])])


def build_freq_labels(loader_):
    """预计算训练集频率标签的均值/标准差"""
    freqs = []
    for cp, _ in loader_:
        last = cp[:, -1, :].numpy()
        for i in range(len(last)):
            freqs.append(compute_dominant_freq(last[i]))
    arr = np.array(freqs)
    return float(arr.mean()), float(arr.std()) + 1e-8


# ============================================================
# ODO Koopman 传播器
# ============================================================

def cayley(A):
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    S = (A - A.T) / 2.0
    return torch.linalg.solve(I + S, I - S)


class ODOKoopmanPropagator(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_U       = nn.Parameter(torch.randn(SPECTRAL_DIM, SPECTRAL_DIM) * 0.01)
        self.A_V       = nn.Parameter(torch.randn(SPECTRAL_DIM, SPECTRAL_DIM) * 0.01)
        self.log_sigma = nn.Parameter(torch.zeros(SPECTRAL_DIM))

    def get_K(self):
        U     = cayley(self.A_U)
        V     = cayley(self.A_V)
        sigma = torch.clamp(F.softplus(self.log_sigma), max=KOOP_RHO)
        return U @ torch.diag(sigma) @ V.t()

    def forward(self, z_osc):
        return z_osc @ self.get_K().t()

    def lyapunov_loss(self):
        K = self.get_K()
        I = torch.eye(SPECTRAL_DIM, device=K.device, dtype=K.dtype)
        return KOOP_LAMBDA * F.relu(
            torch.norm(K.t() @ K - I, p='fro') - KOOP_EPS
        )

    @torch.no_grad()
    def spectral_radius(self):
        return torch.linalg.eigvals(self.get_K()).abs().max().item()


# ============================================================
# A_Scratch：随机初始化（下界）
# ============================================================

print("\n" + "="*60)
print("【A_Scratch】随机初始化 encoder")
enc_A = make_encoder(random_init=True)
for p in enc_A.parameters(): p.requires_grad_(False)
enc_A.eval()

def enc_fn_A(x):
    return torch.sum(enc_A(x), dim=1)

dec_A = make_decoder()
train_head(enc_fn_A, dec_A, "A_Scratch")
res_A = rolling_test(enc_fn_A, dec_A, "A_Scratch")


# ============================================================
# B_JEPA：官方预训练（基准）
# ============================================================

print("\n" + "="*60)
print("【B_JEPA】官方预训练 checkpoint（基准）")
enc_B = make_encoder(random_init=False)
for p in enc_B.parameters(): p.requires_grad_(False)
enc_B.eval()

def enc_fn_B(x):
    return torch.sum(enc_B(x), dim=1)

dec_B = make_decoder()
train_head(enc_fn_B, dec_B, "B_JEPA")
res_B = rolling_test(enc_fn_B, dec_B, "B_JEPA")


# ============================================================
# C_Direct：频率辅助损失，不解耦（对照）
# ============================================================

print("\n" + "="*60)
print("【C_Direct】频率辅助损失，不解耦（对照）")

freq_mean_C, freq_std_C = build_freq_labels(loader)

class FreqAuxLoss_C(nn.Module):
    """直接在全局表征上加频率预测辅助损失"""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, 32), nn.GELU(), nn.Linear(32, 1)
        )
    def forward(self, context_patches, z):
        last   = context_patches[:, -1, :].detach().cpu().numpy()
        labels = torch.tensor(
            [(compute_dominant_freq(last[i]) - freq_mean_C) / freq_std_C
             for i in range(len(last))],
            dtype=torch.float32, device=device
        )
        pred_freq = self.head(z.detach()).squeeze(-1)
        return 0.1 * F.mse_loss(pred_freq, labels)

freq_aux_C = FreqAuxLoss_C().to(device)
dec_C = make_decoder()
# C 的 enc_fn 与 B 相同，辅助损失不改变表征
train_head(enc_fn_B, dec_C, "C_Direct", extra_loss_fn=freq_aux_C)
res_C = rolling_test(enc_fn_B, dec_C, "C_Direct")


# ============================================================
# D_Decoupled：频率解耦（Step3 核心）
# ============================================================

print("\n" + "="*60)
print("【D_Decoupled】频率解耦（Step3 核心）")

freq_mean_D, freq_std_D = freq_mean_C, freq_std_C  # 复用

decoupler = DecoupledProjector().to(device)

class FreqAuxLoss_D(nn.Module):
    """在 z_osc 子空间上加频率预测辅助损失"""
    def __init__(self, dec_proj):
        super().__init__()
        self.dec_proj = dec_proj
        self.head = nn.Sequential(
            nn.Linear(SPECTRAL_DIM, 32), nn.GELU(), nn.Linear(32, 1)
        )
    def forward(self, context_patches, z):
        _, z_osc = self.dec_proj(z)
        last   = context_patches[:, -1, :].detach().cpu().numpy()
        labels = torch.tensor(
            [(compute_dominant_freq(last[i]) - freq_mean_D) / freq_std_D
             for i in range(len(last))],
            dtype=torch.float32, device=device
        )
        pred_freq = self.head(z_osc).squeeze(-1)
        return 0.1 * F.mse_loss(pred_freq, labels)

freq_aux_D = FreqAuxLoss_D(decoupler).to(device)

# D 的 enc_fn：base_enc → decoupler → cat(z_inv, z_osc)
def enc_fn_D(x):
    z = torch.sum(enc_B(x), dim=1)
    return decoupler.fused(z)

dec_D = make_decoder()
# 训练：decoder + decoupler + freq_head 同时更新
param_groups_D = (
    list(dec_D.parameters()) +
    list(decoupler.parameters()) +
    list(freq_aux_D.head.parameters())
)
opt_D = torch.optim.AdamW(param_groups_D, lr=lr)

print(f"  [训练] D_Decoupled")
for epoch in range(num_epochs):
    dec_D.train(); decoupler.train(); freq_aux_D.train()
    total_loss = 0

    for context_patches, target_patch in loader:
        opt_D.zero_grad()
        context_patches = context_patches.to(device)
        target_patch    = target_patch.to(device)

        z    = enc_fn_D(context_patches)
        pred = dec_D(z)
        loss = F.mse_loss(pred, target_patch, reduction="mean")
        loss = loss + freq_aux_D(context_patches,
                                 torch.sum(enc_B(context_patches), dim=1))
        loss.backward()
        opt_D.step()
        total_loss += loss / batch_size

    if epoch % 10 == 0:
        print("  Epoch: {} - Total loss: {}".format(epoch, total_loss))

print("  ✅ D_Decoupled 训练完成")
decoupler.eval()
for p in decoupler.parameters(): p.requires_grad_(False)
res_D = rolling_test(enc_fn_D, dec_D, "D_Decoupled")


# ============================================================
# E_KoopProp：解耦 + Koopman 传播（Step4）
# ============================================================

print("\n" + "="*60)
print("【E_KoopProp】解耦 + ODO Koopman 传播（Step4）★")

propagator = ODOKoopmanPropagator().to(device)

def enc_fn_E(x):
    """base_enc → decoupler(冻结) → propagate z_osc → cat"""
    z = torch.sum(enc_B(x), dim=1)
    with torch.no_grad():
        z_inv, z_osc = decoupler(z)
    z_osc_prop = propagator(z_osc)
    return torch.cat([z_inv, z_osc_prop], dim=-1)

dec_E = make_decoder()
param_groups_E = list(propagator.parameters()) + list(dec_E.parameters())
opt_E = torch.optim.AdamW(param_groups_E, lr=lr)

print(f"  [训练] E_KoopProp")
for epoch in range(num_epochs):
    dec_E.train(); propagator.train()
    total_loss = 0

    for context_patches, target_patch in loader:
        opt_E.zero_grad()
        context_patches = context_patches.to(device)
        target_patch    = target_patch.to(device)

        z    = enc_fn_E(context_patches)
        pred = dec_E(z)
        loss = (F.mse_loss(pred, target_patch, reduction="mean")
                + propagator.lyapunov_loss())
        loss.backward()
        opt_E.step()
        total_loss += loss / batch_size

    if epoch % 10 == 0:
        rho = propagator.spectral_radius()
        print("  Epoch: {} - Total loss: {} | ρ(K)={:.4f}".format(
            epoch, total_loss, rho))

print(f"  ✅ E_KoopProp 训练完成 | ρ(K)={propagator.spectral_radius():.6f}")
res_E = rolling_test(enc_fn_E, dec_E, "E_KoopProp")


# ============================================================
# 消融对比表
# ============================================================

results = {
    "A_Scratch"  : (res_A, "无预训练（下界）"),
    "B_JEPA"     : (res_B, "JEPA预训练（基准）"),
    "C_Direct"   : (res_C, "+频率辅助损失（不解耦）"),
    "D_Decoupled": (res_D, "+频率解耦投影"),
    "E_KoopProp" : (res_E, "+Koopman传播 ★"),
}

bm_off = res_B["official_mse"]
bm_cor = res_B["correct_mse"]

print("\n" + "="*80)
print("📊  消融对比表 — Weather OT（归一化空间）")
print(f"    lr={lr} | epochs={num_epochs} | batch={batch_size} | embed={EMBED_DIM}")
print("="*80)
print(f"{'组别':<16}│{'官方MSE(last)':>14}│{'均值MSE':>10}│"
      f"{'ΔMSE(官方)':>12}│{'ΔMSE(均值)':>12}│{'说明'}")
print("─"*80)
for name, (res, desc) in results.items():
    d_off = "" if name=="B_JEPA" else f"{(res['official_mse']-bm_off)/bm_off*100:+.1f}%"
    d_cor = "" if name=="B_JEPA" else f"{(res['correct_mse']-bm_cor)/bm_cor*100:+.1f}%"
    mark  = " ←基准" if name=="B_JEPA" else ""
    print(f"{name:<16}│{res['official_mse']:>14.6f}│{res['correct_mse']:>10.6f}│"
          f"{d_off:>12}│{d_cor:>12}│{desc}{mark}")
print("="*80)

print("\n📌 注：")
print("  官方MSE(last) = 测试集最后一步MSE，即论文报告数字（官方代码bug）")
print("  均值MSE       = 全部484步MSE均值，统计上更可靠，用于消融结论")
print(f"\n  B_JEPA 官方口径应≈0.003331，当前={res_B['official_mse']:.6f}")
if abs(res_B['official_mse'] - 0.003331) < 0.001:
    print("  ✅ 与论文对齐成功！")
else:
    print("  ⚠️  与预期有偏差，检查 checkpoint 路径或 embed_dim 设置")