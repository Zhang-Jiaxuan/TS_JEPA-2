"""
Koopman-JEPA 损失函数
修复版：新增 loss_variance_penalty，防止 z_osc 坍塌到零
"""
import torch
import torch.nn.functional as F


def loss_invariant(z_inv_pred, z_inv_target):
    """
    不变子空间损失：驱动z_inv学习Koopman特征值=1的本征函数
    """
    return F.mse_loss(z_inv_pred, z_inv_target)


def loss_dynamic(z_osc_pred, z_osc_target):
    """
    动力学子空间损失：驱动z_osc学习Koopman特征值≠1的本征函数
    """
    return F.mse_loss(z_osc_pred, z_osc_target)


def loss_orthogonal(z_inv, z_osc):
    """
    正交性约束：强制两个子空间正交
    z_inv: (B, T, inv_dim)
    z_osc: (B, T, osc_dim)
    """
    B, T, _ = z_inv.shape
    z_inv_flat = z_inv.reshape(B * T, -1)   # (B*T, inv_dim)
    z_osc_flat = z_osc.reshape(B * T, -1)   # (B*T, osc_dim)
    gram = torch.mm(z_inv_flat.T, z_osc_flat)
    return torch.norm(gram, p='fro') / (B * T)


def loss_variance_penalty(z_pred, min_var=0.05):
    """
    方差惩罚：强制预测输出保持足够的方差，防止坍塌到零向量。

    z_pred:  (B, T, D) 或 (B, D)
    min_var: 方差下界阈值

    当 Var(z_pred) < min_var 时产生惩罚，
    当 Var(z_pred) >= min_var 时损失为0（不惩罚正常情况）。
    """
    var = z_pred.var(dim=0).mean()
    return F.relu(min_var - var)


def koopman_jepa_loss(
    z_pred,           # (B, num_masked, embed_dim) — Koopman预测
    z_ema_masked,     # (B, num_masked, embed_dim) — EMA目标
    predictor,        # KoopmanPredictor实例
    lambda_dyn=1.0,
    mu_orth=0.01,
    nu_var=0.1,       # 新增：方差惩罚权重
):
    """
    完整Koopman-JEPA损失：

    L = L_jepa + λ·(L_inv + L_dyn) + μ·L_orth + ν·L_var

      L_jepa : 原始JEPA损失（embed_dim空间对齐）
      L_inv  : 不变子空间对齐
      L_dyn  : 动力学子空间对齐
      L_orth : 两子空间正交约束
      L_var  : 方差惩罚（防z_osc坍塌，新增）
    """
    # ── 原始JEPA损失 ──────────────────────────────────
    l_jepa = torch.mean(torch.abs(z_pred - z_ema_masked))

    # ── 子空间分解 ────────────────────────────────────
    z_inv_pred, z_osc_pred = predictor.get_decoupled(z_pred)
    z_inv_tgt,  z_osc_tgt  = predictor.get_decoupled(z_ema_masked)

    # ── 子空间对齐损失 ────────────────────────────────
    l_inv  = loss_invariant(z_inv_pred, z_inv_tgt)
    l_dyn  = loss_dynamic(z_osc_pred, z_osc_tgt)
    l_orth = loss_orthogonal(z_inv_pred, z_osc_pred)

    # ── 方差惩罚（修复核心）──────────────────────────
    # 同时惩罚 z_osc_pred 和整体 z_pred 的方差过低
    l_var = loss_variance_penalty(z_osc_pred, min_var=0.05) \
          + loss_variance_penalty(z_pred,     min_var=0.01)

    # ── 总损失 ────────────────────────────────────────
    total = (l_jepa
             + lambda_dyn * (l_inv + l_dyn)
             + mu_orth    * l_orth
             + nu_var     * l_var)

    return total, {
        "l_jepa": l_jepa.item(),
        "l_inv":  l_inv.item(),
        "l_dyn":  l_dyn.item(),
        "l_orth": l_orth.item(),
        "l_var":  l_var.item(),
    }