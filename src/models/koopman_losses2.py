"""
Koopman-JEPA的三项损失函数
"""
import torch
import torch.nn.functional as F


def loss_invariant(z_inv_pred, z_inv_target):
    """
    不变子空间损失: 驱动z_inv学习Koopman特征值=1的本征函数
    对应Ruiz-Morales的near-identity约束
    """
    return F.mse_loss(z_inv_pred, z_inv_target)


def loss_dynamic(z_osc_pred, z_osc_target):
    """
    动力学子空间损失: 驱动z_osc学习Koopman特征值≠1的本征函数
    """
    return F.mse_loss(z_osc_pred, z_osc_target)


def loss_orthogonal(z_inv, z_osc):
    """
    正交性约束: 强制两个子空间正交
    z_inv: (B, T, inv_dim)
    z_osc: (B, T, osc_dim)
    """
    B, T, _ = z_inv.shape
    # 展平时序维度
    z_inv_flat = z_inv.reshape(B * T, -1)  # (B*T, inv_dim)
    z_osc_flat = z_osc.reshape(B * T, -1)  # (B*T, osc_dim)
    
    # 计算跨子空间的Gram矩阵
    gram = torch.mm(z_inv_flat.T, z_osc_flat)  # (inv_dim, osc_dim)
    return torch.norm(gram, p='fro') / (B * T)


def koopman_jepa_loss(
    z_pred,           # (B, num_masked, embed_dim) — Koopman预测
    z_ema_masked,     # (B, num_masked, embed_dim) — EMA目标
    predictor,        # KoopmanPredictor实例（用于获取分解）
    lambda_dyn=1.0,
    mu_orth=0.01,
):
    """
    完整的Koopman-JEPA损失
    
    L = L_jepa + λ·L_dyn_decomp + μ·L_orth
    
    其中:
      L_jepa      = ||z_pred - z_ema_masked||   (原始JEPA损失，在embed_dim空间)
      L_dyn_decomp = 分解后子空间的对齐损失
      L_orth       = 正交性约束
    """
    # 原始JEPA损失（保持与原始代码兼容）
    l_jepa = torch.mean(torch.abs(z_pred - z_ema_masked))
    
    # 对预测和目标分别做子空间分解
    z_inv_pred, z_osc_pred = predictor.get_decoupled(z_pred)
    z_inv_tgt,  z_osc_tgt  = predictor.get_decoupled(z_ema_masked)
    
    # 子空间对齐损失
    l_inv  = loss_invariant(z_inv_pred, z_inv_tgt)
    l_dyn  = loss_dynamic(z_osc_pred, z_osc_tgt)
    l_orth = loss_orthogonal(z_inv_pred, z_osc_pred)
    
    total = l_jepa + lambda_dyn * (l_inv + l_dyn) + mu_orth * l_orth
    
    return total, {
        "l_jepa": l_jepa.item(),
        "l_inv":  l_inv.item(),
        "l_dyn":  l_dyn.item(),
        "l_orth": l_orth.item(),
    }
