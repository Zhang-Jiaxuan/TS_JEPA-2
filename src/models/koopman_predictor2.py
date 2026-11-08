"""
Koopman Predictor — 替换原始Transformer Predictor
将JEPA的预测机制从"语义上下文补全"改为"动力学线性传播"

理论基础:
  - Ruiz-Morales et al. (AAAI 2026): JEPA隐式学习Koopman不变子空间
  - 本模块显式建模完整Koopman谱: 不变子空间(z_inv) + 动力学子空间(z_osc)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Cayley变换：生成稳定的Koopman矩阵
# ─────────────────────────────────────────────
def cayley_map(A):
    """
    将任意方阵映射为正交矩阵（谱范数≤1）
    A: (d, d) 反对称矩阵
    返回: (d, d) 正交矩阵
    """
    d = A.shape[-1]
    I = torch.eye(d, device=A.device, dtype=A.dtype)
    # 反对称化
    A_skew = (A - A.transpose(-1, -2)) / 2.0
    return torch.linalg.solve(I + A_skew, I - A_skew)


# ─────────────────────────────────────────────
# 2. PatchDecoupler: 将表示分解为不变+动力学子空间
# ─────────────────────────────────────────────
class PatchDecoupler(nn.Module):
    """
    将encoder输出的表示分解为两个正交子空间:
      z_inv: Koopman特征值=1的不变子空间 (体制信息)
      z_osc: Koopman特征值≠1的动力学子空间 (演化信息)
    """
    def __init__(self, embed_dim, inv_dim, osc_dim):
        super().__init__()
        assert inv_dim + osc_dim == embed_dim, \
            f"inv_dim({inv_dim}) + osc_dim({osc_dim}) must equal embed_dim({embed_dim})"
        
        self.inv_dim = inv_dim
        self.osc_dim = osc_dim
        
        # 投影到两个子空间
        self.proj_inv = nn.Sequential(
            nn.Linear(embed_dim, inv_dim),
            nn.LayerNorm(inv_dim)
        )
        self.proj_osc = nn.Sequential(
            nn.Linear(embed_dim, osc_dim),
            nn.LayerNorm(osc_dim)
        )
        # 重建投影（用于拼接回embed_dim）
        self.proj_back = nn.Linear(inv_dim + osc_dim, embed_dim)

    def forward(self, z):
        """
        z: (B, T, embed_dim)
        返回: z_inv (B, T, inv_dim), z_osc (B, T, osc_dim)
        """
        z_inv = self.proj_inv(z)
        z_osc = self.proj_osc(z)
        return z_inv, z_osc

    def merge(self, z_inv, z_osc):
        """将两个子空间拼接回embed_dim"""
        return self.proj_back(torch.cat([z_inv, z_osc], dim=-1))


# ─────────────────────────────────────────────
# 3. FiLM超网络: 根据全局语义生成K的调制参数
# ─────────────────────────────────────────────
class FiLMHyperNet(nn.Module):
    """
    根据全局语义z_global生成条件化Koopman矩阵的调制参数
    实现: K(z_g) = γ(z_g) ⊙ K_base + β(z_g)
    """
    def __init__(self, global_dim, osc_dim):
        super().__init__()
        self.osc_dim = osc_dim
        
        self.net = nn.Sequential(
            nn.Linear(global_dim, global_dim * 2),
            nn.GELU(),
            nn.Linear(global_dim * 2, osc_dim * osc_dim * 2)
        )
        # 初始化: γ≈1, β≈0，使初始K接近K_base
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_global):
        """
        z_global: (B, global_dim)
        返回: gamma (B, d, d), beta (B, d, d)
        """
        params = self.net(z_global)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        d = self.osc_dim
        # gamma初始化为1（恒等调制）
        gamma = gamma.view(-1, d, d) + torch.eye(d, device=z_global.device)
        beta  = beta.view(-1, d, d)
        return gamma, beta


# ─────────────────────────────────────────────
# 4. 条件化Koopman算子
# ─────────────────────────────────────────────
class ConditionedKoopmanOperator(nn.Module):
    """
    条件化Koopman算子: K(z_global) = γ ⊙ K_base + β
    K_base通过Cayley变换保证稳定性
    """
    def __init__(self, osc_dim):
        super().__init__()
        self.osc_dim = osc_dim
        # 基础Koopman矩阵的参数（反对称，用于Cayley变换）
        self.A = nn.Parameter(torch.randn(osc_dim, osc_dim) * 0.01)

    def get_K_base(self):
        return cayley_map(self.A.unsqueeze(0)).squeeze(0)  # (d, d)

    def forward(self, z_osc_t, gamma, beta, steps=1):
        """
        z_osc_t: (B, d_osc)  — 起始状态
        gamma:   (B, d, d)   — FiLM增益
        beta:    (B, d, d)   — FiLM偏置
        steps:   int         — 传播步数h
        返回: z_osc_{t+h} (B, d_osc)
        """
        K_base = self.get_K_base()  # (d, d)
        # 条件化: K_cond = γ * K_base + β
        K_cond = gamma * K_base.unsqueeze(0) + beta  # (B, d, d)
        
        # 多步传播: z_{t+h} = K^h · z_t
        z = z_osc_t  # (B, d)
        for _ in range(steps):
            # bmm: (B, 1, d) × (B, d, d)^T → (B, 1, d)
            z = torch.bmm(z.unsqueeze(1), K_cond.transpose(1, 2)).squeeze(1)
        return z


# ─────────────────────────────────────────────
# 5. 完整Koopman Predictor（替换原始Transformer Predictor）
# ─────────────────────────────────────────────
class KoopmanPredictor(nn.Module):
    """
    替换TS-JEPA中的Transformer Predictor。
    
    原始Predictor: 可见patch表示 → Transformer → 预测masked patch表示
    本模块:        可见patch表示 → Koopman传播 → 预测masked patch表示
    
    接口与原始Predictor保持兼容，便于直接替换。
    """
    def __init__(
        self,
        num_patches,
        encoder_embed_dim=128,
        inv_dim=96,
        osc_dim=32,
    ):
        super().__init__()
        assert inv_dim + osc_dim == encoder_embed_dim
        
        self.num_patches = num_patches
        self.encoder_embed_dim = encoder_embed_dim
        self.inv_dim = inv_dim
        self.osc_dim = osc_dim

        # 子空间分解器
        self.decoupler = PatchDecoupler(encoder_embed_dim, inv_dim, osc_dim)
        
        # 全局语义提取（用于FiLM条件化）
        self.global_pool = nn.Sequential(
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
            nn.GELU(),
        )
        
        # FiLM超网络
        self.film_net = FiLMHyperNet(encoder_embed_dim, osc_dim)
        
        # 条件化Koopman算子
        self.koopman = ConditionedKoopmanOperator(osc_dim)
        
        # 最终投影回encoder_embed_dim（与原始Predictor输出对齐）
        self.output_proj = nn.Linear(inv_dim + osc_dim, encoder_embed_dim)

    def forward(self, encoded_vals, mask=None, non_masks=None, masked_positions=None):
        """
        encoded_vals:     (B, num_visible, embed_dim)  — 可见patch的encoder输出
        mask:             (B, num_masked)               — masked patch的位置索引
        non_masks:        (B, num_visible)              — 可见patch的位置索引
        masked_positions: list of int                   — masked patch相对于最后可见patch的步数
        
        返回: z_pred (B, num_masked, embed_dim)
        """
        B, num_visible, D = encoded_vals.shape
        
        # 1. 全局语义：对可见patch做平均池化
        z_global = self.global_pool(encoded_vals.mean(dim=1))  # (B, D)
        
        # 2. 生成FiLM调制参数
        gamma, beta = self.film_net(z_global)  # (B, d_osc, d_osc)
        
        # 3. 对可见patch做子空间分解
        z_inv, z_osc = self.decoupler(encoded_vals)
        # z_inv: (B, num_visible, inv_dim)
        # z_osc: (B, num_visible, osc_dim)
        
        # 4. 以最后一个可见patch为起点，逐步传播
        z_inv_last = z_inv[:, -1, :]   # (B, inv_dim)
        z_osc_last = z_osc[:, -1, :]   # (B, osc_dim)
        
        # 5. 对每个masked位置预测
        num_masked = mask.shape[1] if mask is not None else self.num_patches - num_visible
        
        preds = []
        for h in range(1, num_masked + 1):
            # 不变子空间：恒等传播（near-identity，继承Ruiz-Morales理论）
            z_inv_pred = z_inv_last  # (B, inv_dim)
            
            # 动力学子空间：Koopman传播h步
            z_osc_pred = self.koopman(z_osc_last, gamma, beta, steps=h)  # (B, osc_dim)
            
            # 拼接并投影
            z_pred_h = torch.cat([z_inv_pred, z_osc_pred], dim=-1)  # (B, D)
            z_pred_h = self.output_proj(z_pred_h)  # (B, embed_dim)
            preds.append(z_pred_h)
        
        # (B, num_masked, embed_dim)
        return torch.stack(preds, dim=1)

    def get_decoupled(self, z):
        """
        暴露分解接口，供损失函数使用
        z: (B, T, embed_dim)
        返回: z_inv (B, T, inv_dim), z_osc (B, T, osc_dim)
        """
        return self.decoupler(z)
