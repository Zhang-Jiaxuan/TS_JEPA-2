"""
Koopman Predictor — 替换原始Transformer Predictor
修复版：
  1. 维度与config对齐（embed_dim=384, osc_dim=64）
  2. FiLMHyperNet初始化修复（gamma正确初始化为I）
  3. ConditionedKoopmanOperator加入能量归一化（防坍塌）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Cayley变换：生成稳定的Koopman矩阵
# ─────────────────────────────────────────────
def cayley_map(A):
    """
    将任意方阵映射为正交矩阵（谱范数=1）
    A: (..., d, d)
    返回: (..., d, d) 正交矩阵
    """
    d = A.shape[-1]
    I = torch.eye(d, device=A.device, dtype=A.dtype)
    A_skew = (A - A.transpose(-1, -2)) / 2.0
    return torch.linalg.solve(I + A_skew, I - A_skew)


# ─────────────────────────────────────────────
# 2. PatchDecoupler
# ─────────────────────────────────────────────
class PatchDecoupler(nn.Module):
    """
    将encoder输出分解为两个子空间:
      z_inv: 不变子空间（Koopman特征值=1）
      z_osc: 动力学子空间（Koopman特征值≠1）
    """
    def __init__(self, embed_dim, inv_dim, osc_dim):
        super().__init__()
        assert inv_dim + osc_dim == embed_dim, \
            f"inv_dim({inv_dim}) + osc_dim({osc_dim}) must equal embed_dim({embed_dim})"
        self.inv_dim = inv_dim
        self.osc_dim = osc_dim

        self.proj_inv = nn.Sequential(
            nn.Linear(embed_dim, inv_dim),
            nn.LayerNorm(inv_dim),
        )
        self.proj_osc = nn.Sequential(
            nn.Linear(embed_dim, osc_dim),
            nn.LayerNorm(osc_dim),
        )
        self.proj_back = nn.Linear(inv_dim + osc_dim, embed_dim)

    def forward(self, z):
        """z: (B, T, embed_dim) → z_inv, z_osc"""
        return self.proj_inv(z), self.proj_osc(z)

    def merge(self, z_inv, z_osc):
        return self.proj_back(torch.cat([z_inv, z_osc], dim=-1))


# ─────────────────────────────────────────────
# 3. FiLM超网络（修复初始化）
# ─────────────────────────────────────────────
class FiLMHyperNet(nn.Module):
    """
    z_global(B, D) → gamma(B, d, d), beta(B, d, d)
    K(z_g) = gamma ⊙ K_base + beta

    修复：使用独立的gamma_head和beta_head
         gamma_head.bias 初始化为 flatten(I)，保证初期 gamma≈I
         beta_head 全零初始化，保证初期 beta≈0
    """
    def __init__(self, global_dim, osc_dim, hidden_dim=128):
        super().__init__()
        self.osc_dim = osc_dim
        out_dim = osc_dim * osc_dim

        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden_dim, out_dim)
        self.beta_head  = nn.Linear(hidden_dim, out_dim)

        # 正确初始化：gamma → I，beta → 0
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
        with torch.no_grad():
            # gamma_head.bias = flatten(I)，使初期 gamma ≈ I
            self.gamma_head.bias.copy_(
                torch.eye(osc_dim).flatten()
            )

    def forward(self, z_global):
        """
        z_global: (B, global_dim)
        → gamma: (B, osc_dim, osc_dim)
        → beta:  (B, osc_dim, osc_dim)
        """
        h     = self.net(z_global)
        gamma = self.gamma_head(h).view(-1, self.osc_dim, self.osc_dim)
        beta  = self.beta_head(h).view(-1, self.osc_dim, self.osc_dim)
        return gamma, beta


# ─────────────────────────────────────────────
# 4. 条件化Koopman算子（加入能量归一化）
# ─────────────────────────────────────────────
class ConditionedKoopmanOperator(nn.Module):
    """
    K(z_global) = gamma ⊙ K_base + beta
    K_base 通过Cayley变换保证正交性（谱范数=1）

    修复：多步传播后加入能量归一化，防止范数衰减到零
    """
    def __init__(self, osc_dim):
        super().__init__()
        self.osc_dim = osc_dim
        self.A = nn.Parameter(torch.randn(osc_dim, osc_dim) * 0.01)

    def get_K_base(self):
        """返回基础Koopman矩阵 (d, d)，正交"""
        return cayley_map(self.A.unsqueeze(0)).squeeze(0)

    def forward(self, z_osc_t, gamma, beta, steps=1):
        """
        z_osc_t: (B, d)   — 起始状态
        gamma:   (B, d, d) — FiLM增益
        beta:    (B, d, d) — FiLM偏置
        steps:   int       — 传播步数h

        修复：每步传播后做能量归一化
              norm(z_new) = norm(z_old)，防止坍塌
        """
        K_base = self.get_K_base()                          # (d, d)
        K_cond = gamma * K_base.unsqueeze(0) + beta        # (B, d, d)

        z = z_osc_t                                         # (B, d)
        for _ in range(steps):
            z_new = torch.bmm(
                z.unsqueeze(1), K_cond.transpose(1, 2)
            ).squeeze(1)                                    # (B, d)

            # ── 能量归一化（修复核心）──────────────────────
            norm_in  = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            norm_out = z_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            z = z_new * (norm_in / norm_out)
            # ────────────────────────────────────────────────

        return z


# ─────────────────────────────────────────────
# 5. 完整Koopman Predictor
# ─────────────────────────────────────────────
class KoopmanPredictor(nn.Module):
    """
    替换TS-JEPA中的Transformer Predictor。

    修复：
      - embed_dim 与 config 对齐（384）
      - inv_dim/osc_dim 通过参数传入，不再硬编码
      - FiLM初始化修复
      - 能量归一化防坍塌
    """
    def __init__(
        self,
        num_patches,
        encoder_embed_dim = 128,
        inv_dim           = 64,
        osc_dim           = 64,
    ):
        super().__init__()
        assert inv_dim + osc_dim == encoder_embed_dim, \
            f"inv_dim({inv_dim}) + osc_dim({osc_dim}) != embed_dim({encoder_embed_dim})"

        self.num_patches        = num_patches
        self.encoder_embed_dim  = encoder_embed_dim
        self.inv_dim            = inv_dim
        self.osc_dim            = osc_dim

        # 子空间分解器
        self.decoupler = PatchDecoupler(encoder_embed_dim, inv_dim, osc_dim)

        # 全局语义提取
        self.global_pool = nn.Sequential(
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
            nn.LayerNorm(encoder_embed_dim),
            nn.GELU(),
        )

        # FiLM超网络（修复版）
        self.film_net = FiLMHyperNet(
            global_dim = encoder_embed_dim,
            osc_dim    = osc_dim,
            hidden_dim = 128,
        )

        # 条件化Koopman算子（修复版）
        self.koopman = ConditionedKoopmanOperator(osc_dim)

        # 投影回encoder_embed_dim
        self.output_proj = nn.Linear(inv_dim + osc_dim, encoder_embed_dim)

    def forward(self, encoded_vals, mask=None, non_masks=None, masked_positions=None):
        """
        encoded_vals: (B, num_visible, embed_dim)
        返回: z_pred  (B, num_masked, embed_dim)
        """
        B, num_visible, D = encoded_vals.shape

        # 1. 全局语义
        z_global = self.global_pool(encoded_vals.mean(dim=1))  # (B, D)

        # 2. FiLM调制参数
        gamma, beta = self.film_net(z_global)                  # (B, d, d)

        # 3. 子空间分解
        z_inv, z_osc = self.decoupler(encoded_vals)
        # z_inv: (B, num_visible, inv_dim)
        # z_osc: (B, num_visible, osc_dim)

        # 4. 以最后一个可见patch为起点
        z_inv_last = z_inv[:, -1, :]   # (B, inv_dim)
        z_osc_last = z_osc[:, -1, :]   # (B, osc_dim)

        # 5. 逐步预测每个masked位置
        num_masked = mask.shape[1] if mask is not None \
                     else self.num_patches - num_visible

        preds = []
        for h in range(1, num_masked + 1):
            # 不变子空间：恒等传播
            z_inv_pred = z_inv_last                            # (B, inv_dim)

            # 动力学子空间：Koopman传播h步（含能量归一化）
            z_osc_pred = self.koopman(
                z_osc_last, gamma, beta, steps=h
            )                                                  # (B, osc_dim)

            # 拼接 + 投影
            z_pred_h = self.output_proj(
                torch.cat([z_inv_pred, z_osc_pred], dim=-1)
            )                                                  # (B, embed_dim)
            preds.append(z_pred_h)

        return torch.stack(preds, dim=1)                      # (B, num_masked, D)

    def get_decoupled(self, z):
        """
        暴露分解接口，供损失函数使用
        z: (B, T, embed_dim)
        返回: z_inv (B, T, inv_dim), z_osc (B, T, osc_dim)
        """
        return self.decoupler(z)