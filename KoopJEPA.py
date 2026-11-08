"""
KoopJEPA: JEPA + Koopman Generative Augmentation Framework
===========================================================
References:
  [1] NeuralKoopmanSVD (NeurIPS 2025): https://github.com/MinchanJeong/NeuralKoopmanSVD
  [2] PyTorch Spectral Norm: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html
  [3] T-JEPA (ICLR 2025): https://openreview.net/pdf?id=gx3LMRB15C
  [4] Koopman & SSM relationship (arXiv:2411.12940)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import math


# ============================================================
# 0. Utility: Cayley Transform (Antisymmetric → Orthogonal)
# ============================================================

def cayley_transform(A: Tensor) -> Tensor:
    """
    Maps a skew-symmetric matrix A to an orthogonal matrix Q.
    Q = (I - A)(I + A)^{-1}
    Guarantees Q^T Q = I, used for ODO parameterization of K.
    Ref: NeuralKoopmanSVD [1]
    """
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    # Ensure skew-symmetry
    A_skew = (A - A.transpose(-2, -1)) / 2.0
    Q = torch.linalg.solve(I + A_skew, I - A_skew)
    return Q


def make_skew_symmetric(A: Tensor) -> Tensor:
    """Force a matrix to be skew-symmetric: (A - A^T) / 2"""
    return (A - A.transpose(-2, -1)) / 2.0


# ============================================================
# 1. ODO Koopman Propagator (Core Module)
# ============================================================

class KoopmanPropagator(nn.Module):
    """
    Spectrally Constrained Koopman Propagator via ODO Decomposition.

    K = U * Sigma * V^T
      - U, V: orthogonal matrices via Cayley transform
      - Sigma: diagonal with entries in (0, sigma_max]
      - Hard spectral constraint: rho(K) <= spectral_radius_bound
      - Soft Lyapunov regularization: ||K^T K - I||_F

    Ref: NeuralKoopmanSVD (NeurIPS 2025) [1]
         DeepKoopFormer spectral constraint design
    """

    def __init__(
        self,
        latent_dim: int,
        spectral_radius_bound: float = 0.99,
        lyapunov_lambda: float = 1e-3,
        lyapunov_epsilon: float = 0.1,
    ):
        super().__init__()
        self.d = latent_dim
        self.rho_max = spectral_radius_bound
        self.lyapunov_lambda = lyapunov_lambda
        self.lyapunov_epsilon = lyapunov_epsilon

        # Learnable skew-symmetric parameters → Cayley → orthogonal U, V
        self.A_U = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.A_V = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)

        # Learnable log-singular values (ensure positivity via softplus)
        self.log_sigma = nn.Parameter(torch.zeros(latent_dim))

    def get_K(self) -> Tensor:
        """
        Construct the Koopman matrix K = U * Sigma * V^T
        with hard spectral radius constraint.
        """
        U = cayley_transform(make_skew_symmetric(self.A_U))   # [d, d] orthogonal
        V = cayley_transform(make_skew_symmetric(self.A_V))   # [d, d] orthogonal

        # Singular values: softplus ensures positivity, clamp ensures rho <= bound
        sigma = F.softplus(self.log_sigma)                    # [d]
        sigma = torch.clamp(sigma, max=self.rho_max)          # Hard constraint

        Sigma = torch.diag(sigma)                             # [d, d]
        K = U @ Sigma @ V.t()                                 # [d, d]
        return K

    def forward(self, z: Tensor, steps: int = 1) -> Tensor:
        """
        Propagate latent state z forward by `steps` steps.
        z: [B, d]  →  z_{t+steps}: [B, d]
        """
        K = self.get_K()
        # Efficient multi-step: z_{t+k} = K^k z_t
        # Use matrix power for small steps, iterative for large
        if steps == 1:
            return z @ K.t()
        else:
            K_power = torch.linalg.matrix_power(K, steps)
            return z @ K_power.t()

    def propagate_sequence(self, z0: Tensor, horizon: int) -> Tensor:
        """
        Generate full trajectory from initial state z0.
        z0: [B, d]  →  trajectory: [B, horizon, d]
        """
        K = self.get_K()
        traj = [z0]
        z = z0
        for _ in range(horizon - 1):
            z = z @ K.t()
            traj.append(z)
        return torch.stack(traj, dim=1)   # [B, horizon, d]

    def lyapunov_loss(self) -> Tensor:
        """
        Lyapunov stability regularization:
        L_stab = lambda * max(0, ||K^T K - I||_F - epsilon)
        Penalizes transient energy growth in latent space.
        Ref: DeepKoopFormer Lyapunov regularization
        """
        K = self.get_K()
        I = torch.eye(self.d, device=K.device, dtype=K.dtype)
        residual = torch.norm(K.t() @ K - I, p='fro')
        loss = self.lyapunov_lambda * F.relu(residual - self.lyapunov_epsilon)
        return loss


# ============================================================
# 2. Disentangled Encoder (z_inv + z_osc)
# ============================================================

class DisentangledEncoder(nn.Module):
    """
    Transformer-based encoder that produces disentangled latent codes:
      - z_inv: invariant/regime representation (slow dynamics)
      - z_osc: oscillatory mode representation (fast dynamics)

    Architecture follows JEPA context encoder design.
    Ref: T-JEPA (ICLR 2025) [3], TS-JEPA (arXiv:2509.25449)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        latent_dim: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_seq_len)

        # Transformer encoder (causal mask for temporal modeling)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection heads for disentanglement
        self.proj_inv = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.proj_osc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x: [B, T, input_dim]
        Returns:
          h:     [B, T, d_model]  full hidden states
          z_inv: [B, latent_dim]  invariant representation (mean pooled)
          z_osc: [B, T, latent_dim] oscillatory per-timestep representation
        """
        # Input projection + positional encoding
        h = self.input_proj(x)                        # [B, T, d_model]
        h = self.pos_encoding(h)

        # Causal mask for autoregressive-style encoding
        T = x.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device
        )

        h = self.transformer(h, mask=causal_mask,
                             src_key_padding_mask=mask)  # [B, T, d_model]

        # Disentangle
        z_inv = self.proj_inv(h.mean(dim=1))           # [B, latent_dim] — global
        z_osc = self.proj_osc(h)                       # [B, T, latent_dim] — local

        return h, z_inv, z_osc


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        T = x.shape[1]
        positions = torch.arange(T, device=x.device)
        return x + self.pe(positions).unsqueeze(0)


# ============================================================
# 3. JEPA Predictor
# ============================================================

class JEPAPredictor(nn.Module):
    """
    Narrow Transformer predictor that maps context latents
    to target latents in embedding space.

    Key: predictor is intentionally NARROW (smaller than encoder)
    to prevent trivial identity solutions.
    Ref: I-JEPA / T-JEPA design principle [3]
    """

    def __init__(
        self,
        latent_dim: int,
        predictor_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(latent_dim, predictor_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=nhead,
            dim_feedforward=predictor_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(predictor_dim, latent_dim)

    def forward(self, z_ctx: Tensor) -> Tensor:
        """
        z_ctx: [B, T_ctx, latent_dim]
        Returns predicted target: [B, T_ctx, latent_dim]
        """
        h = self.input_proj(z_ctx)
        h = self.transformer(h)
        return self.output_proj(h)


# ============================================================
# 4. Decoder (Latent → Time Series)
# ============================================================

class TimeSeriesDecoder(nn.Module):
    """
    Decodes latent Koopman states back to observation space.
    Used for generative augmentation sample reconstruction.
    Ref: Koopman Autoencoder continuous-time [4]
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """
        z: [B, T, latent_dim]  →  x̃: [B, T, output_dim]
        """
        return self.net(z)


# ============================================================
# 5. Latent Augmentation Strategies
# ============================================================

class KoopmanLatentAugmentor(nn.Module):
    """
    Three latent-space augmentation strategies using the
    learned Koopman propagator.

    ① Extrapolation:    z̃_{T+k} = K^k z_T
    ② Interpolation:    z̃_α = α·z_A + (1-α)·z_B
    ③ Mode Perturbation: z̃ = z_inv + (z_osc + δ), δ~N(0,σ²I)

    Ref: KoVAE generative design [2], DeepKoopFormer propagation
    """

    def __init__(self, propagator: KoopmanPropagator, noise_std: float = 0.05):
        super().__init__()
        self.propagator = propagator
        self.noise_std = noise_std

    def extrapolate(self, z_T: Tensor, horizon: int) -> Tensor:
        """
        Strategy ①: Future trajectory synthesis.
        z_T: [B, latent_dim]  →  z̃: [B, horizon, latent_dim]
        Suitable for: long-horizon forecasting augmentation.
        """
        return self.propagator.propagate_sequence(z_T, horizon)

    def interpolate(
        self,
        z_A: Tensor,
        z_B: Tensor,
        n_interp: int = 5,
    ) -> Tensor:
        """
        Strategy ②: Between-trajectory mixing in Koopman space.
        z_A, z_B: [B, latent_dim]
        Returns: [B, n_interp, latent_dim]
        Suitable for: diversity augmentation, rare event synthesis.
        """
        alphas = torch.linspace(0, 1, n_interp, device=z_A.device)
        interps = []
        for alpha in alphas:
            z_interp = alpha * z_A + (1 - alpha) * z_B
            interps.append(z_interp)
        return torch.stack(interps, dim=1)   # [B, n_interp, latent_dim]

    def mode_perturbation(
        self,
        z_inv: Tensor,
        z_osc: Tensor,
        n_samples: int = 4,
    ) -> Tensor:
        """
        Strategy ③: Regime-preserving oscillatory perturbation.
        Fixes z_inv (system regime), perturbs z_osc (dynamics).
        z_inv: [B, latent_dim]
        z_osc: [B, T, latent_dim]
        Returns: [B*n_samples, T, latent_dim]
        Suitable for: same-class diverse sample generation.
        """
        B, T, d = z_osc.shape
        z_inv_exp = z_inv.unsqueeze(1).expand(-1, T, -1)  # [B, T, d]

        aug_samples = []
        for _ in range(n_samples):
            delta = torch.randn_like(z_osc) * self.noise_std
            z_aug = z_inv_exp + (z_osc + delta)
            aug_samples.append(z_aug)

        return torch.cat(aug_samples, dim=0)   # [B*n_samples, T, d]

    def forward(
        self,
        z_inv: Tensor,
        z_osc: Tensor,
        strategy: str = 'all',
        horizon: int = 24,
    ) -> dict:
        """
        Run all augmentation strategies and return dict of results.
        """
        B, T, d = z_osc.shape
        z_T = z_osc[:, -1, :]   # Last timestep as initial state

        results = {}
        if strategy in ('extrapolate', 'all'):
            results['extrap'] = self.extrapolate(z_T, horizon)

        if strategy in ('interpolate', 'all') and B >= 2:
            # Pair consecutive samples for interpolation
            z_A = z_osc[:B//2, -1, :]
            z_B = z_osc[B//2:B//2*2, -1, :]
            results['interp'] = self.interpolate(z_A, z_B)

        if strategy in ('perturb', 'all'):
            results['perturb'] = self.mode_perturbation(z_inv, z_osc)

        return results


# ============================================================
# 6. Disentanglement Loss (Mutual Information Minimization)
# ============================================================

class DisentanglementLoss(nn.Module):
    """
    Approximate mutual information minimization between z_inv and z_osc
    via HSIC (Hilbert-Schmidt Independence Criterion) — efficient & differentiable.

    L_disent = HSIC(z_inv, z_osc_mean)

    Ref: Independence criterion for Koopman disentanglement [4]
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def rbf_kernel(self, X: Tensor) -> Tensor:
        """Radial basis function kernel matrix."""
        sq_dists = torch.cdist(X, X, p=2).pow(2)
        return torch.exp(-sq_dists / (2 * self.sigma ** 2))

    def forward(self, z_inv: Tensor, z_osc: Tensor) -> Tensor:
        """
        z_inv: [B, d]
        z_osc: [B, T, d] → mean pooled to [B, d]
        Returns scalar HSIC loss.
        """
        z_osc_mean = z_osc.mean(dim=1)   # [B, d]
        B = z_inv.shape[0]

        K_inv = self.rbf_kernel(z_inv)    # [B, B]
        K_osc = self.rbf_kernel(z_osc_mean)  # [B, B]

        # Center kernel matrices
        H = torch.eye(B, device=z_inv.device) - torch.ones(B, B, device=z_inv.device) / B
        K_inv_c = H @ K_inv @ H
        K_osc_c = H @ K_osc @ H

        hsic = torch.trace(K_inv_c @ K_osc_c) / (B - 1) ** 2
        return hsic


# ============================================================
# 7. Full KoopJEPA Model
# ============================================================

class KoopJEPA(nn.Module):
    """
    Full KoopJEPA Model:
      JEPA Self-Supervised Pretraining
      + Spectrally Constrained Koopman Propagation
      + Latent Generative Augmentation

    Two-stage training:
      Stage 1: Self-supervised pretraining with L_JEPA + L_stab + L_disent
      Stage 2: Downstream fine-tuning with augmented samples

    Ref:
      [1] NeuralKoopmanSVD (NeurIPS 2025)
      [2] KoVAE (ICLR 2024)
      [3] T-JEPA (ICLR 2025)
      [4] Koopman-SSM relationship (arXiv:2411.12940)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        d_model: int = 128,
        nhead: int = 8,
        enc_layers: int = 4,
        pred_layers: int = 2,
        dec_hidden: int = 256,
        spectral_radius_bound: float = 0.99,
        lyapunov_lambda: float = 1e-3,
        disent_lambda: float = 1e-2,
        ema_momentum: float = 0.996,
        mask_ratio: float = 0.5,
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum
        self.mask_ratio = mask_ratio
        self.disent_lambda = disent_lambda

        # ── Encoders ──────────────────────────────────────────
        self.context_encoder = DisentangledEncoder(
            input_dim, d_model, nhead, enc_layers, latent_dim
        )
        # Target encoder: EMA copy, no gradient
        self.target_encoder = DisentangledEncoder(
            input_dim, d_model, nhead, enc_layers, latent_dim
        )
        self._init_target_encoder()

        # ── Predictor ─────────────────────────────────────────
        self.predictor = JEPAPredictor(latent_dim, latent_dim // 2, nhead // 2, pred_layers)

        # ── Koopman Propagator ────────────────────────────────
        self.propagator = KoopmanPropagator(
            latent_dim, spectral_radius_bound, lyapunov_lambda
        )

        # ── Augmentor ─────────────────────────────────────────
        self.augmentor = KoopmanLatentAugmentor(self.propagator, noise_std)

        # ── Decoder ───────────────────────────────────────────
        self.decoder = TimeSeriesDecoder(latent_dim, input_dim, dec_hidden)

        # ── Losses ────────────────────────────────────────────
        self.disent_loss_fn = DisentanglementLoss()

    @torch.no_grad()
    def _init_target_encoder(self):
        """Initialize target encoder as exact copy of context encoder."""
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_tgt.data.copy_(p_ctx.data)
            p_tgt.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self):
        """
        EMA update: θ̄ ← τ·θ̄ + (1-τ)·θ
        Called after each optimizer step.
        Ref: T-JEPA / I-JEPA EMA design [3]
        """
        tau = self.ema_momentum
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(tau).add_(p_ctx.data, alpha=1 - tau)

    def temporal_mask(self, T: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Random temporal masking: split T timesteps into context/target.
        Returns boolean masks of shape [T].
        """
        n_mask = int(T * self.mask_ratio)
        mask_idx = torch.randperm(T, device=device)[:n_mask]
        target_mask = torch.zeros(T, dtype=torch.bool, device=device)
        target_mask[mask_idx] = True
        context_mask = ~target_mask
        return context_mask, target_mask

    def forward(
        self,
        x: Tensor,
        return_aug: bool = False,
        aug_horizon: int = 24,
    ) -> dict:
        """
        Full forward pass.
        x: [B, T, input_dim]

        Returns dict with losses and optionally augmented samples.
        """
        B, T, _ = x.shape
        ctx_mask, tgt_mask = self.temporal_mask(T, x.device)

        x_ctx = x[:, ctx_mask, :]   # [B, T_ctx, input_dim]
        x_tgt = x[:, tgt_mask, :]   # [B, T_tgt, input_dim]

        # ── Context encoding ──────────────────────────────────
        _, z_inv, z_osc_ctx = self.context_encoder(x_ctx)
        # z_inv:     [B, latent_dim]
        # z_osc_ctx: [B, T_ctx, latent_dim]

        # ── JEPA prediction in latent space ───────────────────
        z_pred = self.predictor(z_osc_ctx)   # [B, T_ctx, latent_dim]

        # ── Target encoding (stop-gradient EMA) ───────────────
        with torch.no_grad():
            _, _, z_osc_tgt = self.target_encoder(x_tgt)
            # Align prediction length to target length
            T_pred = min(z_pred.shape[1], z_osc_tgt.shape[1])
            z_tgt_sg = z_osc_tgt[:, :T_pred, :].detach()

        # ── JEPA Loss ─────────────────────────────────────────
        z_pred_aligned = z_pred[:, :T_pred, :]
        L_jepa = F.mse_loss(z_pred_aligned, z_tgt_sg)

        # ── Koopman propagation for reconstruction ────────────
        z_T = z_osc_ctx[:, -1, :]                          # [B, latent_dim]
        z_koopman_traj = self.propagator.propagate_sequence(
            z_T, T_pred
        )                                                   # [B, T_pred, latent_dim]

        # ── Reconstruction Loss ───────────────────────────────
        x_recon = self.decoder(z_koopman_traj)             # [B, T_pred, input_dim]
        L_recon = F.mse_loss(x_recon, x_tgt[:, :T_pred, :])

        # ── Lyapunov Stability Loss ───────────────────────────
        L_stab = self.propagator.lyapunov_loss()

        # ── Disentanglement Loss ──────────────────────────────
        L_disent = self.disent_loss_fn(z_inv, z_osc_ctx)

        # ── Total Loss ────────────────────────────────────────
        L_total = L_jepa + L_recon + L_stab + self.disent_lambda * L_disent

        output = {
            'loss':       L_total,
            'loss_jepa':  L_jepa.item(),
            'loss_recon': L_recon.item(),
            'loss_stab':  L_stab.item(),
            'loss_disent':L_disent.item(),
            'z_inv':      z_inv,
            'z_osc':      z_osc_ctx,
        }

        # ── Generative Augmentation ───────────────────────────
        if return_aug:
            aug_dict = self.augmentor(z_inv, z_osc_ctx, strategy='all',
                                      horizon=aug_horizon)
            # Decode augmented latents to observation space
            aug_samples = {}
            for key, z_aug in aug_dict.items():
                if z_aug.dim() == 2:
                    z_aug = z_aug.unsqueeze(1)
                aug_samples[key] = self.decoder(z_aug)
            output['aug_samples'] = aug_samples

        return output


# ============================================================
# 8. Training Loop
# ============================================================

class KoopJEPATrainer:
    """
    Two-stage trainer:
      Stage 1: Self-supervised pretraining (no labels)
      Stage 2: Downstream fine-tuning with augmented data

    Ref: T-JEPA two-stage training protocol [3]
         NeuralKoopmanSVD optimization strategy [1]
    """

    def __init__(
        self,
        model: KoopJEPA,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )

    def pretrain_step(self, x: Tensor) -> dict:
        """Single pretraining step (Stage 1)."""
        self.model.train()
        x = x.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(x, return_aug=False)
        output['loss'].backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # EMA update for target encoder
        self.model.update_target_encoder()

        return {k: v for k, v in output.items() if 'loss' in k}

    def generate_augmented_batch(
        self,
        x: Tensor,
        strategy: str = 'all',
        aug_horizon: int = 24,
    ) -> dict:
        """
        Generate augmented samples for downstream fine-tuning (Stage 2).
        Returns dict of augmented time series tensors.
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x, return_aug=True, aug_horizon=aug_horizon)
        return output.get('aug_samples', {})

    def pretrain(self, dataloader, epochs: int = 100, log_every: int = 10):
        """Full pretraining loop."""
        for epoch in range(epochs):
            epoch_losses = {
                'loss_jepa': 0., 'loss_recon': 0.,
                'loss_stab': 0., 'loss_disent': 0.
            }
            n_batches = 0

            for x, *_ in dataloader:
                losses = self.pretrain_step(x)
                for k in epoch_losses:
                    epoch_losses[k] += losses.get(k, 0.)
                n_batches += 1

            if (epoch + 1) % log_every == 0:
                avg = {k: v / n_batches for k, v in epoch_losses.items()}
                print(
                    f"Epoch [{epoch+1:4d}/{epochs}] "
                    f"JEPA: {avg['loss_jepa']:.4f} | "
                    f"Recon: {avg['loss_recon']:.4f} | "
                    f"Stab: {avg['loss_stab']:.4f} | "
                    f"Disent: {avg['loss_disent']:.4f}"
                )


# ============================================================
# 9. Quick Test
# ============================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Model instantiation
    model = KoopJEPA(
        input_dim=7,
        latent_dim=64,
        d_model=128,
        nhead=8,
        enc_layers=4,
        spectral_radius_bound=0.99,
        lyapunov_lambda=1e-3,
        disent_lambda=1e-2,
        mask_ratio=0.5,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"KoopJEPA total parameters: {total_params:,}\n")

    # Dummy batch: B=8, T=96, input_dim=7
    x_dummy = torch.randn(8, 96, 7)

    # Stage 1: Pretraining forward pass
    output = model(x_dummy, return_aug=True, aug_horizon=24)
    print("=== Stage 1: Pretraining ===")
    print(f"  Total Loss:       {output['loss'].item():.4f}")
    print(f"  JEPA Loss:        {output['loss_jepa']:.4f}")
    print(f"  Recon Loss:       {output['loss_recon']:.4f}")
    print(f"  Stability Loss:   {output['loss_stab']:.4f}")
    print(f"  Disentangle Loss: {output['loss_disent']:.4f}")
    print(f"  z_inv shape:      {output['z_inv'].shape}")
    print(f"  z_osc shape:      {output['z_osc'].shape}")

    # Stage 2: Augmented samples
    print("\n=== Stage 2: Generative Augmentation ===")
    for key, aug in output['aug_samples'].items():
        print(f"  [{key}] augmented sample shape: {aug.shape}")

    # Koopman matrix spectral radius check
    K = model.propagator.get_K()
    eigvals = torch.linalg.eigvals(K)
    rho = eigvals.abs().max().item()
    print(f"\n=== Koopman Spectral Check ===")
    print(f"  K shape:          {K.shape}")
    print(f"  Spectral radius:  {rho:.6f}  (should be <= 0.99 ✓)")