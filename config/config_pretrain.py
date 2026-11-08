config = {

    "wandb_project_name": "",
    "log_wandb": False,

    "batch_size": 32,

    # Printing and Logging settings
    "checkpoint_save":  5000,
    "checkpoint_print": 30,

    # Loader
    "mask_ratio":    0.7,
    "ratio_patches": 10,
    "clip_grad":     10,
    "warmup_ratio":  0.15,
    "ipe_scale":     1.25,

    # Optim
    "lr":     1e-7,
    "end_lr": 1e-4,

    "num_epochs":   5001,
    "ema_momentum": 0.998,

    # Encoder（与命令行保持一致）
    "encoder_embed_dim":   128,
    "encoder_nhead":         2,
    "encoder_num_layers":    1,
    "encoder_kernel_size":   3,
    "encoder_embed_bias":  True,

    # 原始Predictor参数（保留）
    "predictor_embed":      128,
    "predictor_nhead":        2,
    "predictor_num_layers":   1,

    # ── Koopman Predictor 专属参数 ──────────────────────
    # inv_dim + osc_dim == encoder_embed_dim == 128
    "koopman_inv_dim": 64,   # 不变子空间（原32→64，提升表达能力）
    "koopman_osc_dim": 64,   # 动力学子空间（原32→64，修复坍塌核心）

    # 损失权重
    "lambda_dyn": 1.0,   # 动力学子空间对齐
    "mu_orth":    0.01,  # 正交性约束
    "nu_var":     0.1,   # 方差惩罚（防坍塌）
}