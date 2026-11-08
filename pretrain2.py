"""
    Script to pretrain a TS-JEPA before using it on the forecasting downstream.
"""

import warnings

warnings.filterwarnings("ignore")

import time
import copy
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from main.utils import prepare_args_pretrain, init_weights, grad_logger

from src.models.encoder import Encoder
from src.models.predictor import Predictor

from src.models.utils.mask_utils import apply_mask
from src.data_loaders.data_loader import get_jepa_loaders

from config.config_pretrain import config


def loss_pred(pred, target_ema):
    loss = 0.0
    for pred_i, target_ema_i in zip(pred, target_ema):
        loss = loss + torch.mean(torch.abs(pred_i - target_ema_i))
    loss /= len(pred)
    return loss


def save_model(model, epoch):
    save_dict = {"encoder": model.state_dict(), "epoch": epoch}

    path_name = path_save + "_epoch_" + str(epoch) + ".pt"
    torch.save(save_dict, path_name)
    # except:
    #     print("Problem saving checkpoint")


# Define the custom learning rate schedule
def lr_lambda(epoch):
    start_lr = config["lr"]
    end_lr = config["end_lr"]
    if epoch < config["num_epochs"]:
        return start_lr + (end_lr - start_lr) * (epoch / (config["num_epochs"] - 1))
    else:
        return end_lr


if __name__ == "__main__":
    # Load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init config and args
    config = prepare_args_pretrain(config)
    if config["data"] == "fordA":
        from src.data_loaders.data_loader import get_forda_loaders
        train_loader, _ = get_forda_loaders(
            path_train="./data/fordA/fordA_TRAIN.tsv",
            path_test="./data/fordA/fordA_TEST.tsv",
            batch_size=config["batch_size"],
            patch_size=50,           # 500 / 10 patches = 50
            mask_ratio=config["mask_ratio"]
        )
        loader = train_loader
    else:
        

        # Load Data
        loader = get_jepa_loaders(
            config["path_data"],
            config["batch_size"],
            config["ratio_patches"],
            config["mask_ratio"],
        )

    input_dim = len(loader.dataset[0][0][0])

    # Load Encoder
    encoder = Encoder(
        num_patches=len(loader.dataset[0][0]),
        dim_in=input_dim,
        kernel_size=config["encoder_kernel_size"],
        embed_dim=config["encoder_embed_dim"],
        embed_bias=config["encoder_embed_bias"],
        nhead=config["encoder_nhead"],
        num_layers=config["encoder_num_layers"],
        jepa=True,
    )

    # Load Predictor
    predictor = Predictor(
        num_patches=len(loader.dataset[0][0]),
        encoder_embed_dim=config["encoder_embed_dim"],
        predictor_embed_dim=config["predictor_embed"],
        nhead=config["predictor_nhead"],
        num_layers=config["predictor_num_layers"],
    )

    # Init weights -- Similar to VJEPA
    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    param_groups = [
        {"params": (p for n, p in encoder.named_parameters())},
        {"params": (p for n, p in predictor.named_parameters())},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])

    # Initialize the scheduler
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.5, total_iters=config["num_epochs"]
    )

    encoder = encoder.to(device)
    predictor = predictor.to(device)

    # Initialize the EMA-Encoder
    encoder_ema = copy.deepcopy(encoder)

    # Stop-gradient step in the EMA
    for p in encoder_ema.parameters():
        p.requires_grad = False

    checkpoint_save = config["checkpoint_save"]
    checkpoint_print = config["checkpoint_print"]
    path_save = config["path_save"]
    clip_grad = config["clip_grad"]
    warmup = config["warmup_ratio"] * config["num_epochs"]

    # Initialize the EMA Scheduler (parameter m in the paper)
    ema_scheduler = (
        config["ema_momentum"]
        + i
        * (1 - config["ema_momentum"])
        / (config["num_epochs"] * config["ipe_scale"])
        for i in range(int(config["num_epochs"] * config["ipe_scale"]) + 1)
    )

    num_batches = len(loader)

    # total_loss, total_var_encoder, total_var_decoder = 0.0, 0.0, 0.0


    # ---- 修改后 ----

    # [改动1] 在循环外，增加 collapse 监控的累计变量
    total_loss = 0.0
    total_var_encoder = 0.0   # 原来就有，现在真正用起来
    total_var_decoder = 0.
    collapse_threshold = 1e-3  # ← 低于此值视为 collapse 预警

    # Save Initial Model -- Useful to compare when evaluating
    save_model(encoder, 0)

    for epoch in range(config["num_epochs"]):
        scheduler.step()
        m = next(ema_scheduler)
        encoder.train()
        predictor.train()

        for patches, masks, non_masks , *_ in loader:
            optimizer.zero_grad()
            patches   = patches.to(device)
            masks     = masks.to(device)
            non_masks = non_masks.to(device)

            with torch.no_grad():
                target_ema = encoder_ema(patches)
                target_ema = F.layer_norm(target_ema, (target_ema.size(-1),))
                target_ema = apply_mask(target_ema, masks)

            tokens = encoder(patches, mask=non_masks)
            pred   = predictor(tokens, mask=masks, non_masks=non_masks)

            loss = loss_pred(pred, target_ema)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for param_q, param_k in zip(encoder.parameters(), encoder_ema.parameters()):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

            # -------------------------------------------------------
            # [改动2] 在 EMA 更新之后，batch 循环内计算表征方差
            # tokens: shape [B, N_visible, D]  ← encoder 对可见 patch 的输出
            # pred:   shape [B, N_masked,  D]  ← predictor 对遮掩 patch 的预测
            with torch.no_grad():
                # 在 batch 维度上计算每个特征维度的方差，再取均值
                # 含义：这批样本的表征有多"分散"，趋近 0 = collapse
                var_encoder = tokens.var(dim=0).mean().item()  # 编码器输出的方差
                var_pred    = pred.var(dim=0).mean().item()    # predictor 输出的方差

            total_var_encoder += var_encoder
            total_var_decoder += var_pred
            # -------------------------------------------------------

            total_loss += loss

        # [改动3] epoch 结束后归一化
        total_loss        = total_loss        / num_batches
        total_var_encoder = total_var_encoder / num_batches
        total_var_decoder = total_var_decoder / num_batches

        if epoch % 10 == 0:
            # [改动4] 打印时加入方差，并给出 collapse 预警
            collapse_flag = "⚠️ COLLAPSE?" if total_var_encoder < collapse_threshold else "✅"
            print(
                f"Epoch {epoch:4d} | "
                f"lr: {optimizer.param_groups[0]['lr']:.3g} | "
                f"Loss: {total_loss:.4f} | "
                f"Var(encoder): {total_var_encoder:.5f} {collapse_flag} | "
                f"Var(pred): {total_var_decoder:.5f} | "
                f"EMA m: {m:.5f}"
            )

        # [改动5] 每个 epoch 重置累计变量
        total_loss, total_var_encoder, total_var_decoder = 0.0, 0.0, 0.0


        # Save model's checkpoint
        if epoch % checkpoint_save == 0 and epoch != 0:
            save_model(encoder, epoch)
