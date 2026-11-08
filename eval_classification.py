"""
Script to run the Classification downstream task on FordA.
---
    Strategy: Linear Probing
        - Load pretrained Encoder (frozen)
        - Train only a linear classification head
        - Evaluate on test set
"""

from config.config_downstream import config
from main.utils import prepare_args

import torch
import torch.nn as nn
import numpy as np
import argparse

from src.data_loaders.data_loader_classification import get_eval_loaders
from src.models.encoder import Encoder

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Pooling helper: mean-pool over patch dimension
# ─────────────────────────────────────────────
def pool_embeddings(encoded, mode="mean"):
    """
    encoded: (B, num_patches, embed_dim)
    returns: (B, embed_dim)
    """
    if mode == "mean":
        return encoded.mean(dim=1)
    elif mode == "sum":
        return encoded.sum(dim=1)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────
def evaluate(encoder, classifier, loader, device, pool_mode="mean"):
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = pool_embeddings(encoder(x), mode=pool_mode)
            logits = classifier(z)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # --- Parse args (reuse existing arg parser) ---
    config = prepare_args(config)

    # --- Hyperparameters ---
    num_epochs  = 50       # Linear probing converges fast
    pool_mode   = "mean"   # "mean" or "sum"
    num_classes = 2        # FordA is binary

    # --- Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    path_train = f"./data/fordA/fordA_TRAIN.tsv"
    path_test  = f"./data/fordA/fordA_TEST.tsv"

    train_loader, test_loader = get_eval_loaders(
        path_train, path_test,
        batch_size=config["batch_size"],
        ratio_patches=config["ratio_patches"]
    )

    # --- Infer dims from data ---
    sample_x, _ = train_loader.dataset[0]
    num_patches = sample_x.shape[0]   # 10
    patch_size  = sample_x.shape[1]   # 50  (FordA: 500/10)
    print(f"num_patches={num_patches}, patch_size(dim_in)={patch_size}")

    # --- Build Encoder (same architecture as pretrain) ---
    encoder = Encoder(
        num_patches = num_patches,
        dim_in      = patch_size,                           # 50 for FordA
        kernel_size = config["pretrain_encoder_kernel_size"],
        embed_dim   = config["pretrain_encoder_embed_dim"],
        embed_bias  = config["pretrain_encoder_embed_bias"],
        nhead       = config["pretrain_encoder_nhead"],
        num_layers  = config["pretrain_encoder_num_layers"],
        jepa        = False,   # No masking during evaluation
    ).to(device)

    # --- Load pretrained weights ---
    path_name = (
        "/lr_"
        + str(config["lr_pretrain"])
        + "_ema_momentum_"
        + str(config["ema_pretrain"])
        + "_mask_ratio_"
        + str(config["mask_ratio"])
        + "_ratio_patches_"
        + str(config["ratio_patches"])
        + "_encoder_"
        + str(config["pretrain_encoder_embed_dim"])
        + "_"
        + str(config["pretrain_encoder_nhead"])
        + "_"
        + str(config["pretrain_encoder_num_layers"])
        + "_predictor_"
        + str(config["pretrain_decoder_embed_dim"])
        + "_"
        + str(config["pretrain_decoder_nhead"])
        + "_"
        + str(config["pretrain_decoder_num_layers"])
        + "_epoch_"
        + str(config["checkpoint_to_use"])
    )

    checkpoint = torch.load(
        config["path_save"] + path_name + ".pt",
        map_location="cpu"
    )
    encoder.load_state_dict(checkpoint["encoder"])
    print(f"Pretrained encoder loaded from: {config['path_save'] + path_name}.pt")

    # --- Freeze Encoder (Linear Probing) ---
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    print("Encoder frozen (Linear Probing mode)")

    # --- Build Classification Head ---
    classifier = nn.Linear(
        config["pretrain_encoder_embed_dim"], num_classes
    ).to(device)

    # --- Optimizer (only classifier params) ---
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print(f"\nStart Linear Probing training ({num_epochs} epochs)...")
    best_acc = 0.0

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Frozen encoder: no grad needed
            with torch.no_grad():
                z = pool_embeddings(encoder(x), mode=pool_mode)

            logits = classifier(z)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            train_acc = evaluate(encoder, classifier, train_loader, device, pool_mode)
            test_acc  = evaluate(encoder, classifier, test_loader,  device, pool_mode)
            best_acc  = max(best_acc, test_acc)
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test  Acc: {test_acc:.4f}")

    print(f"\n{'='*50}")
    print(f"Final Test Accuracy : {test_acc:.4f}")
    print(f"Best  Test Accuracy : {best_acc:.4f}")
    print(f"EMA momentum used   : {config['ema_pretrain']}")
    print(f"{'='*50}")