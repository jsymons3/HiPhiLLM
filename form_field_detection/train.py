"""Training script for the form field detection model.

Handles the full training loop: data loading, model construction,
optimization, checkpointing, and validation metrics logging.
"""

import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import build_datasets, collate_fn
from .model import build_model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train the model for one epoch.

    Returns:
        Dict of average losses for the epoch.
    """
    model.train()
    running_losses = {}
    num_batches = 0

    progress = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in progress:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Accumulate losses for logging
        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()
        num_batches += 1

        progress.set_postfix(loss=f"{losses.item():.4f}")

    avg_losses = {k: v / num_batches for k, v in running_losses.items()}
    return avg_losses


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Run evaluation and compute average losses.

    Note: For full mAP evaluation, use the COCO evaluator in evaluate.py.
    This function provides a quick loss-based validation check.

    Returns:
        Dict of average losses.
    """
    model.train()  # Keep in train mode to get losses
    running_losses = {}
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()
        num_batches += 1

    if num_batches == 0:
        return {}

    avg_losses = {k: v / num_batches for k, v in running_losses.items()}
    return avg_losses


def train(config_path="config.yaml"):
    """Run the full training pipeline.

    Args:
        config_path: Path to YAML configuration file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Device setup
    if train_cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg["device"])
    print(f"Using device: {device}")

    # Build datasets
    train_dataset, val_dataset = build_datasets(
        data_cfg["processed_dir"],
        train_split=data_cfg["train_split"],
        augment_train=True,
    )

    if len(train_dataset) == 0:
        print("Error: No training samples found. Run extract_fields.py first.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
    )

    # Build model
    model = build_model(
        num_classes=model_cfg["num_classes"],
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
    )
    model.to(device)

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=train_cfg["learning_rate"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg["lr_step_size"],
        gamma=train_cfg["lr_gamma"],
    )

    # Checkpoint directory
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    print(f"\nStarting training for {train_cfg['num_epochs']} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        start_time = time.time()

        # Train
        train_losses = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )

        # Validate
        val_losses = evaluate(model, val_loader, device)

        lr_scheduler.step()

        # Logging
        elapsed = time.time() - start_time
        train_total = sum(train_losses.values())
        val_total = sum(val_losses.values()) if val_losses else float("nan")

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_total:.4f} | "
              f"Val Loss: {val_total:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Save checkpoint periodically
        if epoch % train_cfg["save_every"] == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "train_loss": train_total,
                "val_loss": val_total,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best model
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_classes": model_cfg["num_classes"],
                "backbone": model_cfg["backbone"],
                "val_loss": val_total,
            }, best_path)
            print(f"  New best model (val_loss={val_total:.4f})")

    # Save final model
    final_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch": train_cfg["num_epochs"],
        "model_state_dict": model.state_dict(),
        "num_classes": model_cfg["num_classes"],
        "backbone": model_cfg["backbone"],
    }, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train form field detector")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    train(args.config)
