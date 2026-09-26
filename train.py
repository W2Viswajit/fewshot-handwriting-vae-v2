"""
train.py — Entry point for training the Few-shot Handwriting VAE.

Usage:
    python train.py                              # Use default config
    python train.py --config config/custom.yaml  # Use custom config

What happens:
    1. Load config from YAML
    2. Load dataset (EMNIST or Omniglot) with augmentation
    3. Apply few-shot sampling (K examples per class)
    4. Split into train/validation sets
    5. Build VAE model
    6. Train with reconstruction + KL loss
    7. Save checkpoints + reconstruction samples to outputs/
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.models import FewShotVAE
from src.data import load_dataset, FewShotSubset, create_dataloader
from src.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load hyperparameters from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    # ── Parse arguments ──────────────────────────────────
    parser = argparse.ArgumentParser(description="Train Few-shot Handwriting VAE")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────
    config = load_config(args.config)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # ── Reproducibility ──────────────────────────────────
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Load dataset with augmentation ───────────────────
    dataset_name = config["dataset"]
    augment = config.get("augment", True)
    print(f"Loading {dataset_name} dataset (augment={augment})...")

    full_dataset = load_dataset(
        name=dataset_name,
        root="./data",
        split="train",
        image_size=config.get("image_size", 28),
        augment=augment,
    )

    # Apply few-shot sampling
    k_shot = config.get("k_shot", 5)
    fewshot_dataset = FewShotSubset(full_dataset, k_shot=k_shot, seed=seed)
    print(f"Few-shot subset: {len(fewshot_dataset)} samples "
          f"({k_shot}-shot per class)")

    # ── Train/validation split ───────────────────────────
    val_split = config.get("val_split", 0.1)
    val_loader = None

    if val_split > 0:
        total = len(fewshot_dataset)
        val_size = max(1, int(total * val_split))
        train_size = total - val_size

        train_subset, val_subset = torch.utils.data.random_split(
            fewshot_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        print(f"Split: {train_size} train / {val_size} validation")

        train_loader = create_dataloader(
            train_subset,
            batch_size=config.get("batch_size", 64),
            shuffle=True,
            num_workers=config.get("num_workers", 0),
        )
        val_loader = create_dataloader(
            val_subset,
            batch_size=config.get("batch_size", 64),
            shuffle=False,
            num_workers=config.get("num_workers", 0),
        )
    else:
        train_loader = create_dataloader(
            fewshot_dataset,
            batch_size=config.get("batch_size", 64),
            shuffle=True,
            num_workers=config.get("num_workers", 0),
        )

    # ── Build model ──────────────────────────────────────
    model = FewShotVAE(
        in_channels=config.get("in_channels", 1),
        latent_dim=config.get("latent_dim", 64),
        style_dim=config.get("style_dim", 32),
        num_classes=config.get("num_classes", 26),
        char_embed_dim=config.get("char_embed_dim", 16),
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"FewShotVAE model: {total_params:,} parameters")
    print()

    # ── Train ────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
    )
    history = trainer.train()

    # ── Print final summary ──────────────────────────────
    final = history[-1]
    print(f"\nFinal epoch loss — "
          f"Total: {final['total_loss']:.2f}, "
          f"Recon: {final['recon_loss']:.2f}, "
          f"KL: {final['kl_loss']:.2f}")

    if "val_total_loss" in final:
        print(f"Final validation  — "
              f"Total: {final['val_total_loss']:.2f}, "
              f"Recon: {final['val_recon_loss']:.2f}, "
              f"KL: {final['val_kl_loss']:.2f}")


if __name__ == "__main__":
    main()
