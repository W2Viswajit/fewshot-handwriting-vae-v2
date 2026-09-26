"""
Extended training (50 epochs) + page generation test.
"""
import sys
sys.path.insert(0, ".")

import torch
from pathlib import Path
from PIL import Image
import glob

from src.models import FewShotVAE
from src.data import load_dataset, FewShotSubset, create_dataloader
from src.training.trainer import Trainer
from src.data import get_transform
from src.utils.page_composer import compose_page, save_page


def main():
    config = {
        "dataset": "emnist",
        "image_size": 28,
        "k_shot": 5,
        "latent_dim": 64,
        "in_channels": 1,
        "style_dim": 32,
        "num_classes": 26,
        "char_embed_dim": 16,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
        "beta": 1.0,
        "max_grad_norm": 1.0,
        "augment": True,
        "val_split": 0.0,
        "save_every": 10,
        "save_recon_every": 10,
        "output_dir": "./outputs",
        "num_workers": 0,
        "seed": 42,
    }

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Train ──
    print("Loading EMNIST...")
    full_dataset = load_dataset(
        name="emnist", root="./data", split="train",
        image_size=28, augment=True,
    )

    fewshot = FewShotSubset(full_dataset, k_shot=5, seed=42)
    print(f"Few-shot: {len(fewshot)} samples (5-shot × 26 classes = 130)")

    loader = create_dataloader(fewshot, batch_size=64, shuffle=True)

    model = FewShotVAE(
        in_channels=1, latent_dim=64, style_dim=32,
        num_classes=26, char_embed_dim=16,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters\n")

    trainer = Trainer(model=model, train_loader=loader, config=config)
    trainer.train()

    # ── Generate page ──
    print("\n" + "=" * 60)
    print("Generating 'HELLO WORLD' on plain paper...")
    print("=" * 60)

    model.eval()

    # Load refs
    transform = get_transform(28, augment=False)
    from torchvision import transforms as T
    orig_transform = T.Compose([T.Grayscale(1), T.ToTensor()])

    paths = sorted(glob.glob("samples/refs/*.png"))
    model_imgs = []
    orig_imgs = []
    for p in paths[:5]:
        img = Image.open(p).convert("L")
        model_imgs.append(transform(img))
        orig_imgs.append(orig_transform(img))

    ref_model = torch.stack(model_imgs)
    ref_orig = torch.stack(orig_imgs)

    # Generate characters
    text = "HELLO WORLD"
    char_images = model.generate_text(text, ref_model, device)

    # Compose page
    page = compose_page(
        text=text,
        char_images=char_images,
        ref_images=ref_orig,
        paper_type="plain",
    )

    out = Path("outputs/generated")
    out.mkdir(parents=True, exist_ok=True)
    save_page(page, str(out / "hello_world_50ep.png"))

    # Also save char grid
    from torchvision.utils import save_image
    valid = [img for img in char_images if img is not None]
    if valid:
        grid = torch.cat(valid, dim=0)
        save_image(grid.cpu(), str(out / "chars_50ep.png"),
                   nrow=len(valid), normalize=True, value_range=(0, 1))

    print(f"\nDone! → outputs/generated/hello_world_50ep.png")


if __name__ == "__main__":
    main()
