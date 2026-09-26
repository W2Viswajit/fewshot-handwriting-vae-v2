"""
Quick end-to-end test: Train → Generate → Page output.

This script:
1. Trains the FewShotVAE for a few epochs on EMNIST
2. Generates "HELLO WORLD" using reference images
3. Outputs a plain A4 page
"""

import sys
import os
sys.path.insert(0, ".")

import torch
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Step 0: Create synthetic reference images if none exist ──
# (Using the user's uploaded images saved in samples/refs/)
# If no images exist, create simple ones for pipeline testing

def create_test_refs(output_dir: str = "samples/refs"):
    """Create simple handwriting-like reference images for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    existing = list(Path(output_dir).glob("*.png")) + list(Path(output_dir).glob("*.jpg"))
    if existing:
        print(f"Found {len(existing)} existing reference images in {output_dir}")
        return
    
    print("Creating test reference images...")
    # Create simple grayscale letters with PIL
    letters = ["A", "D", "K", "H", "M"]
    for letter in letters:
        img = Image.new("L", (64, 64), 255)  # White background
        draw = ImageDraw.Draw(img)
        # Draw letter in dark
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except OSError:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), letter, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (64 - w) // 2
        y = (64 - h) // 2
        draw.text((x, y), letter, fill=30, font=font)
        
        img.save(os.path.join(output_dir, f"{letter.lower()}.png"))
    print(f"Created {len(letters)} test reference images")


# ── Step 1: Train quickly ──

def quick_train():
    """Train for a few epochs to get a working checkpoint."""
    from src.models import FewShotVAE
    from src.data import load_dataset, FewShotSubset, create_dataloader
    from src.training.trainer import Trainer
    
    config = {
        "dataset": "emnist",
        "image_size": 28,
        "k_shot": 5,
        "latent_dim": 64,
        "in_channels": 1,
        "style_dim": 32,
        "num_classes": 26,
        "char_embed_dim": 16,
        "epochs": 5,          # Quick test — just 5 epochs
        "batch_size": 64,
        "learning_rate": 0.001,
        "beta": 1.0,
        "max_grad_norm": 1.0,
        "augment": True,
        "val_split": 0.0,     # No validation for quick test
        "save_every": 5,
        "save_recon_every": 5,
        "output_dir": "./outputs",
        "num_workers": 0,
        "seed": 42,
    }
    
    torch.manual_seed(42)
    
    print("=" * 60)
    print("STEP 1: Training (5 epochs)")
    print("=" * 60)
    
    # Load EMNIST
    print("Loading EMNIST dataset...")
    full_dataset = load_dataset(
        name="emnist", root="./data", split="train",
        image_size=28, augment=True,
    )
    
    # Few-shot subset
    fewshot = FewShotSubset(full_dataset, k_shot=5, seed=42)
    print(f"Few-shot: {len(fewshot)} samples")
    
    loader = create_dataloader(fewshot, batch_size=64, shuffle=True)
    
    # Build model
    model = FewShotVAE(
        in_channels=1, latent_dim=64, style_dim=32,
        num_classes=26, char_embed_dim=16,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    
    # Train
    trainer = Trainer(model=model, train_loader=loader, config=config)
    trainer.train()
    
    print("Training complete!\n")
    return "outputs/checkpoints/vae_final.pt"


# ── Step 2: Generate page ──

def generate_page(checkpoint_path: str, ref_dir: str = "samples/refs"):
    """Generate 'HELLO WORLD' on a plain sheet."""
    from src.models import FewShotVAE
    from src.data import get_transform
    from src.utils.page_composer import compose_page, save_page
    import glob
    
    print("=" * 60)
    print("STEP 2: Generating 'HELLO WORLD' on plain paper")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    model = FewShotVAE(
        in_channels=config.get("in_channels", 1),
        latent_dim=config.get("latent_dim", 64),
        style_dim=config.get("style_dim", 32),
        num_classes=config.get("num_classes", 26),
        char_embed_dim=config.get("char_embed_dim", 16),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded (epoch {checkpoint['epoch']})")
    
    # Load reference images
    transform = get_transform(28, augment=False)
    from torchvision import transforms as T
    orig_transform = T.Compose([T.Grayscale(1), T.ToTensor()])
    
    paths = sorted(glob.glob(str(Path(ref_dir) / "*.png")) + 
                   glob.glob(str(Path(ref_dir) / "*.jpg")))
    
    model_imgs = []
    orig_imgs = []
    for p in paths[:5]:
        img = Image.open(p).convert("L")
        model_imgs.append(transform(img))
        orig_imgs.append(orig_transform(img))
    
    ref_model = torch.stack(model_imgs)
    ref_orig = torch.stack(orig_imgs)
    print(f"Loaded {len(paths)} reference images")
    
    # Generate text
    text = "HELLO WORLD"
    print(f"Generating: '{text}'")
    
    char_images = model.generate_text(text, ref_model, device)
    valid = sum(1 for img in char_images if img is not None)
    print(f"Generated {valid} character images")
    
    # Compose page
    page = compose_page(
        text=text,
        char_images=char_images,
        ref_images=ref_orig,
        paper_type="plain",
    )
    
    # Save
    output_dir = Path("outputs/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_page(page, str(output_dir / "hello_world_plain.png"))
    
    # Also save the individual character grid
    from torchvision.utils import save_image
    valid_images = [img for img in char_images if img is not None]
    if valid_images:
        grid = torch.cat(valid_images, dim=0)
        save_image(grid.cpu(), str(output_dir / "chars_grid.png"),
                   nrow=len(valid_images), normalize=True, value_range=(0, 1))
        print(f"Character grid saved: {output_dir / 'chars_grid.png'}")
    
    print(f"\n✓ Page saved: {output_dir / 'hello_world_plain.png'}")
    return str(output_dir / "hello_world_plain.png")


# ── Main ──

if __name__ == "__main__":
    create_test_refs()
    ckpt = quick_train()
    output = generate_page(ckpt)
    print(f"\n{'=' * 60}")
    print(f"DONE! Output: {output}")
    print(f"{'=' * 60}")
