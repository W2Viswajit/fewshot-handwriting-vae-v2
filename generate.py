"""
generate.py — Few-shot handwriting generation.

Two modes:
    1. CHARACTERS: Generate individual character images
    2. PAGE: Generate a full A4 handwritten page

Usage:
    # Generate individual characters
    python generate.py --checkpoint outputs/checkpoints/vae_best.pt \\
                       --ref_dir samples/ --text "HELLO" --num_samples 1

    # Generate a full A4 page (the main use case!)
    python generate.py --checkpoint outputs/checkpoints/vae_best.pt \\
                       --ref_dir samples/ \\
                       --text "Hello World this is my handwriting" \\
                       --paper lined \\
                       --mode page

User input required:
    1. Reference images (1–5 handwriting samples)  → --references or --ref_dir
    2. Message to print                             → --text
    3. Paper background                             → --paper (plain/lined/grid/college)
"""

import argparse
import glob
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

from src.models import FewShotVAE
from src.data import get_transform
from src.utils.page_composer import compose_page, save_page


def load_reference_images(
    paths: list[str] | None = None,
    ref_dir: str | None = None,
    image_size: int = 28,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load reference handwriting images.

    Returns two versions:
        1. Preprocessed (28×28) for the model
        2. Original-size for character size estimation

    Args:
        paths:      List of image file paths.
        ref_dir:    Directory containing reference images.
        image_size: Model input size.

    Returns:
        (model_images, original_images):
            model_images: [K, 1, 28, 28] normalized
            original_images: [K, 1, H, W] at original resolution
    """
    transform = get_transform(image_size, augment=False)

    if ref_dir:
        extensions = ("*.png", "*.jpg", "*.jpeg")
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(str(Path(ref_dir) / ext)))
        paths = sorted(paths)

    if not paths:
        raise ValueError("No reference images provided. Use --references or --ref_dir.")

    print(f"Loading {len(paths)} reference image(s):")
    model_images = []
    original_images = []

    for p in paths:
        print(f"  → {p}")
        img = Image.open(p).convert("L")

        # Model-sized version (28×28)
        model_img = transform(img)  # [1, H, W]
        model_images.append(model_img)

        # Original-size version (resized to uniform 64x64 for stacking)
        from torchvision import transforms as T
        orig_transform = T.Compose([T.Grayscale(1), T.Resize((64, 64)), T.ToTensor()])
        orig_img = orig_transform(img)
        original_images.append(orig_img)

    return torch.stack(model_images), torch.stack(original_images)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> FewShotVAE:
    """Load a trained FewShotVAE from checkpoint."""
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

    print(f"Model loaded from: {checkpoint_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Best loss: {checkpoint.get('best_loss', 'N/A')}")

    return model


def generate_characters_mode(
    model: FewShotVAE,
    text: str,
    ref_images: torch.Tensor,
    device: torch.device,
    output_dir: str,
) -> None:
    """Generate individual character images and save them."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating characters for: \"{text}\"")
    char_images = model.generate_text(text, ref_images, device)

    # Save each character
    char_count = 0
    for i, (ch, img) in enumerate(zip(text, char_images)):
        if img is not None:
            img_path = output_path / f"char_{i:03d}_{ch}.png"
            save_image(img.cpu(), str(img_path), normalize=True, value_range=(0, 1))
            char_count += 1

    # Save as grid
    valid_images = [img for img in char_images if img is not None]
    if valid_images:
        grid_tensor = torch.cat(valid_images, dim=0)
        grid_path = output_path / "characters_grid.png"
        save_image(
            grid_tensor.cpu(), str(grid_path),
            nrow=min(len(valid_images), 10),
            normalize=True, value_range=(0, 1),
        )
        print(f"Grid saved: {grid_path}")

    print(f"Generated {char_count} characters → {output_dir}/")


def generate_page_mode(
    model: FewShotVAE,
    text: str,
    ref_model_images: torch.Tensor,
    ref_original_images: torch.Tensor,
    device: torch.device,
    paper_type: str,
    output_dir: str,
) -> None:
    """Generate a full A4 handwritten page."""
    print(f"\nGenerating page:")
    print(f"  Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
    print(f"  Paper: {paper_type}")

    # Generate all characters in the message
    print("  Generating characters...")
    char_images = model.generate_text(text, ref_model_images, device)

    valid_count = sum(1 for img in char_images if img is not None)
    print(f"  Generated {valid_count} character images")

    # Compose into a full page
    print("  Composing page...")
    page = compose_page(
        text=text,
        char_images=char_images,
        ref_images=ref_original_images,
        paper_type=paper_type,
    )

    # Save as PNG and PDF
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_page(page, str(output_path / "handwritten_page.png"))
    save_page(page, str(output_path / "handwritten_page.pdf"))

    print(f"\nDone! Output saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate handwriting — individual characters or full pages"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--references", type=str, nargs="+", default=None,
        help="Paths to 1–5 reference handwriting images",
    )
    parser.add_argument(
        "--ref_dir", type=str, default=None,
        help="Directory containing reference images",
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="Message to generate in handwriting",
    )
    parser.add_argument(
        "--mode", type=str, default="page", choices=["chars", "page"],
        help="'chars' = individual letters, 'page' = full A4 page (default: page)",
    )
    parser.add_argument(
        "--paper", type=str, default="lined",
        choices=["plain", "lined", "grid", "college"],
        help="Paper background type (default: lined)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/generated",
        help="Output directory",
    )
    parser.add_argument(
        "--image_size", type=int, default=28,
        help="Model input image size (must match training)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model
    model = load_checkpoint(args.checkpoint, device)

    # Load references (both model-sized and original-sized)
    ref_model, ref_original = load_reference_images(
        paths=args.references,
        ref_dir=args.ref_dir,
        image_size=args.image_size,
    )
    print(f"References: {ref_model.shape[0]} images")

    # Generate
    if args.mode == "chars":
        generate_characters_mode(
            model, args.text, ref_model, device, args.output_dir,
        )
    else:
        generate_page_mode(
            model, args.text, ref_model, ref_original,
            device, args.paper, args.output_dir,
        )


if __name__ == "__main__":
    main()
