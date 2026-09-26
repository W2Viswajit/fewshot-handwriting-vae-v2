# ============================================================
# Few-Shot Handwriting VAE — Google Colab Training (FIXED v2)
# ============================================================
#
# FIXES from v1:
# - latent_dim: 64 → 8 (prevents z from encoding character identity)
# - char_embed_dim: 16 → 64 (makes character embedding dominant)
# - Added CharClassifier auxiliary head + classification loss
# - beta: 1.0 → 0.5 (better reconstruction quality)
# - gamma: 10.0 (classification loss weight)
#
# USAGE:
# 1. Open Colab → Runtime → Change runtime type → GPU (T4)
# 2. Upload this file and run: !python colab_notebook.py
# 3. When prompted, upload your 5 reference handwriting images
# ============================================================

import os
import sys

# ── Step 1: Create project structure ──────────────────────

dirs = [
    "src/models", "src/data", "src/training", "src/utils",
    "config", "samples/refs",
    "outputs/checkpoints", "outputs/reconstructions", "outputs/generated",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

print("✓ Directories created")


# ── Step 2: Write all source files ────────────────────────

with open("src/__init__.py", "w") as f:
    f.write("")

with open("src/models/__init__.py", "w") as f:
    f.write("""from .vae import VAE
from .style_encoder import StyleEncoder
from .fewshot_vae import FewShotVAE
__all__ = ['VAE', 'StyleEncoder', 'FewShotVAE']
""")

with open("src/data/__init__.py", "w") as f:
    f.write("""from .dataset import load_dataset, FewShotSubset, create_dataloader, get_transform
__all__ = ['load_dataset', 'FewShotSubset', 'create_dataloader', 'get_transform']
""")

with open("src/training/__init__.py", "w") as f:
    f.write("""from .losses import vae_loss
from .trainer import Trainer
__all__ = ['vae_loss', 'Trainer']
""")

with open("src/utils/__init__.py", "w") as f:
    f.write("")


# ── src/models/vae.py ────────────────────────────────────

with open("src/models/vae.py", "w") as f:
    f.write('''
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=8):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self._flat_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

    def forward(self, x):
        h = torch.flatten(self.conv_layers(x), 1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=8):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv_layers(self.fc(z).view(-1, 64, 7, 7))

class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def reparameterize(self, mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)

    def forward(self, x):
        mu, lv = self.encoder(x)
        return self.decoder(self.reparameterize(mu, lv)), mu, lv
''')


# ── src/models/style_encoder.py ──────────────────────────

with open("src/models/style_encoder.py", "w") as f:
    f.write('''
import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleEncoder(nn.Module):
    def __init__(self, style_dim=32, in_channels=1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Linear(128 * 4 * 4, style_dim)

    def forward(self, x):
        h = self.conv_layers(x).view(x.size(0), -1)
        return F.normalize(self.fc(h), p=2, dim=1)
''')


# ── src/models/fewshot_vae.py (FIXED — information bottleneck) ──

with open("src/models/fewshot_vae.py", "w") as f:
    f.write('''
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.vae import Encoder
from src.models.style_encoder import StyleEncoder

CHAR_TO_INDEX = {chr(ord("A") + i): i for i in range(26)}
for i in range(26):
    CHAR_TO_INDEX[chr(ord("a") + i)] = i

def text_to_indices(text):
    result = []
    for ch in text:
        if ch in CHAR_TO_INDEX:
            result.append(CHAR_TO_INDEX[ch])
        elif ch == " ":
            result.append(None)
    return result


class CharClassifier(nn.Module):
    """Auxiliary classifier on decoder output — forces correct letter generation."""
    def __init__(self, in_channels=1, num_classes=26):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
        )
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class ConditionedDecoder(nn.Module):
    """Decoder: concat(z[8], style[32], char_embed[64]) -> image."""
    def __init__(self, out_channels=1, latent_dim=8, style_dim=32, char_embed_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim + style_dim + char_embed_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, style, char_embed):
        h = self.fc(torch.cat([z, style, char_embed], dim=1))
        return self.deconv(h.view(-1, 64, 7, 7))


class FewShotVAE(nn.Module):
    """
    FIXED architecture:
      z (dim=8)          — just noise/variation (TOO SMALL for char identity)
      style (dim=32)     — handwriting style
      char_embed (dim=64)— WHICH letter (DOMINANT signal)
      + CharClassifier   — forces decoder to produce correct letter
    """
    def __init__(self, in_channels=1, latent_dim=8, style_dim=32,
                 num_classes=26, char_embed_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = Encoder(in_channels, latent_dim)
        self.style_encoder = StyleEncoder(style_dim, in_channels)
        self.char_embedding = nn.Embedding(num_classes, char_embed_dim)
        self.decoder = ConditionedDecoder(in_channels, latent_dim, style_dim, char_embed_dim)
        self.classifier = CharClassifier(in_channels, num_classes)

    def reparameterize(self, mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)

    def forward(self, x, labels=None):
        B = x.size(0)
        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=x.device)
        style = self.style_encoder(x)
        mu, lv = self.encoder(x)
        z = self.reparameterize(mu, lv)
        x_recon = self.decoder(z, style, self.char_embedding(labels))
        char_logits = self.classifier(x_recon)
        return x_recon, mu, lv, char_logits

    @torch.no_grad()
    def generate_text(self, text, style_ref, device):
        self.eval()
        style_ref = style_ref.to(device)
        avg_style = self.style_encoder(style_ref).mean(0, keepdim=True)
        result = []
        for idx in text_to_indices(text):
            if idx is None:
                result.append(None)
            else:
                ct = torch.tensor([idx], dtype=torch.long, device=device)
                z = torch.randn(1, self.latent_dim, device=device)
                result.append(self.decoder(z, avg_style, self.char_embedding(ct)))
        return result

    @torch.no_grad()
    def reconstruct(self, x, labels=None):
        out, _, _, _ = self.forward(x, labels)
        return out
''')


# ── src/data/dataset.py ──────────────────────────────────

with open("src/data/dataset.py", "w") as f:
    f.write('''
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict

def get_transform(image_size=28, augment=False):
    t = [transforms.Resize((image_size, image_size)), transforms.Grayscale(1)]
    if augment:
        t.extend([transforms.RandomRotation(10, fill=0),
                  transforms.RandomAffine(0, translate=(0.07, 0.07), fill=0)])
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

def load_dataset(name="emnist", root="./data", split="train",
                 image_size=28, augment=False):
    transform = get_transform(image_size, augment)
    if name == "emnist":
        return datasets.EMNIST(root=root, split="letters", train=(split=="train"),
                               download=True, transform=transform,
                               target_transform=lambda y: y - 1)
    raise ValueError(f"Unknown dataset: {name}")

class FewShotSubset(Subset):
    def __init__(self, dataset, k_shot=5, seed=42):
        indices = self._select(dataset, k_shot, seed)
        super().__init__(dataset, indices)
        self.k_shot = k_shot

    @staticmethod
    def _select(dataset, k_shot, seed):
        groups = defaultdict(list)
        for i in range(len(dataset)):
            _, label = dataset[i]
            if isinstance(label, torch.Tensor): label = label.item()
            groups[label].append(i)
        gen = torch.Generator().manual_seed(seed)
        selected = []
        for _, idxs in sorted(groups.items()):
            k = min(k_shot, len(idxs))
            perm = torch.randperm(len(idxs), generator=gen)[:k]
            selected.extend(idxs[i] for i in perm.tolist())
        return selected

def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)
''')


# ── src/training/losses.py (FIXED — classification loss) ──

with open("src/training/losses.py", "w") as f:
    f.write('''
import torch
import torch.nn.functional as F

def reconstruction_loss(x_recon, x):
    return F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)

def kl_divergence(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)

def classification_loss(char_logits, labels):
    """Forces decoder to produce the CORRECT letter, not just any letter."""
    return F.cross_entropy(char_logits, labels)

def vae_loss(x_recon, x, mu, log_var, beta=0.5,
             char_logits=None, labels=None, gamma=10.0):
    """Total = recon + beta*KL + gamma*classification"""
    recon = reconstruction_loss(x_recon, x)
    kl = kl_divergence(mu, log_var)
    if char_logits is not None and labels is not None:
        cls = classification_loss(char_logits, labels)
    else:
        cls = torch.tensor(0.0, device=x.device)
    total = recon + beta * kl + gamma * cls
    return total, recon, kl, cls
''')


# ── src/training/trainer.py (FIXED — 4 outputs + cls loss) ──

with open("src/training/trainer.py", "w") as f:
    f.write('''
import time
from pathlib import Path
import torch
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm
from src.training.losses import vae_loss

class Trainer:
    def __init__(self, model, train_loader, config, device=None):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
        self.epochs = config.get("epochs", 50)
        self.beta = config.get("beta", 0.5)
        self.gamma = config.get("gamma", 10.0)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.save_every = config.get("save_every", 10)
        self.save_recon_every = config.get("save_recon_every", 10)
        self.out = Path(config.get("output_dir", "./outputs"))
        self.ckpt_dir = self.out / "checkpoints"
        self.recon_dir = self.out / "reconstructions"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.recon_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.history = []

    def train(self):
        print(f"Training on {self.device} | {self.epochs} epochs | beta={self.beta} | gamma={self.gamma}")
        print("-" * 60)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            m = self._train_epoch(epoch)
            self.history.append(m)
            print(f"Epoch {epoch:3d} | Loss: {m['total']:8.2f} | "
                  f"Recon: {m['recon']:7.2f} | KL: {m['kl']:6.2f} | "
                  f"Cls: {m['cls']:5.3f} | {m['time']:.1f}s")
            if m["total"] < self.best_loss:
                self.best_loss = m["total"]
                self._save(epoch, "best")
            if epoch % self.save_every == 0: self._save(epoch)
            if epoch % self.save_recon_every == 0: self._save_recon(epoch)
        self._save(self.epochs, "final")
        print(f"Done! Best loss: {self.best_loss:.2f}")
        return self.history

    def _train_epoch(self, epoch):
        ts = rs = ks = cs = 0.0; n = 0; t0 = time.time()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d}/{self.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device) if isinstance(labels, torch.Tensor) else labels
            # 4 outputs: recon, mu, logvar, char_logits
            x_recon, mu, lv, char_logits = self.model(images, labels)
            total, recon, kl, cls = vae_loss(
                x_recon, images, mu, lv, self.beta,
                char_logits=char_logits, labels=labels, gamma=self.gamma)
            self.optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            ts += total.item(); rs += recon.item(); ks += kl.item(); cs += cls.item(); n += 1
            pbar.set_postfix(loss=f"{total.item():.1f}", cls=f"{cls.item():.3f}")
        return {"epoch": epoch, "total": ts/n, "recon": rs/n,
                "kl": ks/n, "cls": cs/n, "time": time.time()-t0}

    @torch.no_grad()
    def _save_recon(self, epoch, n=8):
        self.model.eval()
        imgs, labels = next(iter(self.train_loader))
        imgs = imgs[:n].to(self.device)
        labels = labels[:n].to(self.device) if isinstance(labels, torch.Tensor) else None
        recon = self.model.reconstruct(imgs, labels)
        save_image(torch.cat([imgs, recon], 0).cpu(), self.recon_dir / f"recon_{epoch:03d}.png",
                   nrow=n, normalize=True, value_range=(0,1))

    def _save(self, epoch, tag=None):
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                     "optimizer_state_dict": self.optimizer.state_dict(),
                     "config": self.config, "history": self.history,
                     "best_loss": self.best_loss},
                    self.ckpt_dir / f"vae_{tag or f'epoch_{epoch:03d}'}.pt")
''')


# ── src/utils/paper_backgrounds.py ───────────────────────

with open("src/utils/paper_backgrounds.py", "w") as f:
    f.write('''
from PIL import Image, ImageDraw

A4_WIDTH, A4_HEIGHT = 2480, 3508

def create_background(paper_type="plain", width=A4_WIDTH, height=A4_HEIGHT,
                       line_spacing=80, margin_left=150, margin_top=150):
    img = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(img)
    if paper_type == "lined":
        y = margin_top
        while y < height - 50:
            draw.line([(0,y),(width,y)], fill=(180,210,240), width=1); y += line_spacing
    elif paper_type == "grid":
        y = margin_top
        while y < height - 50:
            draw.line([(0,y),(width,y)], fill=(210,210,210), width=1); y += line_spacing
        x = margin_left
        while x < width - 50:
            draw.line([(x,0),(x,height)], fill=(210,210,210), width=1); x += line_spacing
    elif paper_type == "college":
        y = margin_top
        while y < height - 50:
            draw.line([(0,y),(width,y)], fill=(180,210,240), width=1); y += line_spacing
        draw.line([(margin_left,0),(margin_left,height)], fill=(220,120,120), width=2)
    return img
''')


# ── src/utils/page_composer.py ───────────────────────────

with open("src/utils/page_composer.py", "w") as f:
    f.write('''
import random
import numpy as np
import torch
from PIL import Image, ImageFilter
from src.utils.paper_backgrounds import create_background, A4_WIDTH, A4_HEIGHT

def estimate_char_size(ref_images):
    sizes = []
    for i in range(ref_images.size(0)):
        img = ref_images[i, 0].numpy()
        mask = img > 0.1
        if mask.any():
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            sizes.append(max(rows[-1]-rows[0]+1, cols[-1]-cols[0]+1))
    if not sizes: return 60
    return max(30, min(120, int(60 * np.mean(sizes) / 28.0)))

def compose_page(text, char_images, ref_images, paper_type="plain"):
    cell = estimate_char_size(ref_images)
    line_h = int(cell * 1.4)
    word_sp = int(cell * 0.6)
    margin_l, margin_r, margin_t = 180, 150, 180
    page = create_background(paper_type, A4_WIDTH, A4_HEIGHT, line_h)
    cx, cy, ci = margin_l, margin_t, 0
    for ch in text:
        if ch == "\\n": cx, cy = margin_l, cy + line_h; continue
        if ch == " ":
            cx += word_sp; ci += 1
            if cx > A4_WIDTH - margin_r: cx, cy = margin_l, cy + line_h
            continue
        if cy + cell > A4_HEIGHT - 100: break
        if cx + cell > A4_WIDTH - margin_r: cx, cy = margin_l, cy + line_h
        if ci < len(char_images) and char_images[ci] is not None:
            char_pil = _to_pil(char_images[ci], cell)
            char_pil = char_pil.rotate(random.uniform(-2,2), expand=False, fillcolor=(0,0,0,0))
            page.paste(char_pil, (cx, cy + random.randint(-3,3)), char_pil)
        cx += cell + 5 + random.randint(-2, 2); ci += 1
    return page

def _to_pil(tensor, size):
    arr = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, "L").resize((size, size), Image.Resampling.LANCZOS)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:,:,0], rgba[:,:,1], rgba[:,:,2] = 20, 20, 50
    rgba[:,:,3] = np.array(img)
    return Image.fromarray(rgba, "RGBA")

def save_page(page, path):
    from pathlib import Path as P
    P(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".pdf"): page.convert("RGB").save(path, "PDF", resolution=300)
    else: page.save(path)
    print(f"Saved: {path}")
''')

print("✓ All source files written!")


# ── Step 3: Upload reference images ──────────────────────

print("\n" + "=" * 50)
print("📤 Upload your 5 handwriting reference images")
print("   (your a, d, k, h, m letter images)")
print("=" * 50)

try:
    from google.colab import files
    uploaded = files.upload()
    for filename, data in uploaded.items():
        with open(f"samples/refs/{filename}", "wb") as f:
            f.write(data)
        print(f"  ✓ {filename}")
    print(f"\n✓ {len(uploaded)} reference images saved!")
except ImportError:
    from PIL import Image, ImageDraw, ImageFont
    print("Not in Colab — creating test reference images...")
    for letter in ["A", "D", "K", "H", "M"]:
        img = Image.new("L", (64, 64), 255)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), letter, font=font)
        draw.text(((64-(bbox[2]-bbox[0]))//2, (64-(bbox[3]-bbox[1]))//2), letter, fill=30, font=font)
        img.save(f"samples/refs/{letter.lower()}.png")
    print("✓ Test reference images created")


# ── Step 4: Train 200 epochs ─────────────────────────────

import torch
print(f"\n🔧 PyTorch {torch.__version__}")
print(f"🖥️  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

sys.path.insert(0, ".")

from src.models.fewshot_vae import FewShotVAE
from src.data.dataset import load_dataset, FewShotSubset, create_dataloader

config = {
    "dataset": "emnist", "image_size": 28, "k_shot": 10,
    "latent_dim": 8,              # SMALL — just noise
    "in_channels": 1,
    "style_dim": 32,
    "num_classes": 26,
    "char_embed_dim": 64,         # LARGE — dominant character signal
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 0.001,
    "beta": 0.5,                  # Lower KL weight
    "gamma": 10.0,                # Classification loss weight (KEY FIX)
    "max_grad_norm": 1.0,
    "augment": True,
    "save_every": 50,
    "save_recon_every": 25,
    "output_dir": "./outputs",
    "num_workers": 2 if torch.cuda.is_available() else 0,
    "seed": 42,
}

torch.manual_seed(42)

print("\n📂 Loading EMNIST...")
full_dataset = load_dataset("emnist", "./data", "train", 28, augment=True)
fewshot = FewShotSubset(full_dataset, k_shot=config["k_shot"], seed=42)
print(f"   {len(fewshot)} samples (10-shot × 26 classes = 260)")

loader = create_dataloader(fewshot, batch_size=64, shuffle=True,
                           num_workers=config["num_workers"])

model = FewShotVAE(in_channels=1, latent_dim=8, style_dim=32,
                   num_classes=26, char_embed_dim=64)
print(f"   Model: {sum(p.numel() for p in model.parameters()):,} params\n")

from src.training.trainer import Trainer
trainer = Trainer(model=model, train_loader=loader, config=config)
history = trainer.train()


# ── Step 5: Plot training curves ─────────────────────────

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
epochs_list = [h["epoch"] for h in history]
axes[0].plot(epochs_list, [h["total"] for h in history], "b-"); axes[0].set_title("Total Loss")
axes[1].plot(epochs_list, [h["recon"] for h in history], "g-"); axes[1].set_title("Recon Loss")
axes[2].plot(epochs_list, [h["kl"] for h in history], "r-"); axes[2].set_title("KL Divergence")
axes[3].plot(epochs_list, [h["cls"] for h in history], "m-"); axes[3].set_title("Classification Loss")
fig.suptitle("Training (200 Epochs) — cls should drop toward 0", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/training_curves.png", dpi=150); plt.show()


# ── Step 6: Generate "HELLO WORLD" ───────────────────────

import glob
from PIL import Image
from torchvision import transforms as T
from src.data.dataset import get_transform
from src.utils.page_composer import compose_page, save_page
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

transform_28 = get_transform(28, augment=False)
orig_transform = T.Compose([T.Grayscale(1), T.Resize((64, 64)), T.ToTensor()])

ref_paths = sorted(glob.glob("samples/refs/*.*"))
print(f"\nLoading {len(ref_paths)} reference images...")

model_imgs, orig_imgs = [], []
for p in ref_paths[:5]:
    img = Image.open(p).convert("L")
    print(f"  → {os.path.basename(p)} ({img.size[0]}x{img.size[1]})")
    model_imgs.append(transform_28(img))
    orig_imgs.append(orig_transform(img))

ref_model = torch.stack(model_imgs)
ref_orig = torch.stack(orig_imgs)

# Generate HELLO WORLD
text = "HELLO WORLD"
print(f'\n🖊️ Generating: "{text}"')
char_images = model.generate_text(text, ref_model, device)

# Show characters
valid = [img for img in char_images if img is not None]
if valid:
    save_image(torch.cat(valid, 0).cpu(), "outputs/generated/chars_grid.png",
               nrow=len(valid), normalize=True, value_range=(0, 1))

# Generate all 4 paper types
for paper in ["plain", "lined", "grid", "college"]:
    page = compose_page(text=text, char_images=char_images,
                        ref_images=ref_orig, paper_type=paper)
    save_page(page, f"outputs/generated/hello_world_{paper}.png")

# Show results
try:
    from IPython.display import Image as IPImage, display
    print("\n📄 Characters:")
    display(IPImage(filename="outputs/generated/chars_grid.png", width=500))
    for paper in ["plain", "lined", "grid", "college"]:
        print(f"\n📄 {paper.upper()} paper:")
        display(IPImage(filename=f"outputs/generated/hello_world_{paper}.png", width=400))
except:
    print("Outputs saved in outputs/generated/")


# ── Step 7: Generate full pangram ─────────────────────────

long_text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
print(f'\n🖊️ Generating pangram: "{long_text}"')
char_long = model.generate_text(long_text, ref_model, device)
page_long = compose_page(text=long_text, char_images=char_long,
                         ref_images=ref_orig, paper_type="lined")
save_page(page_long, "outputs/generated/pangram_lined.png")

try:
    from IPython.display import Image as IPImage, display
    print("\n📄 Pangram on lined paper:")
    display(IPImage(filename="outputs/generated/pangram_lined.png", width=500))
except:
    pass


# ── Step 8: Download everything ──────────────────────────

try:
    from google.colab import files
    import shutil
    shutil.make_archive("handwriting_outputs", "zip", "outputs")
    files.download("handwriting_outputs.zip")
    print("\n📥 All outputs downloaded as ZIP!")
except:
    print(f"\n✓ All outputs saved in: outputs/generated/")

print("\n" + "=" * 50)
print("🎉 DONE! Check outputs/generated/ for results")
print("=" * 50)
