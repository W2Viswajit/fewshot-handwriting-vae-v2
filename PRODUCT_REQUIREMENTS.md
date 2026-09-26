# Product Requirements Document: Few-Shot Handwriting Generator

## 1. Purpose

This document is the project handoff for a new engineering/modeling agent. It describes the intended product, what is implemented, what has actually been demonstrated, known gaps, and the recommended order of work. Treat repository code and checkpoint metadata as authoritative when they disagree with old comments or experiment scripts.

## 2. Product Summary

The project is a local Python/PyTorch prototype that aims to generate a requested English message in a writer's handwriting from a small set of reference images. The user provides reference handwriting images and text; the system generates one image per supported character, lays the images onto an A4 paper background, and writes PNG and PDF outputs.

The intended differentiator is disentangled conditioning:

- Character identity comes from a learned A-Z embedding.
- Writer appearance comes from a style encoder averaged over reference images.
- A stochastic latent vector supplies per-character variation.
- A page composer provides spacing, line wrapping, paper styles, and small visual perturbations.

This is currently a research/demo prototype, not a reliable handwriting product. The existing generated page and character-grid artifacts are not legible enough to meet the product goal.

## 3. Users and Primary Workflow

### Target user

A user who wants a short text rendered to resemble a particular writer, using a handful of that writer's handwritten character images.

### Intended workflow

1. Install dependencies and train a compatible model on the EMNIST Letters dataset.
2. Place writer reference images in a folder or pass explicit image paths.
3. Supply text and choose character-image or full-page output.
4. Choose one of plain, lined, grid, or college-ruled paper.
5. Receive generated character PNGs and, for page mode, an A4 PNG and PDF.

There is no GUI or hosted service in the repository. The product surface is command-line driven.

## 4. Product Goals

### MVP goals

- Generate recognizable uppercase and lowercase English letters from A-Z.
- Preserve a consistent, visibly reference-conditioned handwriting style across a complete message.
- Support a small set of reference images without requiring labeled reference text.
- Lay out text with readable spacing, explicit newlines, and predictable line wrapping.
- Export a printable A4 PNG and PDF, plus optional per-character previews.
- Make training and generation reproducible from a saved checkpoint and config.

### Non-goals for the current MVP

- Natural cursive word synthesis or connected strokes; current generation is character-by-character.
- Reliable digits, punctuation, symbols, multilingual text, or arbitrary scripts.
- A web/mobile interface, user accounts, cloud inference, or a production deployment.
- Claims of biometric writer identification or guaranteed identity preservation.

## 5. Functional Requirements

### FR-1: Text support

- MVP text set is English A-Z, case-insensitive, and spaces.
- Unsupported characters must not be silently dropped or misalign generated characters with page positions. The next implementation must either support them or return a clear validation error.
- Newlines should create a new page line without consuming a generated character.

### FR-2: Reference image handling

- Accept 1-5 grayscale or color image files and normalize them consistently for the model.
- Estimate page character size from reference content, not merely the full image dimensions.
- Validate that references contain usable foreground strokes and report actionable errors for blank/invalid files.
- Establish and document a single foreground/background polarity convention across EMNIST, references, model outputs, and page composition.

### FR-3: Character generation

- Generate the requested character sequence in order, preserving spaces and line breaks as layout tokens.
- Allow stochastic variations while retaining recognizable character identity and consistent writer style.
- Support reproducible generation with a seed.

### FR-4: Page composition and export

- Offer plain, lined, grid, and college-ruled paper.
- Preserve margins and line spacing, wrap before crossing the right margin, and stop or paginate cleanly at the bottom.
- Export PNG and PDF with dimensions and resolution documented by the CLI.
- Keep the generated character visually distinct from the paper background with correct ink transparency/polarity.

### FR-5: Training and artifacts

- Train from YAML configuration with a deterministic few-shot subset when a seed is supplied.
- Record train and validation reconstruction, KL, and classification metrics where applicable.
- Save compatible best, periodic, and final checkpoints containing model config and training metadata.
- Generation must instantiate the model from checkpoint metadata rather than unrelated hard-coded defaults.

## 6. Current Implementation

### Data pipeline

- `src/data/dataset.py` loads the EMNIST Letters split by default and also declares an Omniglot loader.
- Images are resized to 28x28 grayscale tensors in [0, 1]. Optional rotation and translation augmentation is available.
- EMNIST labels are remapped from 1-26 to 0-25.
- `FewShotSubset` samples up to K examples per class. The default config uses 10 examples per each of 26 classes.
- The train entry point makes a random train/validation split from the sampled subset.

### Model

- `src/models/vae.py` contains the shared convolutional encoder used by the conditioned model and a standalone VAE.
- `src/models/style_encoder.py` encodes grayscale references into normalized style vectors.
- `src/models/fewshot_vae.py` combines an 8-dimensional stochastic latent, 32-dimensional style vector, 64-dimensional character embedding, conditioned decoder, and auxiliary character classifier.
- Generation currently averages style vectors over all supplied reference images, then samples a latent vector independently for each character.
- Text mapping in the current model recognizes letters and spaces only. Other characters are discarded by `text_to_indices` today.

### Training

- `train.py` reads `config/default.yaml`, loads EMNIST, samples K examples per class, splits validation data, builds `FewShotVAE`, and invokes `Trainer`.
- `src/training/losses.py` defines BCE reconstruction + beta-weighted KL + gamma-weighted classifier cross-entropy.
- `src/training/trainer.py` uses Adam, gradient clipping, periodic checkpoints, best-checkpoint tracking, and reconstruction snapshots.
- `test_pipeline.py` and `test_full_training.py` are experiment scripts that train for a few/50 epochs and generate a sample page; they are not isolated unit tests.
- `colab_notebook.py` is a separate script that writes project source files and then trains/generates in a Colab-like workflow. It duplicates implementation and configuration, so it can drift from the canonical source tree.

### Generation and composition

- `generate.py` provides `chars` and `page` modes, accepts either explicit reference paths or a reference directory, and reads architecture settings from the checkpoint config.
- `src/utils/page_composer.py` estimates a character cell size, adds spacing/rotation/jitter, and composites characters into a page.
- `src/utils/paper_backgrounds.py` creates four paper styles at nominal A4/300 DPI dimensions (2480x3508 pixels).
- Page mode saves both `handwritten_page.png` and `handwritten_page.pdf`.

## 7. Verified Repository State (2026-09-26)

- A checkpoint set exists in `outputs/checkpoints/`: best, final, and epochs 5, 10, 20, 30, 40, and 50.
- Inspection of `vae_final.pt` shows epoch 50, K=5, latent_dim=64, style_dim=32, num_classes=26, char_embed_dim=16, beta=1.0, and 50 history entries.
- That checkpoint's history contains only total, reconstruction, KL, and time metrics; it predates the current classifier-loss configuration. Its final recorded total loss is about 204.33 and reconstruction loss about 181.32. These values are not quality scores and should not be treated as evidence of legible output.
- The current `config/default.yaml` instead specifies K=10, latent_dim=8, char_embed_dim=64, beta=0.5, gamma=10, and 200 epochs. No 200-epoch checkpoint was found in the current artifact inventory.
- `outputs/generated/` contains example PNGs, including a 50-epoch page and character grids. Visual inspection shows very faint/noisy, unreadable output. Thus the older end-to-end path has run, but product-quality generation has not been demonstrated.
- `samples/refs/` contains five 64x64 files named `a.png`, `d.png`, `h.png`, `k.png`, and `m.png`. Their existence and appearance do not establish that they are genuine same-writer samples; the test and Colab scripts can create font-rendered synthetic substitutes.
- The Python diagnostics queried for `train.py`, `generate.py`, `fewshot_vae.py`, and `trainer.py` reported no editor errors. No full test suite was run as part of this handoff.

## 8. Known Gaps and Risks

### P0: Model/checkpoint version mismatch

The canonical default config describes the newer classifier-conditioned architecture, while the saved 50-epoch checkpoints and both training experiment scripts use the older 64/32/16 architecture without a gamma classifier-loss setting. The checkpoint is loadable by generation because its config is used, but it does not validate the current default design. Do not overwrite or label old checkpoints as results from the new model.

### P0: Pixel polarity and output quality

Reference images are processed without explicit foreground inversion. Page composition treats generated pixel intensity as ink alpha. Confirm the actual EMNIST and reference polarity with representative tensors and a visual test; the current output being faint/noisy makes this a first-order quality issue. Define one convention and test it end to end before long training.

### P0: Evaluation is insufficient

There are no focused assertions for model tensor shapes, label mapping, checkpoint compatibility, generation order, or page layout. Current training metrics alone do not measure recognizable letters or writer-style similarity. Establish an evaluation set and a visual/quantitative acceptance protocol before tuning the new architecture.

### P1: Model factorization is an assumption, not a demonstrated property

The small latent and large character embedding are intended to force character identity into the embedding, but dimensionality alone does not guarantee disentanglement. The style encoder receives reference images that are not labeled as belonging to a particular requested character, while the encoder/decoder training path uses EMNIST samples. Evaluate whether style changes with reference writer while character identity remains fixed.

### P1: Text token alignment

`text_to_indices` omits punctuation and newline tokens, but page composition indexes generated outputs against the original text. Unsupported text can therefore disappear or shift later characters. Spaces are represented as `None`; newline handling needs an explicit, consistent token contract.

### P1: Training and validation metrics are incomplete/inconsistent

Trainer validation computes classifier loss as part of total loss but does not report validation classifier loss. Reconstruction snapshots call the model's stochastic reconstruction path. The CLI's final summary and some experiment scripts assume metric names that differ from the Colab script. Consolidate a single history schema.

### P1: Dataset portability and reproducibility

EMNIST raw files are present under `data/EMNIST/raw`, but the expected torchvision loading/download behavior was not verified in a clean environment. Few-shot selection scans the dataset through the transformed dataset, which can invoke random augmentation during selection. DataLoader shuffling and per-worker randomness are not fully seeded.

### P2: Page layout is character-cell based

Each letter is independently resized into a square and rendered as a separate glyph. This limits natural proportions, baseline behavior, kerning, and cursive continuity. Current generation stops at the bottom of one page rather than explicitly reporting truncation or producing multiple pages.

### P2: Repository hygiene and documentation

There is no root README in the listed project tree. `colab_notebook.py` generates source files as text, duplicates the canonical implementation, and includes experiment-specific behavior; choose whether to maintain it or replace it with instructions that invoke the shared code.

## 9. Recommended Continuation Plan for GPT-6 Astra

Proceed in this order. Keep the existing checkpoint artifacts intact.

1. **Establish a baseline contract.** Read `config/default.yaml`, `src/models/fewshot_vae.py`, `src/training/trainer.py`, `train.py`, and `generate.py`. Record the intended current architecture as 8/32/64 with beta=0.5 and gamma=10. Record the existing 50-epoch checkpoint as legacy 64/32/16.
2. **Prove data and rendering conventions.** Load a small batch from EMNIST and one reference; save a montage with labels. Check foreground/background polarity and orientation. Pass known synthetic glyph tensors through composition and assert visible ink on each supported paper background.
3. **Add narrow tests before expensive training.** Test EMNIST label range, few-shot counts and deterministic selection, model output shapes, loss finiteness and gradients, supported text tokenization, checkpoint save/load, and page layout/wrapping. Use tiny in-memory fixtures where possible so tests do not require downloads or long training.
4. **Resolve the model-version decision.** Make the 8/32/64 classifier-conditioned model the canonical experiment, or explicitly select the older architecture. Align both experiment scripts and Colab usage with that choice. Store unique checkpoint names/config snapshots per experiment.
5. **Fix text and polarity contracts.** Preserve one output token per input character/layout token; reject unsupported characters clearly. Correct normalization/inversion and alpha compositing with tests. Do not tune network architecture to compensate for a preprocessing bug.
6. **Run a short controlled training experiment.** Use a small K-shot subset and few epochs, seed all RNG sources, record train/validation reconstruction, KL, and classifier metrics, and inspect per-class generated grids. Compare output to a simple nearest-neighbor/reference baseline.
7. **Evaluate the stated product goals.** Use held-out EMNIST letters for recognizability and multiple reference writers for style retention. Report character accuracy from an independently trained/frozen evaluator, reconstruction metrics, and a blinded visual review. Keep metrics and sample images tied to a specific checkpoint/config.
8. **Only then train the full default run.** Run the 200-epoch config after the short experiment demonstrates improvement. Save curves, config, checkpoint metadata, and a fixed-seed sample page for comparison.
9. **Finish the user workflow.** Add a concise README with environment setup and exact train/generate commands; ensure clear errors for unsupported text, bad references, missing checkpoints, and page overflow.

## 10. Acceptance Criteria

The MVP is ready for a demo only when all of the following are true:

- A clean environment can install declared dependencies, load the selected dataset, run a short training smoke test, and load the resulting checkpoint for generation.
- The checkpoint's architecture/config can be reproduced from its metadata; no ambiguous legacy/current checkpoint naming remains.
- Unit tests cover label mapping, deterministic K-shot selection, model/loss contracts, text tokens, checkpoint round-trip, and page composition.
- A fixed-seed character grid renders all A-Z as visible, correctly oriented glyphs with no blank or clipped outputs; a reviewer can identify at least 90% of generated uppercase letters in a documented held-out sample review. This threshold is a proposed MVP gate and should be revisited if the evaluation set is too small.
- Using a different reference set changes visual style while the same requested character remains recognizable; assess with a documented multi-writer comparison rather than claiming this from architecture alone.
- A mixed-case message with spaces and newlines retains exact character order and layout. Unsupported punctuation produces a clear error until punctuation is implemented.
- Page output has correct ink/paper polarity, respects margins, is not silently truncated, and opens as both PNG and PDF.
- Training and generation commands, expected inputs, outputs, and known limitations are documented.

## 11. Existing Commands and Paths

Run commands from the `fewshot-handwriting-vae-v2` project directory.

Install declared Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train with the current default configuration (200 epochs; do not treat this as validated until the P0 items above are addressed):

```powershell
python train.py --config config/default.yaml
```

Generate a page using an existing checkpoint and the sample references:

```powershell
python generate.py --checkpoint outputs/checkpoints/vae_final.pt --ref_dir samples/refs --text "HELLO WORLD" --paper lined --mode page
```

Important: the checked-in `vae_final.pt` is a 50-epoch legacy checkpoint, not a model trained from the current default config. Generation reads model dimensions from the checkpoint; retain that behavior and label the artifact accordingly.

## 12. Source Map

- Entrypoints: `train.py`, `generate.py`
- Current training config: `config/default.yaml`
- Data and few-shot sampling: `src/data/dataset.py`
- Conditioned model and text mapping: `src/models/fewshot_vae.py`
- Style encoder: `src/models/style_encoder.py`
- Losses and trainer: `src/training/losses.py`, `src/training/trainer.py`
- Page and paper rendering: `src/utils/page_composer.py`, `src/utils/paper_backgrounds.py`
- Experiment scripts: `test_pipeline.py`, `test_full_training.py`, `colab_notebook.py`
- Legacy checkpoint: `outputs/checkpoints/vae_final.pt`
- Existing visual evidence: `outputs/generated/`
- Current reference samples: `samples/refs/`