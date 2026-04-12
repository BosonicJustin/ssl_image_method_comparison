# Self-Supervised Image Representation Learning: A Comparative Study

Systematic comparison of self-supervised learning methods on STL-10, evaluating how different pretext tasks shape learned visual representations.

## Methods

| Method | Family | Core Idea | Backbone |
|--------|--------|-----------|----------|
| **Autoencoder** | Reconstruction | Compress and reconstruct full image through bottleneck | CNN encoder-decoder |
| **MAE** | Masked Reconstruction | Mask 75% of ViT patches, reconstruct pixels from visible patches | ViT |
| **SimCLR** | Contrastive | Pull augmented views of same image together, push different images apart | CNN / ViT encoder + projection head |
| **BYOL** | Self-Distillation | Online network predicts target network representations; no negatives needed | CNN / ViT encoder + projector + predictor |
| **I-JEPA** | Joint-Embedding Predictive | Predict target patch representations from context patches (no pixel reconstruction) | ViT |

## Dataset: STL-10

- **96x96** color images, 10 classes
- **5,000** labeled training images (500/class)
- **8,000** labeled test images
- **100,000** unlabeled images (used for self-supervised pretraining)

The large unlabeled pool makes STL-10 ideal for self-supervised benchmarking.

## Evaluation Protocol

All evaluations run on **frozen** encoder features (no finetuning of the backbone).

### 1. k-NN Accuracy (Frozen)
Extract features from the trained encoder, run k-NN (k=20) on the test set. No training involved -- purely measures whether the learned feature space groups semantically similar images together.

### 2. Attention Map Visualization
For ViT-based models (MAE, I-JEPA, optionally SimCLR/BYOL with ViT backbone), visualize attention from [CLS] token across heads and layers. Goal: does the model attend to the foreground object or background texture?

### 3. Low-Data Linear Probe (1% Labeled)
Train a linear classifier on frozen features using only ~50 labeled images (1% of STL-10 train). This stress-tests feature quality -- good representations should be linearly separable even with minimal supervision.

## Project Structure

```
.
├── models/              # Model definitions (one file per method)
│   ├── autoencoder.py
│   ├── mae.py
│   ├── simclr.py
│   ├── byol.py
│   └── ijepa.py
├── configs/             # Training configs (YAML)
├── evaluation/          # k-NN, linear probe, attention vis
├── utils/               # Data loading, augmentations, logging
├── scripts/             # Training & eval entry points
├── notebooks/           # Analysis & visualization notebooks
├── results/             # Outputs, plots, metrics
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Key Design Decisions

> TBD -- see followup questions below. These decisions will be locked in before implementation begins.

- Backbone architecture: unified ViT across all methods, or method-native (CNN for AE, ViT for MAE/I-JEPA)?
- ViT scale: ViT-Tiny (5.7M params) vs ViT-Small (22M) given STL-10's 96x96 resolution?
- Pretraining epochs and compute budget
- Augmentation strategy per method
