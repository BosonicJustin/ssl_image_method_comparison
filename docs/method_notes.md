# SSL Method Notes

## Masked Autoencoder (MAE)

### How it works

1. **Patch & mask** — Split the image into non-overlapping patches (e.g., 6x6 grid of 16x16 patches for 96x96 images). Randomly mask 75%, keep 25% visible.

2. **Encode visible patches only** — The visible patches are linearly projected into token vectors. Learnable positional embeddings (one per grid position) are added according to each patch's original position, so the encoder knows where in the image each patch came from. The sequence (+ optional CLS token) is passed through a deep ViT encoder (stacked transformer blocks: LayerNorm -> MHSA -> residual -> LayerNorm -> MLP -> residual).

3. **Insert mask tokens** — After encoding, a shared learnable `[MASK]` vector is placed at every masked position. Positional embeddings are added again so the decoder can distinguish mask positions from each other.

4. **Decode full sequence** — The full sequence (encoded visible tokens + mask tokens) is passed through a shallow ViT decoder (e.g., 4 blocks vs 12 in the encoder).

5. **Reconstruct pixels** — The decoder output for each masked patch is projected to pixel values. Loss = MSE on masked patches only.

### Why the encoder is efficient

The expensive deep encoder only processes ~25% of tokens. The cheap shallow decoder handles the full sequence. This makes pretraining roughly 3-4x faster than processing all patches.

### MAE vs CNN Autoencoder

| | CNN Autoencoder | MAE |
|---|---|---|
| **Bottleneck** | Architectural (narrow latent vector) | Information gap (75% masking) |
| **Representation** | Single global vector | Sequence of contextualized patch tokens |
| **Output** | Reconstructs full image | Reconstructs only masked patches |
| **Backbone** | CNN encoder-decoder | ViT encoder + shallow ViT decoder |
| **Scaling** | Limited | Benefits from larger models/data |

### MAE weaknesses

- **Pixel-level objective** — Wastes capacity learning high-frequency texture details (exact colors, edges, noise) that aren't semantically meaningful. Can become a great texture synthesizer without understanding image content.
- **No semantic pressure** — Nothing in the loss encourages grouping semantically similar images. Contrastive methods (SimCLR, BYOL) directly optimize for this.
- **Decoder dependence** — A too-strong decoder can carry reconstruction, letting the encoder be lazy.

## SimCLR

### How it works

1. **Dual-view augmentation** — Each image is augmented twice (random crop, color jitter, blur, flip) to produce two views.
2. **Encode** — Both views pass through a shared encoder (ResNet-18) producing feature vectors.
3. **Project** — Features pass through a projection head (MLP with BN) to a lower-dim space.
4. **NT-Xent loss** — Pull projections of the same image together, push different images apart. Cosine similarity scaled by temperature, cross-entropy over positive pairs vs all negatives in the batch.

Evaluation uses encoder features **before** the projection head (paper finding: encoder features outperform projection features for downstream tasks).

### SimCLR strengths

- Direct semantic pressure — the loss explicitly groups similar images together.
- Simple and well-understood.

### SimCLR weaknesses

- Needs large batch sizes for enough negatives.
- Sensitive to augmentation strategy.

## BYOL (Bootstrap Your Own Latent)

### How it works

BYOL has **two networks** looking at **two augmented views** of the same image:

1. **Online network** — encoder + projector + predictor. Processes view 1.
2. **Target network** — encoder + projector only (no predictor). Processes view 2. Its weights are an **exponential moving average (EMA)** of the online network — never trained by gradient descent directly.

The online network tries to **predict the target network's representation** of the other view. Loss = MSE between the online prediction and the target projection (both L2-normalized).

After each step, the target network is updated: `target = momentum * target + (1 - momentum) * online` (momentum ~0.996).

### BYOL vs Autoencoder

| | Autoencoder | BYOL |
|---|---|---|
| **Predicts** | Pixels (reconstruct the image) | Representations (match another network's embedding) |
| **Architecture** | Encoder-decoder | Twin encoders (online + target) with predictor |
| **What it learns** | Compression/reconstruction | View-invariant semantic features |
| **Needs negatives?** | N/A | No — this is the key innovation |

### BYOL vs SimCLR

SimCLR needs negatives (other images in the batch) to prevent collapse — without them, the model could output a constant vector for everything. BYOL avoids collapse through the **asymmetry** between the two networks: the predictor (only on online) and the EMA update (target is a slow-moving version of online) together prevent degenerate solutions.

### BYOL weaknesses

- Sensitive to EMA momentum schedule and augmentation
- Harder to understand theoretically — why it doesn't collapse is still debated (batch norm, EMA, predictor all play a role)

## I-JEPA (motivation)

Keeps the masking/prediction idea from MAE but predicts **patch representations** instead of pixels. This avoids the pixel-level texture trap — the model is forced to learn semantic features rather than low-level reconstruction details.
