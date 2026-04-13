# Prediction-Ready Representations: A Systematic Comparison of Self-Supervised Learning Objectives for Latent-Space Dynamics

## Abstract

Recent work demonstrates that freezing a pre-trained self-supervised encoder and training a lightweight predictor in its latent space can match or exceed end-to-end approaches for planning and control (Bharadhwaj et al., 2025; Assran et al., 2025; Apple Research, 2025). However, each study evaluates only a single encoder, leaving unanswered the question of which self-supervised learning (SSL) objective produces representations most amenable to downstream prediction. We propose a controlled comparison of six SSL objectives — contrastive (SimCLR), self-distillation (BYOL, DINO), pixel reconstruction (Autoencoder, MAE), and joint-embedding predictive (I-JEPA) — trained under identical conditions, then frozen while a standardised predictor is trained in each latent space. We evaluate across spatial prediction on static images, temporal prediction on video, multi-step error accumulation, and geometric analysis of the latent space. Our central hypothesis is that joint-embedding predictive representations (I-JEPA) will exhibit the lowest compounding error under multi-step rollouts, because their pre-training objective directly optimises for predictability in representation space. This work aims to provide the first systematic evidence linking SSL objective choice to latent-space prediction quality — a prerequisite for principled world-model design.

## 1. Introduction

### 1.1 The Frozen-Encoder Paradigm

A growing body of work decouples visual representation learning from downstream prediction. Rather than training an encoder jointly with a dynamics model or policy — as in the Dreamer family (Hafner et al., 2020; 2023) or TD-MPC (Hansen et al., 2022; 2023) — these approaches first train a self-supervised encoder on unlabelled data, freeze it, and then train a separate predictor that operates entirely in the frozen latent space. DINO-WM (Bharadhwaj et al., 2025) freezes a DINOv2 encoder and trains a ViT-based dynamics model on its patch embeddings, achieving zero-shot planning that outperforms Dreamer and TD-MPC on manipulation tasks. V-JEPA 2-AC (Assran et al., 2025) freezes a V-JEPA encoder pre-trained on internet video and trains an action-conditioned latent predictor, achieving 65-80% success on real robot pick-and-place tasks with no task-specific reward. SALT (Apple Research, 2025) freezes a teacher encoder and trains a student to predict its representations, outperforming V-JEPA 2 with 20-30% fewer FLOPs.

This paradigm offers compelling advantages. A frozen encoder provides a stable coordinate system for the predictor — eliminating the moving-target problem that plagues joint training, where the representation shifts under the dynamics model at every gradient step. It enables compositional reuse: multiple predictors (different tasks, horizons, or planning algorithms) can share the same frozen representation without interfering. And it is computationally efficient: the encoder is trained once, amortised across all downstream prediction tasks.

### 1.2 The Missing Comparison

Despite these successes, each study selects a single SSL encoder and demonstrates that it works. No existing work systematically compares how different SSL objectives shape the latent space for prediction. This gap is consequential. Schneider et al. (2024) found that pre-trained visual representations are "surprisingly ineffective" for model-based reinforcement learning, suggesting that the choice of SSL objective matters far more than whether to use pre-training at all. Van Assel et al. (2025) provide a theoretical basis: joint-embedding methods impose a strictly weaker alignment condition than reconstruction methods, making them provably superior when irrelevant features have large magnitude — a common property of natural images. Yet neither study identifies which specific SSL objective to choose.

LeCun (2022) articulates the broader vision: a hierarchical architecture (H-JEPA) in which representations are learned at each level and predictive world models are trained in those representation spaces. The JEPA framework explicitly argues that prediction should happen in abstract representation space rather than pixel space, because pixel-level prediction wastes capacity on high-frequency details (texture, noise, exact colour) that are unpredictable and semantically irrelevant. But this leaves open the empirical question: among the diversity of SSL methods available today, which ones actually produce latent spaces where prediction works best?

### 1.3 Contributions

We propose the first controlled study comparing SSL representations as substrates for latent-space prediction. Our contributions are:

1. **Controlled comparison**: Six SSL objectives trained under identical conditions (architecture, optimiser, schedule, data), eliminating confounds from training recipes.
2. **Prediction-specific evaluation**: A measurement framework comprising spatial prediction, transformation prediction, temporal prediction, and multi-step error accumulation — metrics that directly probe prediction amenability rather than classification accuracy.
3. **Geometric characterisation**: Analysis of latent-space properties (local linearity, effective dimensionality, smoothness) and their correlation with prediction quality.
4. **Static-to-temporal transfer**: Extension from spatial prediction on images to temporal prediction on video, testing whether findings generalise across prediction domains.

## 2. Background and Related Work

### 2.1 Self-Supervised Learning Families

We organise SSL methods along two axes: the prediction target (pixels vs. representations) and the learning signal (augmentation invariance vs. masking).

**Contrastive methods** (Chen et al., 2020; He et al., 2020) learn by pulling augmented views of the same image together while pushing different images apart. SimCLR (Chen et al., 2020) uses in-batch negatives with an NT-Xent loss. Representations are augmentation-invariant by construction, yielding strong linear-probe accuracy but potentially discarding augmentation-sensitive information relevant to dynamics.

**Self-distillation methods** (Grill et al., 2020; Caron et al., 2021) use a teacher-student setup without explicit negatives. BYOL (Grill et al., 2020) prevents collapse via an EMA target network and a predictor asymmetry. DINO (Caron et al., 2021) uses centering and sharpening of the teacher output, producing features with emergent semantic segmentation properties. DINOv2 (Oquab et al., 2023) scales this approach to 142M curated images.

**Reconstruction methods** predict pixels from compressed or masked input. Convolutional autoencoders use an architectural bottleneck. MAE (He et al., 2022) masks 75% of ViT patches and reconstructs pixel values for masked patches only. Both encode all visual information including low-level texture, which Van Assel et al. (2025) show is theoretically disadvantageous when irrelevant features dominate.

**Joint-embedding predictive methods** predict representations rather than pixels. I-JEPA (Assran et al., 2023) masks large contiguous blocks and predicts their representations from context using an EMA target encoder. By operating in representation space, the model is freed from predicting unpredictable pixel-level details. V-JEPA (Bardes et al., 2024) extends this to video with spatiotemporal masking, outperforming VideoMAE by +4.2 points on standard benchmarks and substantially outperforming it on intuitive physics tasks.

**Redundancy-reduction methods** (Zbontar et al., 2021; Bardes et al., 2022) prevent collapse by decorrelating feature dimensions. Barlow Twins (Zbontar et al., 2021) pushes the cross-correlation matrix of embeddings toward identity. This family is notable because NE-Dreamer (2026) and R2-Dreamer (2026) use Barlow Twins loss for decoder-free world models.

### 2.2 Frozen Encoders for Planning and World Models

The dominant paradigm in model-based RL trains encoders jointly with dynamics models. The Dreamer family (Hafner et al., 2020; 2022; 2023) uses a Recurrent State-Space Model with joint encoder-decoder training. TD-MPC (Hansen et al., 2022; 2023) trains an implicit (decoder-free) model end-to-end via temporal-difference learning. IRIS (Micheli et al., 2023) trains a discrete autoencoder followed by an autoregressive transformer. MuZero (Schrittwieser et al., 2020) trains a latent dynamics model to predict rewards, values, and policies rather than observations.

The frozen-encoder alternative is more recent. DINO-WM (Bharadhwaj et al., 2025) demonstrated that frozen DINOv2 patch embeddings support zero-shot planning that outperforms jointly-trained baselines on manipulation and locomotion tasks. V-JEPA 2-AC (Assran et al., 2025) scaled this to real robot control using a frozen V-JEPA encoder pre-trained on 1M+ hours of internet video. SALT (Apple Research, 2025) showed that a frozen teacher encoder produces students that outperform V-JEPA 2 with substantially less compute, and that student quality is remarkably robust to teacher quality.

However, Schneider et al. (2024) provide a critical counterpoint: in their benchmark of pre-trained visual representations for model-based RL, current pre-trained encoders did not improve sample efficiency or generalisation over learning from scratch. This suggests that naive application of frozen encoders is insufficient — the choice of SSL objective and the structure of the latent space likely matter.

### 2.3 Decoder-Free World Models

A parallel trend is eliminating pixel reconstruction from world models entirely. NE-Dreamer (2026) replaces reconstruction with next-step encoder embedding prediction aligned via Barlow Twins, matching DreamerV3 on DMControl and substantially outperforming it on memory-intensive tasks. R2-Dreamer (2026) uses redundancy reduction between the encoder output and RSSM latent state, training 1.59x faster than DreamerV3. Dreamer-CDP (2026) introduces a JEPA-style predictor on continuous deterministic representations. These results suggest the field is converging on representation-space prediction as a replacement for pixel reconstruction — further motivating the question of which representations are best suited for this role.

### 2.4 Theoretical Foundations

LeCun (2022) argues that JEPA architectures are fundamentally better suited for world models than generative models because they can abstract away unpredictable details rather than being forced to model them. Van Assel et al. (2025) formalise this: joint-embedding methods impose a weaker alignment condition than reconstruction, making them provably superior when the data contains large-magnitude irrelevant features. Theoretical analysis of I-JEPA's implicit bias (NeurIPS 2024) shows that JEPA training preferentially learns "influential features" (high regression coefficient) rather than merely high-variance features, explaining why it produces more semantic representations than MAE.

## 3. Research Questions

**RQ1 (Core).** Which SSL objectives produce frozen representations most amenable to latent-space prediction, as measured by predictor accuracy and multi-step error accumulation?

**RQ2 (Geometry).** What geometric properties of a latent space — local linearity, effective dimensionality, smoothness, isotropy — correlate with prediction amenability?

**RQ3 (Training dynamics).** How does prediction amenability evolve during encoder training? Does the optimal encoder checkpoint for prediction differ from the optimal checkpoint for classification?

**RQ4 (Frozen vs. joint).** Under what conditions can the two-stage approach (frozen encoder + trained predictor) match jointly-trained end-to-end systems? When does it fail?

**RQ5 (Domain transfer).** Do findings from static spatial prediction on images transfer to temporal prediction on video? Does the encoder need exposure to temporal structure during pre-training?

## 4. Methodology

### 4.1 Phase 0: Encoder Suite

All encoders are trained on STL-10 unlabelled data (100,000 images, 96x96) under a unified protocol: 500 epochs, batch size 256, AdamW optimiser (lr=1e-3, weight decay=1e-5), cosine annealing to zero. Only the SSL objective varies.

| Method | Family | Backbone | Feature dim | Status |
|---|---|---|---|---|
| SimCLR | Contrastive | ResNet-18 | 512 | Trained |
| BYOL | Self-distillation | ResNet-18 | 512 | Trained |
| Autoencoder | Reconstruction | ResNet-18 | 512 | Trained |
| MAE | Masked reconstruction | ViT-Small | 384 | Training |
| I-JEPA | Joint-embedding predictive | ViT-Small | 384 | Implemented |
| DINO | Self-distillation (ViT) | ViT-Small | 384 | To implement |

Controls: a randomly initialised encoder (lower bound) and a supervised pre-trained encoder (reference). Optional: Barlow Twins (redundancy reduction family, motivated by NE-Dreamer).

### 4.2 Phase 1: Static Image Experiments

**Experiment 1 — Spatial prediction on frozen representations.** For each frozen encoder, extract patch-level representations for all STL-10 images. Train a lightweight predictor (small transformer, architecture held constant) to predict target patch representations from context patches, using I-JEPA's multi-block masking strategy. Metrics: MSE in representation space; cosine similarity between predicted and actual representations; downstream classification accuracy (k-NN, linear probe) evaluated on predicted representations to measure semantic fidelity.

**Experiment 2 — Multi-step spatial prediction.** Chain predictions: predict block A from context, then predict block B using predicted A as input, continuing for N steps. Measure error at each step. Fit exponential growth rates. This directly measures the effective planning horizon each latent space supports.

**Experiment 3 — Transformation prediction.** Given a frozen representation z(x) and a geometric transformation (rotation, translation, scaling), train a predictor to map (z(x), transformation parameters) to z(T(x)). This simulates action-conditioned prediction. Note: contrastive/self-distillation methods are explicitly trained for augmentation invariance, so they should struggle with this task — testing the invariance-equivariance trade-off.

**Experiment 4 — Latent space geometry.** For each frozen encoder, measure: effective dimensionality (PCA participation ratio), local linearity (consistency of linear interpolations), smoothness (Lipschitz constant estimation), and isotropy (eigenvalue ratio of representation covariance). Correlate each metric with prediction performance from Experiments 1-3.

**Experiment 5 — Encoder quality curve.** Using periodic checkpoints saved during encoder training (every 50 epochs), freeze the encoder at each stage and run Experiment 1. Plot prediction quality versus encoder training progress alongside k-NN accuracy. Test whether the classification-optimal and prediction-optimal checkpoints diverge.

**Experiment 6 — Frozen vs. joint training.** Compare three regimes: (a) frozen SSL encoder + separate predictor, (b) SSL-initialised encoder fine-tuned jointly with predictor, (c) randomly initialised encoder + predictor trained end-to-end. This quantifies the cost of freezing and the value of SSL initialisation.

### 4.3 Phase 2: Video/Temporal Experiments

**Datasets.** Moving MNIST or bouncing-ball physics simulations as a proof-of-concept environment with known dynamics. Something-Something v2 (Goyal et al., 2017) as the main benchmark — its temporal directionality (e.g., "pushing left to right" vs. "right to left") ensures that temporal prediction is genuinely required. Optionally, BAIR Robot Pushing (Ebert et al., 2017) for action-conditioned prediction with recorded actions.

**Experiment 7 — Temporal prediction in frozen latent space.** Encode each video frame with a frozen encoder. Train a temporal predictor (small transformer or GRU) on frozen frame representations: given [z_1, ..., z_t], predict z_{t+1}. Use the same predictor architecture across all encoders. Evaluate prediction MSE, cosine similarity, and action-recognition accuracy from predicted representations.

**Experiment 8 — Temporal rollout depth.** Chain temporal predictions autoregressively: z_t -> z_{t+1} -> z_{t+2} -> ... -> z_{t+N}, using predicted (not ground-truth) representations as input at each step. Measure error accumulation. This is the most direct proxy for world-model planning horizon.

**Experiment 9 — Image encoders vs. video encoders.** Compare frozen image encoders (from Phase 1, applied frame-by-frame) against frozen video encoders (V-JEPA, VideoMAE) that were exposed to temporal structure during pre-training. This tests whether the encoder needs temporal awareness, or whether a strong spatial encoder combined with a temporal predictor suffices.

**Experiment 10 — Action-conditioned prediction (optional).** On BAIR Robot Pushing or DMControl: train an action-conditioned predictor (z_t, a_t) -> z_{t+1} on frozen representations. Evaluate planning by optimising action sequences (via CEM or gradient-based search) to reach goal states in latent space. Compare with DINO-WM and DreamerV3 baselines.

### 4.4 Predictor Architecture

To isolate the effect of the encoder, the predictor architecture is standardised across all experiments. For spatial and temporal prediction: a small transformer (4 blocks, 6 attention heads, matching the encoder's embedding dimension). For transformation prediction: an MLP mapping (z, action_params) to z'. The predictor is deliberately capacity-limited so that prediction quality depends primarily on the structure of the latent space rather than the predictor's memorisation capacity.

## 5. Hypotheses

Based on the theoretical and empirical evidence reviewed above, we make the following predictions:

**H1.** I-JEPA representations will exhibit the lowest multi-step prediction error accumulation, because the I-JEPA pre-training objective is itself a latent-space prediction task — the representation is shaped to be predictable from partial context.

**H2.** Self-distillation methods (BYOL, DINO) will achieve competitive single-step prediction accuracy due to their smooth, semantically structured latent spaces, but will accumulate error faster than I-JEPA over multi-step rollouts because their training objective does not directly incentivise predictability.

**H3.** Contrastive representations (SimCLR) will show high local linearity but will underperform on transformation prediction (Experiment 3) because they are explicitly trained for augmentation invariance — the transformations are precisely what the encoder was trained to discard.

**H4.** Reconstruction methods (MAE, Autoencoder) will show the weakest prediction amenability because they encode high-frequency visual details (texture, noise, exact colour) that are difficult to predict and semantically irrelevant, consistent with both the theoretical results of Van Assel et al. (2025) and the empirical findings of Schneider et al. (2024).

**H5.** The encoder checkpoint that maximises classification accuracy (k-NN or linear probe) will not be the same checkpoint that maximises prediction quality. Classification rewards discriminative features; prediction rewards smooth, structured, predictable features. These objectives may diverge, particularly for contrastive methods.

## 6. Expected Contributions

**Primary.** The first systematic study comparing SSL representations as substrates for latent-space prediction. While DINO-WM, V-JEPA 2-AC, and SALT each demonstrate the frozen-encoder paradigm with one encoder, we provide the missing comparative analysis across the major SSL families.

**Measurement framework.** A suite of prediction-specific metrics — spatial prediction accuracy, transformation prediction, multi-step error accumulation rate, and geometric characterisation — that capture "prediction readiness" rather than the standard classification metrics (k-NN, linear probe) used in SSL evaluations.

**Geometric insights.** Characterisation of which latent-space properties (linearity, dimensionality, smoothness) are shaped by which SSL objectives, and which properties predict downstream prediction quality. This provides interpretable guidance beyond "method X beats method Y."

**Practical guidelines.** Actionable recommendations for practitioners building frozen-encoder world models: which SSL objective to choose, how long to train it, and when the frozen approach suffices versus when joint training is needed.

## 7. References

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR*.

Assran, M., Bardes, A., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. *arXiv:2506.09985*.

Apple Research (2025). SALT: Rethinking JEPA — Compute-Efficient Video SSL with Frozen Teachers. *arXiv:2509.24317*.

Balestriero, R., et al. (2023). A Cookbook of Self-Supervised Learning. *arXiv:2304.12210*.

Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., & Ballas, N. (2024). V-JEPA: Latent Video Prediction for Visual Representation Learning. *TMLR*.

Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR*.

Bharadhwaj, H., et al. (2025). DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning. *ICML*.

Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV*.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*.

Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, P. H., Buchatskaya, E., ... & Valko, M. (2020). Bootstrap Your Own Latent. *NeurIPS*.

Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR*.

Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104*.

Hansen, N., Su, H., & Wang, X. (2022). Temporal Difference Learning for Model Predictive Control. *ICML*.

Hansen, N., Su, H., & Wang, X. (2023). TD-MPC2: Scalable, Robust World Models for Continuous Control. *arXiv:2310.16828*.

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.

He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. *CVPR*.

LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview*.

Micheli, V., Alonso, E., & Fleuret, F. (2023). Transformers are Sample-Efficient World Models. *ICLR*.

NE-Dreamer (2026). Next Embedding Prediction Makes World Models Stronger. *arXiv:2603.02765*.

Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv:2304.07193*.

R2-Dreamer (2026). Redundancy-Reduced World Models. *ICLR*. *arXiv:2603.18202*.

Schneider, T., Mosbach, M., et al. (2024). The Surprising Ineffectiveness of Pre-Trained Visual Representations for Model-Based RL. *NeurIPS*.

Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature*.

Schwarzer, M., Anand, A., Goel, R., Hjelm, R. D., Courville, A., & Bachman, P. (2021). Data-Efficient Reinforcement Learning with Self-Predictive Representations. *ICLR*.

Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. *NeurIPS*.

Van Assel, R., et al. (2025). Joint-Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for SSL. *NeurIPS*.

Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. *ICML*.
