# Reading List: Prediction in Frozen SSL Latent Spaces

Track reading progress and key insights for the research project.

Status key: `[ ]` unread | `[~]` skimmed | `[x]` read

---

## Tier 1: Must-Read (directly shape experimental design)

- [ ] **Terver et al. 2025** — "What Drives Success in Physical Planning with JEPA World Models?"
  [arXiv:2512.24497](https://arxiv.org/abs/2512.24497) | [Code](https://github.com/facebookresearch/jepa-wms)
  Closest existing work. Compares frozen DINOv2, DINOv3, V-JEPA, V-JEPA-2 for latent planning.
  DINO image encoders beat V-JEPA video encoders for manipulation. Only tests DINO/V-JEPA family — our project fills the gap with contrastive/reconstruction methods.
  - Insights:

- [ ] **Garrido et al. 2024** — "Learning and Leveraging World Models in Visual Representation Learning"
  [arXiv:2403.00504](https://arxiv.org/abs/2403.00504)
  Invariant representations (contrastive) → better linear probes. Equivariant representations (MAE-like) → better world model fine-tuning with learned predictors. May flip our hypothesis that MAE will be worst.
  - Insights:

- [ ] **Schneider et al. NeurIPS 2024** — "The Surprising Ineffectiveness of Pre-Trained Visual Representations for Model-Based RL"
  [arXiv:2411.10175](https://arxiv.org/abs/2411.10175) | [Project](https://schneimo.com/pvr4mbrl/)
  8 encoders (CLIP, DINOv2, R3M, VC-1, etc.) tested on DreamerV3/TD-MPC2. PVRs fail at reward prediction — representations don't separate reward-relevant states. But see the rebuttal below.
  - Insights:

- [ ] **Littwin et al. NeurIPS 2024** — "How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self-Distillation Networks"
  [arXiv:2407.03475](https://arxiv.org/abs/2407.03475)
  JEPA preferentially encodes features with high regression coefficients (high signal-to-noise). MAE encodes high-covariance features (potentially noisy). Deep encoders amplify this difference. Key theoretical prediction for why I-JEPA should produce better prediction substrates.
  - Insights:

- [ ] **Van Assel et al. NeurIPS 2025** — "Joint-Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for SSL"
  [arXiv:2505.12477](https://arxiv.org/abs/2505.12477)
  JE needs strictly less augmentation strength than reconstruction to filter irrelevant features. Formal proof that JE > reconstruction when irrelevant features have large magnitude.
  - Insights:

---

## Tier 2: Should-Read (inform methodology and interpretation)

### Frozen encoder systems

- [x] **Bharadhwaj et al. ICML 2025** — "DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning"
  [arXiv:2411.04983](https://arxiv.org/abs/2411.04983) | [Project](https://dino-wm.github.io/)
  Frozen DINOv2 patch features + ViT dynamics model. SR=0.90 on Push-T (DreamerV3: 0.04). Only tests DINOv2.
  - Insights:

- [ ] **Assran et al. 2025** — "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
  [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
  Frozen V-JEPA encoder + 300M action-conditioned predictor. 65-80% real robot success on 62 hours of training data.
  - Insights:

- [ ] **Apple Research 2025** — "SALT: Rethinking JEPA — Compute-Efficient Video SSL with Frozen Teachers"
  [arXiv:2509.24317](https://arxiv.org/abs/2509.24317)
  Frozen teacher + trainable student outperforms V-JEPA 2. Student quality robust to teacher quality.
  - Insights:

- [ ] **Rebuttal to Schneider — 2025** — "Pre-trained Visual Representations Generalize Where it Matters in Model-Based RL"
  [arXiv:2509.12531](https://arxiv.org/abs/2509.12531)
  PVRs DO help on truly hard OOD shifts. Partially fine-tuned DINOv2: 28% drop vs baseline CNN: 106% drop. Schneider only tested mild shifts.
  - Insights:

### Encoder benchmarks

- [ ] **Majumdar et al. NeurIPS 2023** — "VC-1: Where Are We in the Search for an Artificial Visual Cortex for Embodied Intelligence?"
  [arXiv:2303.18240](https://arxiv.org/abs/2303.18240) | [Project](https://eai-vc.github.io/)
  CortexBench: 17 embodied AI tasks. No single encoder dominates. MAE-trained models prioritize edges and boundaries.
  - Insights:

- [ ] **Theia — Shang et al. 2024** — "Distilling Diverse Vision Foundation Models for Robot Learning"
  [arXiv:2407.20179](https://arxiv.org/abs/2407.20179)
  Distills CLIP + DINOv2 + ViT into one encoder. High spatial token entropy correlates with robot learning performance (R=0.943).
  - Insights:

### Geometry and metrics

- [ ] **"Global Geometry Is Not Enough for Vision Representations" (2025)**
  [arXiv:2602.03282](https://arxiv.org/abs/2602.03282)
  Isotropy, uniformity, effective rank show near-zero correlation with compositional binding across 21 encoders. Functional sensitivity (Jacobian) works. Geometry experiments need this.
  - Insights:

- [ ] **Huang et al. ICLR 2024** — "LDReg: Local Dimensionality Regularized Self-Supervised Learning"
  [arXiv:2401.10474](https://arxiv.org/abs/2401.10474)
  Representations can look high-dimensional globally but collapse locally. Local intrinsic dimensionality matters for prediction.
  - Insights:

- [ ] **Jing et al. ICLR 2022** — "Understanding Dimensional Collapse in Contrastive Self-supervised Learning"
  [arXiv:2110.09348](https://arxiv.org/abs/2110.09348)
  Dimensional collapse happens in both contrastive AND non-contrastive SSL. Strong augmentation → more projector collapse but better encoder representations.
  - Insights:

### Error accumulation

- [ ] **"Learning with Imperfect Models" (2025)** — "When Multi-step Prediction Mitigates Compounding Error"
  [arXiv:2504.01766](https://arxiv.org/abs/2504.01766)
  Partial observability (frozen encoder missing state info) causes near-exponential error growth. Multi-step predictors mitigate. Directly relevant to frozen-encoder error analysis.
  - Insights:

- [ ] **Lambert et al. 2022** — "Investigating Compounding Prediction Errors in Learned Dynamics Models"
  [arXiv:2203.09637](https://arxiv.org/abs/2203.09637)
  Underlying dynamics matter more than model architecture for error shape. 25 pages, 19 figures.
  - Insights:

### Equivariance vs invariance

- [ ] **seq-JEPA 2025** — "Autoregressive Predictive Learning of Invariant-Equivariant World Models"
  [arXiv:2505.03176](https://arxiv.org/abs/2505.03176)
  Gets BOTH invariant (aggregate) and equivariant (encoder) features from one architecture. 84.12% STL-10 classification + 0.80 positional equivariance.
  - Insights:

- [ ] **Xie et al. CVPR 2022 Workshop** — "What Should Be Equivariant in Self-Supervised Learning"
  [PDF](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Xie_What_Should_Be_Equivariant_in_Self-Supervised_Learning_CVPRW_2022_paper.pdf)
  Pure invariance limits expressiveness. Preserving transformation order in embedding space helps.
  - Insights:

- [ ] **Wang et al. NeurIPS 2024** — "Understanding the Role of Equivariance in Self-supervised Learning"
  [arXiv:2411.06508](https://arxiv.org/abs/2411.06508)
  Information-theoretic: learning to predict augmentation parameters forces extraction of class-relevant features ("explaining-away effect").
  - Insights:

---

## Tier 3: Context (broader landscape)

### Decoupling representation learning

- [ ] **Stooke et al. ICML 2021** — "Decoupling Representation Learning from Reinforcement Learning"
  [arXiv:2009.08319](https://arxiv.org/abs/2009.08319)
  ATC-trained frozen encoder matches end-to-end RL. Foundational decoupling paper.
  - Insights:

- [ ] **Radosavovic et al. CoRL 2022** — "MVP: Real-World Robot Learning with Masked Visual Pre-training"
  [arXiv:2210.03109](https://arxiv.org/abs/2210.03109)
  Frozen MAE ViT (307M params). Outperforms CLIP by 75%, supervised by 81%. Evidence that MAE features work well for control.
  - Insights:

- [ ] **PIE-G NeurIPS 2022** — "Pre-Trained Image Encoder for Generalizable Visual RL"
  [arXiv:2212.08860](https://arxiv.org/abs/2212.08860)
  Frozen early-layer ResNet features. 55% generalization gain. Early layers > deep layers for transfer.
  - Insights:

### Video SSL

- [ ] **Bardes et al. TMLR 2024** — "V-JEPA: Latent Video Prediction for Visual Representation Learning"
  [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
  90% spatiotemporal masking, prediction in latent space. +4.2 over VideoMAE on frozen eval. Only method significantly above random on intuitive physics.
  - Insights:

- [ ] **Tong et al. NeurIPS 2022** — "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
  [arXiv:2203.12602](https://arxiv.org/abs/2203.12602)
  Tube masking at 90-95%. Pixel reconstruction. Strong fine-tuned but weaker frozen evaluation than V-JEPA.
  - Insights:

- [ ] **Baldassarre et al. 2025** — "DINO-world: Back to the Features — DINO as a Foundation for Video World Models"
  [arXiv:2507.19468](https://arxiv.org/abs/2507.19468)
  Frozen DINOv2 ViT-B + 40-layer transformer on 60M videos. Beats V-JEPA ViT-H on forecasting. Validates frozen image encoder for video.
  - Insights:

### World models (joint training baselines)

- [ ] **Hafner et al. 2023** — "DreamerV3: Mastering Diverse Domains through World Models"
  [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
  RSSM + discrete latents + KL balancing. Joint encoder-decoder training. Baseline for frozen-encoder comparison.
  - Insights:

- [ ] **NE-Dreamer 2026** — "Next Embedding Prediction Makes World Models Stronger"
  [arXiv:2603.02765](https://arxiv.org/abs/2603.02765)
  Decoder-free. Predicts next encoder embedding via Barlow Twins. Matches DreamerV3 without pixel reconstruction.
  - Insights:

- [ ] **Schwarzer et al. ICLR 2021** — "SPR: Data-Efficient RL with Self-Predictive Representations"
  [arXiv:2007.05929](https://arxiv.org/abs/2007.05929)
  Predicts own latent states multiple steps ahead with EMA target. 55% improvement over prior SOTA on Atari 100k.
  - Insights:

### Theory and foundations

- [ ] **LeCun 2022** — "A Path Towards Autonomous Machine Intelligence"
  [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf)
  JEPA manifesto. Hierarchical prediction in representation space for planning. Theoretical motivation for the entire project.
  - Insights:

- [ ] **Assran et al. CVPR 2023** — "I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
  [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) | [Code](https://github.com/facebookresearch/ijepa)
  Predicts masked patch representations from context. EMA target encoder. <1200 GPU hours for ViT-H/14.
  - Insights:

- [ ] **Jiang AISTATS 2024** — "A Note on Loss Functions and Error Compounding in Model-Based RL"
  [arXiv:2404.09946](https://arxiv.org/abs/2404.09946)
  Linear-in-horizon error accumulation is the best achievable in general. Important theoretical baseline.
  - Insights:

- [ ] **Gelada et al. ICML 2019** — "DeepMDP: Learning Continuous Latent Space Models for Representation Learning"
  [arXiv:1906.02736](https://arxiv.org/abs/1906.02736)
  L2 in DeepMDP representation upper-bounds bisimulation distance. Gold standard for "dynamically relevant" representation metric.
  - Insights:

### Surveys

- [ ] **Balestriero et al. 2023** — "A Cookbook of Self-Supervised Learning"
  [arXiv:2304.12210](https://arxiv.org/abs/2304.12210)
  Comprehensive SSL landscape reference.
  - Insights:

- [ ] **Shwartz-Ziv and LeCun JMLR** — "To Compress or Not to Compress — Self-Supervised Learning and Information Theory: A Review"
  [arXiv:2304.09355](https://arxiv.org/abs/2304.09355)
  Unified information-theoretic framework for all SSL methods.
  - Insights:

- [ ] **Kornblith et al. ICML 2019** — "Similarity of Neural Network Representations Revisited"
  [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
  CKA as standard representation comparison metric. Needed for comparing SSL representation structures.
  - Insights:

---

## Cross-Cutting Themes to Track

### Theme 1: Invariance vs equivariance
Contrastive methods learn invariance (discard transformation info), reconstruction methods learn equivariance (preserve it). For prediction/planning, equivariance may matter more. Track evidence for and against.

### Theme 2: The "objective mismatch" problem
SSL objectives optimize for compression/classification-relevant features, not dynamics-relevant features. Schneider found this causes failure. But MVP and DINO-WM show it can work. What determines when it works?

### Theme 3: Error accumulation and planning horizon
Frozen encoders create effective partial observability. Multi-step prediction may mitigate. Which SSL representations keep errors bounded longest?

### Theme 4: Local vs global representation structure
Global metrics (isotropy, effective rank) may not predict prediction quality. Local dimensionality and functional sensitivity matter. How do different SSL methods differ locally?

### Theme 5: Image encoders vs video encoders
Terver et al. found DINO > V-JEPA for manipulation (spatial precision matters). But V-JEPA wins on temporal reasoning. Task dependence is key.
