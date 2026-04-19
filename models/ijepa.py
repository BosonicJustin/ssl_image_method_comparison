"""I-JEPA (Image-based Joint-Embedding Predictive Architecture) for STL-10 (96x96 RGB).

Predicts representations of masked target blocks using context patches.
Combines MAE-style masking with BYOL-style EMA target encoder.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_components import PatchEmbed, TransformerBlock


class IJEPAEncoder(nn.Module):
    """ViT encoder with full-sequence and masked forward modes."""

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 4,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.num_features = embed_dim
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all patches, return full sequence. (B, N, D)"""
        tokens = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward_masked(self, x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """Encode only selected patches. (B, n_keep, D)"""
        tokens = self.patch_embed(x) + self.pos_embed
        D = tokens.shape[-1]
        visible = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        for block in self.blocks:
            visible = block(visible)
        return self.norm(visible)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eval: encode all patches, average pool -> (B, D)."""
        return self.forward_full(x).mean(dim=1)


class Predictor(nn.Module):
    """Small transformer that predicts target representations from context."""

    def __init__(
        self,
        num_patches: int = 576,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
        ids_target: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target representations from context.

        Args:
            context_tokens: (B, n_context, D) from context encoder
            ids_restore: (B, N) unshuffle [context, target] -> original order
            ids_target: (B, n_target) indices of target patches
        Returns:
            (B, n_target, D) predicted representations
        """
        B, N = ids_restore.shape
        D = context_tokens.shape[-1]

        # Build full sequence: context tokens + mask tokens at target positions
        n_mask = N - context_tokens.shape[1]
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        full = torch.cat([context_tokens, mask_tokens], dim=1)

        # Unshuffle to original spatial order
        full = torch.gather(full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        full = full + self.pos_embed

        for block in self.blocks:
            full = block(full)

        full = self.norm(full)

        # Extract predictions at target positions only
        return torch.gather(full, dim=1, index=ids_target.unsqueeze(-1).expand(-1, -1, D))


class IJEPA(nn.Module):
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 4,
        embed_dim: int = 384,
        encoder_depth: int = 6,
        encoder_heads: int = 6,
        pred_depth: int = 4,
        pred_heads: int = 6,
        momentum: float = 0.996,
        num_target_blocks: int = 4,
        target_block_min: int = 4,
        target_block_max: int = 8,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        self.num_target_blocks = num_target_blocks
        self.target_block_min = target_block_min
        self.target_block_max = target_block_max

        # Context encoder
        self.encoder = IJEPAEncoder(img_size, patch_size, embed_dim, encoder_depth, encoder_heads)
        self.latent_dim = embed_dim

        # Target encoder (EMA of context encoder, no gradients)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = Predictor(self.num_patches, embed_dim, pred_depth, pred_heads)

        self.momentum = momentum

    @torch.no_grad()
    def update_target(self):
        """EMA update: target = m * target + (1 - m) * context."""
        m = self.momentum
        for p_o, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)

    def generate_masks(
        self, B: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample rectangular target blocks (same mask for all batch elements).

        Returns:
            ids_context: (B, n_context)
            ids_target: (B, n_target)
            ids_restore: (B, N) unshuffle [context, target] -> original order
        """
        g = self.grid_size
        target_mask = torch.zeros(g, g, dtype=torch.bool, device=device)

        for _ in range(self.num_target_blocks):
            bh = torch.randint(self.target_block_min, self.target_block_max + 1, (1,)).item()
            bw = torch.randint(self.target_block_min, self.target_block_max + 1, (1,)).item()
            top = torch.randint(0, g - bh + 1, (1,)).item()
            left = torch.randint(0, g - bw + 1, (1,)).item()
            target_mask[top:top + bh, left:left + bw] = True

        target_mask = target_mask.reshape(self.num_patches)
        ids_target = target_mask.nonzero(as_tuple=False).squeeze(-1)
        ids_context = (~target_mask).nonzero(as_tuple=False).squeeze(-1)

        # Restore: [context, target] order -> original order
        ids_shuffle = torch.cat([ids_context, ids_target])
        ids_restore = ids_shuffle.argsort()

        return (
            ids_context.unsqueeze(0).expand(B, -1),
            ids_target.unsqueeze(0).expand(B, -1),
            ids_restore.unsqueeze(0).expand(B, -1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """For downstream eval: encode all patches, average pool -> (B, D)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward. Returns scalar loss."""
        B = x.shape[0]
        ids_context, ids_target, ids_restore = self.generate_masks(B, x.device)

        # Context encoder: encode context patches only
        context_tokens = self.encoder.forward_masked(x, ids_context)

        # Target encoder: encode ALL patches, extract at target positions
        with torch.no_grad():
            target_full = self.target_encoder.forward_full(x)
            D = target_full.shape[-1]
            target_reps = torch.gather(
                target_full, dim=1, index=ids_target.unsqueeze(-1).expand(-1, -1, D)
            )
            target_reps = F.normalize(target_reps, dim=-1)

        # Predict target representations from context
        pred_reps = self.predictor(context_tokens, ids_restore, ids_target)
        pred_reps = F.normalize(pred_reps, dim=-1)

        # MSE per patch (sum over feature dim, mean over batch and patches)
        return ((pred_reps - target_reps) ** 2).sum(dim=-1).mean()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    model = IJEPA()
    x = torch.randn(4, 3, 96, 96)

    # Training forward
    loss = model(x)
    print(f"x:             {tuple(x.shape)}")
    print(f"loss:          {loss.item():.4f}")
    assert loss.dim() == 0
    assert loss.item() >= 0

    # Encode (eval)
    h = model.encode(x)
    print(f"h  (encode):   {tuple(h.shape)}")
    assert h.shape == (4, model.latent_dim)

    # EMA update
    old_param = next(model.target_encoder.parameters()).clone()
    model.update_target()
    new_param = next(model.target_encoder.parameters())
    assert not torch.equal(old_param, new_param)
    print(f"EMA update:    OK (momentum={model.momentum})")

    # Mask generation
    ids_ctx, ids_tgt, ids_rst = model.generate_masks(4, torch.device("cpu"))
    n_ctx, n_tgt = ids_ctx.shape[1], ids_tgt.shape[1]
    print(f"context:       {n_ctx} patches")
    print(f"target:        {n_tgt} patches")
    print(f"total:         {n_ctx + n_tgt} / {model.num_patches}")
    assert n_ctx + n_tgt == model.num_patches

    # Parameter counts
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_tgt_enc = sum(p.numel() for p in model.target_encoder.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    print(f"\nEncoder dim:   {model.latent_dim}")
    print(f"Grid size:     {model.grid_size}x{model.grid_size}")
    print(f"Encoder params:{n_enc:>10,}")
    print(f"Target params: {n_tgt_enc:>10,} (EMA copy, no grad)")
    print(f"Predictor:     {n_pred:>10,}")

    print("\nAll checks passed.")
