"""Masked Autoencoder (MAE) for STL-10 (96x96 RGB)."""

import torch
import torch.nn as nn

from models.vit_components import PatchEmbed, TransformerBlock


class MAEEncoder(nn.Module):
    """ViT encoder that only processes visible (unmasked) patches during training.

    dim=384, depth=6 gives ~11M params, comparable to ResNet-18.
    """

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

    def forward_masked(self, x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """Encode only visible patches (training).

        Args:
            x: (B, 3, H, W) images
            ids_keep: (B, n_keep) indices of visible patches
        Returns:
            (B, n_keep, D) encoded visible tokens
        """
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        D = tokens.shape[-1]

        # Takes all the indices that are unmasked and leaves out the rest
        visible = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Run through the transformer blocks
        for block in self.blocks:
            visible = block(visible)

        return self.norm(visible)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all patches, average pool (evaluation). Returns (B, D)."""
        tokens = self.patch_embed(x) + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)
        
        # (B, N, D) -> (B, D) - averages all dimensions along the sequence axis
        return self.norm(tokens).mean(dim=1)


class MAEDecoder(nn.Module):
    def __init__(
        self,
        num_patches: int = 576,
        patch_size: int = 4,
        encoder_dim: int = 384,
        decoder_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size ** 2 * 3) # Maps from decoder output space to image space
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self, visible_tokens: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Decode full sequence from visible tokens + mask tokens.

        Args:
            visible_tokens: (B, n_keep, encoder_dim)
            ids_restore: (B, N) indices to unshuffle back to original order
        Returns:
            (B, N, patch_size^2 * 3) pixel predictions for all patches
        """
        visible = self.decoder_embed(visible_tokens)  # (B, n_keep, decoder_dim)
        D = visible.shape[-1]
        B, N = ids_restore.shape

        # Expand the mask token to match the number of masked patches + serve as a complement to input
        mask_tokens = self.mask_token.expand(B, N - visible.shape[1], -1)
        full = torch.cat([visible, mask_tokens], dim=1) # Add masked tokens to the end of the encoded input
        full = torch.gather(full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D)) # unshuffle to the original patch order

        # Add positional embeddings
        full = full + self.pos_embed

        # Run through the transformer nested blocks
        for block in self.blocks:
            full = block(full)

        # Return the normalized version + a map into the image space
        return self.pred(self.norm(full))


class MAE(nn.Module):
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 4,
        encoder_dim: int = 384,
        encoder_depth: int = 6,
        encoder_heads: int = 6,
        decoder_dim: int = 192,
        decoder_depth: int = 4,
        decoder_heads: int = 3,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2

        self.encoder = MAEEncoder(
            img_size, patch_size, encoder_dim, encoder_depth, encoder_heads,
        )
        self.latent_dim = encoder_dim
        self.decoder = MAEDecoder(
            self.num_patches, patch_size, encoder_dim, decoder_dim, decoder_depth, decoder_heads,
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, patch_size^2 * 3)"""
        p = self.patch_size
        B, C, H, W = x.shape
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, p * p * C)

    def random_mask(
        self, B: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random mask.

        Returns:
            ids_keep: (B, n_keep) indices of visible patches - each batch conttains unique integers with values from 0 to N-1
            ids_restore: (B, N) indices to unshuffle - each batch conttains unique integers with values from 0 to N-1, this is a map from the shuffled order to the original order
            mask: (B, N) bool — True = masked (removed)
        """
        N = self.num_patches
        n_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=device) # Draw random noise
        ids_shuffle = noise.argsort(dim=1) # Sort along the sequence axis and return indices of elements 
        ids_restore = ids_shuffle.argsort(dim=1) # This is the inverse permutation of ids_shuffle - returns map back to original order

        ids_keep = ids_shuffle[:, :n_keep] # Take first n_keep indices for each batch in the shuffled ids to keep them (indices to keeep)

        # Make a mask of False for the kept indices and True for the rest
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_keep, False)

        return ids_keep, ids_restore, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """For downstream eval: encode all patches, average pool -> (B, D)."""
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward.

        Returns:
            loss: MSE on masked patches only
            pred: (B, N, patch_pixels) predictions for all patches
            mask: (B, N) bool — True = masked
        """
        ids_keep, ids_restore, mask = self.random_mask(x.shape[0], x.device)

        visible = self.encoder.forward_masked(x, ids_keep)
        pred = self.decoder(visible, ids_restore)

        target = self.patchify(x)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # per-patch MSE: (B, N)
        loss = (loss * mask).sum() / mask.sum()  # average over masked patches only

        return loss, pred, mask


if __name__ == "__main__":
    model = MAE()
    x = torch.randn(4, 3, 96, 96)

    # Training forward
    loss, pred, mask = model(x)
    print(f"x:             {tuple(x.shape)}")
    print(f"pred:          {tuple(pred.shape)}")
    print(f"mask:          {tuple(mask.shape)}  (True=masked)")
    n_masked = mask.sum().item()
    n_total = mask.numel()
    print(f"mask ratio:    {n_masked}/{n_total} = {n_masked/n_total:.2f}")
    print(f"loss:          {loss.item():.4f}")
    assert pred.shape == (4, 576, 4 * 4 * 3)
    assert mask.shape == (4, 576)
    assert abs(n_masked / n_total - 0.75) < 0.01
    assert loss.dim() == 0

    # Encode (eval — all patches, average pooled)
    h = model.encode(x)
    print(f"h  (encode):   {tuple(h.shape)}")
    assert h.shape == (4, model.latent_dim)

    # Patchify shape check
    patches = model.patchify(x)
    assert patches.shape == (4, 576, 48)

    # Parameter counts
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"\nEncoder dim:   {model.latent_dim}")
    print(f"Patch size:    {model.patch_size}x{model.patch_size}")
    print(f"Num patches:   {model.num_patches}")
    print(f"Total params:  {n_all:>10,}")
    print(f"  Encoder:     {n_enc:>10,}")
    print(f"  Decoder:     {n_dec:>10,}")

    print("\nAll checks passed.")
