import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

class CausalRevIN(nn.Module):
    epsilon: float = 1e-5
    
    def __call__(self, x, inverse: bool = False, mean=None, var=None):
        # x: (Batch, T, D)
        if not inverse:
            # Compute causal mean and variance
            # cumsum
            t = jnp.arange(1, x.shape[1] + 1, dtype=x.dtype)
            t = t.reshape(1, -1, 1) # Broadcast over batch and D
            
            cum_sum = jnp.cumsum(x, axis=1)
            cum_sq_sum = jnp.cumsum(x**2, axis=1)
            
            mean = cum_sum / t
            # var = E[x^2] - (E[x])^2
            var = (cum_sq_sum / t) - mean**2
            var = jnp.maximum(var, 0.0)
            
            stdev = jnp.sqrt(var + self.epsilon)
            
            x_norm = (x - mean) / stdev
            return x_norm, mean, stdev
        else:
            # Inverse
            assert mean is not None and var is not None
            return x * var + mean # var here is passed as stdev

class RotaryEmbedding(nn.Module):
    dim: int
    
    def __call__(self, x):
        # x: (Batch, Seq, Heads, Dim)
        b, s, h, d = x.shape
        half_dim = d // 2
        
        # Frequencies
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
        
        # Positions
        pos = jnp.arange(s, dtype=jnp.float32)
        
        # Outer product
        freqs = jnp.einsum('i,j->ij', pos, inv_freq) # (Seq, HalfDim)
        
        # Repeat to Full Dim: [sin, sin, ..., cos, cos] or interleaved?
        # Standard RoPE: (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)
        # Using exp(i theta).
        
        freqs = jnp.concatenate([freqs, freqs], axis=-1) # (Seq, Dim)
        
        sin = jnp.sin(freqs)
        cos = jnp.cos(freqs)
        
        # Broadcast to (B, S, H, D)
        sin = sin[None, :, None, :]
        cos = cos[None, :, None, :]
        
        # Rotate
        # [-x2, x1, -x4, x3 ...]
        x_half1 = x[..., :half_dim]
        x_half2 = x[..., half_dim:]
        x_rotated = jnp.concatenate([-x_half2, x_half1], axis=-1)
        
        return (x * cos) + (x_rotated * sin)

class TransformerFilter(nn.Module):
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    patch_size: int = 1
    dim_y: int = 2
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (Batch, T, dim_y)
        b, t, d = x.shape
        
        # 1. Causal RevIN
        revin = CausalRevIN()
        x_norm, mean, stdev = revin(x)
        
        # 2. Patching
        # Pad if T not divisible by patch_size?
        # Assume T is divisible.
        p = self.patch_size
        assert t % p == 0, f"Sequence length {t} must be divisible by patch size {p}"
        
        num_patches = t // p
        x_patched = x_norm.reshape(b, num_patches, p * d)
        
        # 3. Projection
        h = nn.Dense(self.d_model)(x_patched)
        
        # 4. Transformer with RoPE
        rope = RotaryEmbedding(self.d_model // self.n_heads)
        
        # Mask for causality
        mask = nn.make_causal_mask(jnp.ones((b, num_patches)), dtype=bool)
        
        for _ in range(self.n_layers):
            # Attention
            # Norm
            h_norm = nn.LayerNorm()(h)
            
            # MultiHeadAttention
            # We need to inject RoPE into Q, K
            # Flax MHA doesn't expose Q, K easily for modification unless we subclass.
            # OR we compute RoPE outside?
            # Flax MHA allows `q_k_v_features`.
            
            # Let's use a manual attention block or look for a way to inject pos encoding.
            # Alternatively, apply RoPE to h before attention? No, RoPE is relative.
            # It must be on q, k.
            
            # Simple implementation of MHA block with RoPE
            # Q, K, V projections
            q = nn.Dense(self.d_model)(h_norm)
            k = nn.Dense(self.d_model)(h_norm)
            v = nn.Dense(self.d_model)(h_norm)
            
            # Reshape (B, T, H, D)
            q = q.reshape(b, num_patches, self.n_heads, self.d_model // self.n_heads)
            k = k.reshape(b, num_patches, self.n_heads, self.d_model // self.n_heads)
            v = v.reshape(b, num_patches, self.n_heads, self.d_model // self.n_heads)
            
            # RoPE
            q = rope(q)
            k = rope(k)
            
            # Attention
            logits = jnp.einsum('bthd,bshd->bhts', q, k)
            logits = logits / jnp.sqrt(q.shape[-1])
            
            # Mask
            # mask: (B, 1, T, T)
            # logits: (B, H, T, T)
            mask_bias = jnp.where(mask, 0., -1e9)
            logits += mask_bias
            
            weights = nn.softmax(logits, axis=-1)
            out = jnp.einsum('bhts,bshd->bthd', weights, v)
            out = out.reshape(b, num_patches, self.d_model)
            
            # Output proj
            out = nn.Dense(self.d_model)(out)
            
            h = h + out
            
            # FFN
            h_norm = nn.LayerNorm()(h)
            ff = nn.Dense(self.d_model * 4)(h_norm)
            ff = nn.gelu(ff)
            ff = nn.Dense(self.d_model)(ff)
            h = h + ff
            
        # 5. Output Head
        output = nn.Dense(p * d)(h)
        
        # 6. Unpatch
        output = output.reshape(b, t, d)
        
        # 7. Denormalize
        # We need to use the stats used for input.
        # But wait. If we predict y_{t+1}, we should use stats available at t?
        # The output at step `t` (patch `k`) is prediction for `patch k+1`?
        # Standard causal transformer: Output at pos `t` predicts `t+1`.
        # So `output` contains predictions.
        # We denormalize using the SAME stats `mean`, `stdev` because the network worked in normalized space.
        # The target is `x_norm` shifted.
        # Or we denormalize to get `y_hat` and compare with `y` shifted.
        
        y_hat = revin(output, inverse=True, mean=mean, var=stdev)
        
        return y_hat
