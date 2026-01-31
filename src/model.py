import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Any, Tuple, List

class CausalRevIN(nn.Module):
    epsilon: float = 1e-5
    
    def __call__(self, x, inverse: bool = False, mean=None, var=None, return_state: bool = False):
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
            
            if return_state:
                # State for incremental updates after this sequence
                # (count, sum_x, sum_sq_x) at the END of the sequence
                # These must be (Batch, 1, D)
                count = jnp.full((x.shape[0], 1, 1), x.shape[1], dtype=x.dtype)
                sum_x = cum_sum[:, -1:, :]
                sum_sq_x = cum_sq_sum[:, -1:, :]
                state = (count, sum_x, sum_sq_x)
                return x_norm, mean, stdev, state
            
            return x_norm, mean, stdev
        else:
            # Inverse
            assert mean is not None and var is not None
            return x * var + mean # var here is passed as stdev

    def forward_incremental(self, x_t, state):
        # x_t: (Batch, P, D)
        # state: (count, sum_x, sum_sq_x) each (Batch, 1, D)
        # Returns: x_norm_t (Batch, P, D), new_state (Batch, 1, D), mean_t (Batch, P, D), stdev_t (Batch, P, D)
        
        count, sum_x, sum_sq_x = state
        
        # We need to compute running statistics for each token in the patch
        # Cumulative sums over the patch dimension (axis 1)
        # x_t shape: (B, P, D)
        
        patch_size = x_t.shape[1]
        
        # Incremental counts: count + 1, count + 2, ..., count + P
        # count is (B, 1, 1). arange is (P).
        counts_incr = jnp.arange(1, patch_size + 1, dtype=x_t.dtype).reshape(1, -1, 1)
        current_counts = count + counts_incr # (B, P, 1) broadcast
        
        # Cumulative sums
        cumsum_x = jnp.cumsum(x_t, axis=1) # (B, P, D)
        cumsum_sq_x = jnp.cumsum(x_t**2, axis=1) # (B, P, D)
        
        current_sum_x = sum_x + cumsum_x
        current_sum_sq_x = sum_sq_x + cumsum_sq_x
        
        # Compute means/vars for normalization
        # Note: RevIN usually normalizes using stats up to t-1 to predict t?
        # Standard implementation (Kim et al.) normalizes the input x_t using stats computed from x_t (and history).
        # So we normalize x_t[i] using stats including x_t[i].
        
        mean = current_sum_x / current_counts
        var = (current_sum_sq_x / current_counts) - mean**2
        var = jnp.maximum(var, 0.0)
        stdev = jnp.sqrt(var + self.epsilon)
        
        x_norm = (x_t - mean) / stdev
        
        # Update state to the final values of this patch
        # Slice to keep (B, 1, D) dimensions
        new_count = current_counts[:, -1:, :]
        new_sum_x = current_sum_x[:, -1:, :]
        new_sum_sq_x = current_sum_sq_x[:, -1:, :]
        
        return x_norm, (new_count, new_sum_x, new_sum_sq_x), mean, stdev
        
    def init_state(self, batch_size, dim, dtype=jnp.float32):
        return (
            jnp.zeros((batch_size, 1, 1), dtype=dtype), # count
            jnp.zeros((batch_size, 1, dim), dtype=dtype), # sum_x
            jnp.zeros((batch_size, 1, dim), dtype=dtype)  # sum_sq_x
        )

class RotaryEmbedding(nn.Module):
    dim: int
    
    def __call__(self, x, offset=0):
        # x: (Batch, Seq, Heads, Dim)
        b, s, h, d = x.shape
        half_dim = d // 2
        
        # Frequencies
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
        
        # Positions
        pos = jnp.arange(s, dtype=jnp.float32) + offset
        
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

class CausalAttention(nn.Module):
    d_model: int
    n_heads: int
    
    @nn.compact
    def __call__(self, x, mask=None, cache=None, pos_offset=0, return_cache=False):
        # x: (B, T, D)
        b, t, d = x.shape
        head_dim = self.d_model // self.n_heads
        
        q = nn.Dense(self.d_model)(x)
        k = nn.Dense(self.d_model)(x)
        v = nn.Dense(self.d_model)(x)
        
        q = q.reshape(b, t, self.n_heads, head_dim)
        k = k.reshape(b, t, self.n_heads, head_dim)
        v = v.reshape(b, t, self.n_heads, head_dim)
        
        # RoPE
        rope = RotaryEmbedding(head_dim)
        q = rope(q, offset=pos_offset)
        k = rope(k, offset=pos_offset)
        
        # Cache logic
        if cache is not None:
            # Fixed size cache: (k_buffer, v_buffer, index)
            k_buffer, v_buffer, index = cache
            # k_buffer: (B, MaxT, H, D)
            
            # Update cache at index
            # x is (B, T, D) -> k is (B, T, H, D)
            # Update slice starts at (0, index, 0, 0)
            # We assume T=1 for decode usually, or T>1 for prefill?
            # dynamic_update_slice works for T>1 too.
            
            k_buffer = jax.lax.dynamic_update_slice(k_buffer, k, (0, index, 0, 0))
            v_buffer = jax.lax.dynamic_update_slice(v_buffer, v, (0, index, 0, 0))
            
            new_index = index + t
            new_cache = (k_buffer, v_buffer, new_index)
            
            # Attention
            # Attend to [0, new_index)
            # k_buffer has valid data up to new_index.
            # We use the whole buffer as key/value.
            
            # Mask creation:
            # We want mask[b, 1, t_q, t_k]
            # t_q = T (current chunk)
            # t_k = MaxT (buffer size)
            # Valid if t_k < index + t_q?
            # Standard causal: q at pos (index + i) attends to k at pos j if j <= index + i.
            
            max_len = k_buffer.shape[1]
            # range for k: [0, max_len)
            idx_k = jnp.arange(max_len)
            # range for q: [index, index + T)
            # Use jnp.arange(t) + index to avoid dynamic start in arange
            idx_q = (jnp.arange(t) + index)[:, None] # (T, 1)
            
            # mask: 1 if idx_k <= idx_q else 0
            # Broadcast to (1, 1, T, MaxT)
            # And also mask out anything >= new_index (unwritten future)
            # Actually idx_k <= idx_q handles idx_k < new_index implicitly because idx_q < new_index.
            
            causal_mask = (idx_k[None, :] <= idx_q) # (T, MaxT)
            causal_mask = causal_mask[None, None, :, :] # (1, 1, T, MaxT)
            
            # Combine with provided mask if any (though usually None for decode)
            if mask is not None:
                # mask is (B, 1, T, T) usually? Or (B, 1, T, S)?
                # If mask provided during decode, it might be tricky.
                # Assuming no extra mask for now or mask matches shapes.
                pass
                
            mask_bias = jnp.where(causal_mask, 0., -1e9)
            
            k_in = k_buffer
            v_in = v_buffer
            
        elif return_cache:
             # Just return what we computed, but we can't really "return cache" without a buffer.
             # This path is problematic for fixed size.
             # We expect the caller to PROVIDE the buffer if they want caching.
             new_cache = None
             k_in = k
             v_in = v
             mask_bias = mask # Use standard mask passed in
        else:
            new_cache = None
            k_in = k
            v_in = v
            mask_bias = mask
            
        # Attention
        logits = jnp.einsum('bthd,bshd->bhts', q, k_in)
        logits = logits / jnp.sqrt(head_dim)
        
        if mask_bias is not None:
            logits += mask_bias
            
        weights = nn.softmax(logits, axis=-1)
        out = jnp.einsum('bhts,bshd->bthd', weights, v_in)
        out = out.reshape(b, t, self.d_model)
        
        out = nn.Dense(self.d_model)(out)
        
        return out, new_cache

class TransformerFilter(nn.Module):
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    patch_size: int = 1
    dim_y: int = 2
    remove_inverse_norm: bool = False
    
    @nn.compact
    def __call__(self, x, train: bool = True, cache=None, revin_state=None, pos_offset=0, return_cache=False):
        # x: (Batch, T, dim_y)
        b, t, d = x.shape
        revin = CausalRevIN()
        
        # Determine if we are returning cache (requires cache to be provided for fixed size)
        # Actually, for "init", maybe we can't easily do it inside unless we allocate.
        # We'll assume cache is provided if return_cache is True, OR we just handle the logic cleanly.
        
        # 1. Causal RevIN
        if revin_state is None:
            # Full sequence mode (Prefill or Train)
            if return_cache:
                x_norm, mean, stdev, new_revin_state = revin(x, return_state=True)
            else:
                x_norm, mean, stdev = revin(x)
                new_revin_state = None
        else:
            # Incremental mode (Decode)
            # x is (B, 1, D)
            x_norm, new_revin_state, mean, stdev = revin.forward_incremental(x, revin_state)
        
        # 2. Patching
        p = self.patch_size
        if revin_state is None:
            assert t % p == 0, f"Sequence length {t} must be divisible by patch size {p}"
            num_patches = t // p
            x_patched = x_norm.reshape(b, num_patches, p * d)
        else:
            # Incremental: x_norm is (B, p, D) 
            x_patched = x_norm.reshape(b, 1, p * d)
            num_patches = 1
        
        # 3. Projection
        h = nn.Dense(self.d_model)(x_patched)
        
        # 4. Transformer
        # Mask creation
        if revin_state is None and cache is None:
            # Full causal mask (Training / No Cache)
            mask = nn.make_causal_mask(jnp.ones((b, num_patches)), dtype=bool)
            mask_bias = jnp.where(mask, 0., -1e9)
        else:
            # If cache is provided (Prefill or Decode with fixed cache), CausalAttention handles masking.
            mask_bias = None
        
        new_caches = []
        
        for i in range(self.n_layers):
            h_norm = nn.LayerNorm()(h)
            
            # Get cache for this layer
            layer_cache = cache[i] if cache is not None else None
            
            attn = CausalAttention(self.d_model, self.n_heads)
            h_attn, new_layer_cache = attn(h_norm, mask=mask_bias, cache=layer_cache, pos_offset=pos_offset, return_cache=return_cache)
            
            if new_layer_cache is not None:
                new_caches.append(new_layer_cache)
                
            h = h + h_attn
            
            # FFN
            h_norm = nn.LayerNorm()(h)
            ff = nn.Dense(self.d_model * 4)(h_norm)
            ff = nn.gelu(ff)
            ff = nn.Dense(self.d_model)(ff)
            h = h + ff
            
        # 5. Output Head
        output = nn.Dense(p * d)(h)
        
        # 6. Unpatch
        output = output.reshape(b, -1, d)
        
        # 7. Denormalize
        if not self.remove_inverse_norm:
            y_hat = revin(output, inverse=True, mean=mean, var=stdev)
        else:
            y_hat = output
        
        if return_cache:
            return y_hat, new_caches, new_revin_state
        else:
            return y_hat
            
    def init_cache(self, batch_size, seq_len):
        # This is hard because we don't know the full length upfront for pre-allocation if we were using static arrays.
        # But JAX handles dynamic shapes in lists/tuples okay-ish or we just start with empty.
        # Ideally, we start with None or empty arrays.
        # But for JIT, shapes must be known.
        # Let's return a structure of zeros.
        
        # Actually, standard way is to start with initialized dummy cache.
        # Or simpler: The first call to generate creates the cache.
        pass


