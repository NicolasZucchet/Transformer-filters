import jax
import jax.numpy as jnp
import numpy as np
import wandb
from functools import partial
from src.data import generate_sequences
from src.kalman import kalman_filter_final_state

def evaluate_model(model, params, A, C, sigma, KF_A, KF_C, KF_Q, seed, n_eval, batch_size, seq_len=128, warmup_len=64, patch_size=1, horizons=[2, 4, 8, 16, 32, 64], step=None):
    """
    Evaluates model using efficient autoregressive generation with caching.
    """
    print("Starting evaluation...")
    
    # Expand horizons to include all integers up to the max horizon
    if horizons:
        max_h = max(horizons)
        eval_horizons = list(range(1, max_h + 1))
    else:
        eval_horizons = []
    
    num_eval_batches = max(1, n_eval // batch_size)
    
    model_errors = {h: [] for h in eval_horizons}
    kf_errors = {h: [] for h in eval_horizons}
    
    # Kalman Filter Helpers
    get_final_state = jax.vmap(kalman_filter_final_state, in_axes=(0, None, None, None))
    
    # Calculate Cache Shapes
    # warmup_len is in timesteps.
    # We generate up to max horizon (64).
    # Total needed: warmup + 64.
    n_patches_warmup = warmup_len // patch_size
    n_patches_gen = 64 // patch_size
    max_patches = n_patches_warmup + n_patches_gen + 2 # Safety buffer
    
    head_dim = model.d_model // model.n_heads
    
    def init_cache_fn(bs):
        cache = []
        for _ in range(model.n_layers):
            k = jnp.zeros((bs, max_patches, model.n_heads, head_dim))
            v = jnp.zeros((bs, max_patches, model.n_heads, head_dim))
            idx = jnp.array(0, dtype=jnp.int32)
            cache.append((k, v, idx))
        return cache
    
    # 1. Prediction Function (Single Step with Cache)
    def predict_step(carry, _):
        # carry: (current_input_patch, cache, revin_state, pos_offset)
        # current_input_patch: (B, patch_size, dim_y)
        # pos_offset: scalar (integer)
        
        curr_patch, cache, revin_state, pos_offset = carry
        
        # Predict next patch
        y_hat, new_cache, new_revin_state = model.apply(
            {'params': params}, 
            curr_patch, 
            train=False, 
            cache=cache, 
            revin_state=revin_state, 
            pos_offset=pos_offset, 
            return_cache=True
        )
        
        # Autoregressive: output becomes next input
        next_input = y_hat
        new_pos_offset = pos_offset + 1
        
        new_carry = (next_input, new_cache, new_revin_state, new_pos_offset)
        
        return new_carry, y_hat

    # JIT the scan loop
    @jax.jit
    def generate_rollout(warmup_seq):
        # warmup_seq: (B, warmup_len, dim_y)
        bs = warmup_seq.shape[0]
        
        # 1. Init Cache
        init_cache = init_cache_fn(bs)
        
        # 2. Prefill (Process warmup)
        # We pass the full fixed-size cache. 
        # CausalAttention will update it from index 0.
        y_last_hat, cache, revin_state = model.apply(
            {'params': params},
            warmup_seq,
            train=False,
            cache=init_cache,
            revin_state=None,
            pos_offset=0,
            return_cache=True
        )
        
        # Input to first generation step is the prediction for the first future patch
        # y_last_hat has length warmup_len + patch_size
        first_input = y_last_hat[:, -patch_size:, :]
        
        # 3. Generate
        # We used n_patches_warmup + 1 patches (zero + warmup)
        start_pos_offset = n_patches_warmup + 1
        
        init_carry = (first_input, cache, revin_state, start_pos_offset)
        
        # Scan
        # We generate n_patches_gen - 1 additional patches
        # (Since first_input is already the first generated patch)
        patches_to_gen = n_patches_gen - 1
        
        if patches_to_gen > 0:
            _, gen_preds = jax.lax.scan(predict_step, init_carry, None, length=patches_to_gen)
            
            # gen_preds is (L, B, P, D) -> (B, L*P, D)
            gen_preds = jnp.transpose(gen_preds, (1, 0, 2, 3))
            gen_preds = gen_preds.reshape(gen_preds.shape[0], -1, gen_preds.shape[3])
            
            # Concatenate first_input and subsequent predictions
            all_preds = jnp.concatenate([first_input, gen_preds], axis=1)
        else:
            all_preds = first_input
        
        return all_preds

    key = jax.random.PRNGKey(seed)
    
    for i in range(num_eval_batches):
        if i % 5 == 0:
            print(f"Eval batch {i}/{num_eval_batches}")
            
        key, subkey = jax.random.split(key)
        # Use JIT-compiled data generator
        # Note: generate_sequences is decorated with @partial(jit, static_argnums=(1,2))
        xs_eval, ys_eval = generate_sequences(subkey, batch_size, seq_len, A, C, sigma)
        
        warmup = ys_eval[:, :warmup_len]
        truth = ys_eval[:, warmup_len:]
        
        # 1. Model Rollout
        all_preds = generate_rollout(warmup)
        # Limit to 64 steps if seq_len was larger or generation produced more?
        all_preds = all_preds[:, :64]
        
        # 2. Kalman Filter Rollout
        x_last = get_final_state(warmup, KF_A, KF_C, KF_Q)
        
        kf_preds_rollout = []
        x_curr = x_last
        
        for _ in range(64):
            x_curr = x_curr @ KF_A.T 
            y_pred = x_curr @ KF_C.T 
            kf_preds_rollout.append(y_pred[:, None, :])
            
        kf_rollout = jnp.concatenate(kf_preds_rollout, axis=1)
        
        # 3. Compute Metrics
        for h in eval_horizons:
            idx = h - 1 
            if idx < 64:
                m_err = jnp.mean(jnp.square(all_preds[:, idx] - truth[:, idx]), axis=-1)
                k_err = jnp.mean(jnp.square(kf_rollout[:, idx] - truth[:, idx]), axis=-1)
                
                model_errors[h].extend(m_err.tolist())
                kf_errors[h].extend(k_err.tolist())
                
    # Log Results
    log_dict = {}
    table_data = []
    columns = ["Horizon", "Score (Ratio)", "Model MSE", "KF MSE"]

    for h in eval_horizons:
        mean_model = np.mean(model_errors[h])
        mean_kf = np.mean(kf_errors[h])
        ratio = mean_model / mean_kf
        
        log_dict[f"eval/score_t+{h}"] = ratio
        print(f"t+{h}: Ratio={ratio:.4f} (Model={mean_model:.4f}, KF={mean_kf:.4f})")
        
        table_data.append([h, ratio, mean_model, mean_kf])
    
    if step is not None:
        log_dict["step"] = step
    
    # Create and log table
    table = wandb.Table(data=table_data, columns=columns)
    log_dict["eval/detailed_table"] = table
        
    wandb.log(log_dict)