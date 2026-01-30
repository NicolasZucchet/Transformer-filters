import argparse
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import wandb
import numpy as np
import time

from src.data import generate_system_parameters, generate_sequences
from src.model import TransformerFilter
from src.kalman import kalman_filter, kalman_filter_final_state

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_val", type=float, default=0.9)
    parser.add_argument("--diagonal_A", type=int, default=0) # 0 or 1
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--dim_x", type=int, default=64)
    parser.add_argument("--dim_y", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_eval", type=int, default=8192)
    parser.add_argument("--wandb_project", type=str, default="Transformer-filters")
    return parser.parse_args()

def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, config=vars(args))
    
    key = jax.random.PRNGKey(args.seed)
    
    # 1. Generate System
    key, subkey = jax.random.split(key)
    A, C = generate_system_parameters(subkey, args.dim_x, args.dim_y, args.lambda_val, bool(args.diagonal_A))
    
    # 2. Estimate Kalman Filter Baseline Variance
    key, subkey = jax.random.split(key)
    val_T = 1024
    val_bs = 128
    val_xs, val_ys = generate_sequences(subkey, val_bs, val_T, A, C, args.sigma)
    
    # Prepare KF args
    if jnp.iscomplexobj(A):
        KF_A = jnp.block([[A.real, -A.imag], [A.imag, A.real]])
        KF_C = jnp.block([[C.real, -C.imag]])
        KF_Q = args.sigma / jnp.sqrt(2)
    else:
        KF_A = A
        KF_C = C
        KF_Q = args.sigma
        
    kf_filter_vmap = jax.jit(jax.vmap(kalman_filter, in_axes=(0, None, None, None)))
    kf_preds, kf_stats = kf_filter_vmap(val_ys, KF_A, KF_C, KF_Q)
    kf_err = val_ys - kf_preds
    kf_mse = jnp.mean(jnp.square(kf_err)[:, 64:])
    print(f"Baseline KF MSE: {kf_mse}")
    wandb.log({"baseline_kf_mse": float(kf_mse)})
    
    # 3. Model Setup
    model = TransformerFilter(patch_size=args.patch_size, dim_y=args.dim_y)
    
    dummy_input = jnp.zeros((1, args.seq_len, args.dim_y))
    key, subkey = jax.random.split(key)
    variables = model.init(subkey, dummy_input)
    
    tx = optax.adamw(learning_rate=args.lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    # 4. Training Loop
    @jax.jit
    def train_step(state, batch_y):
        def loss_fn(params):
            preds = model.apply({'params': params}, batch_y)
            preds_aligned = preds[:, :-1]
            targets_aligned = batch_y[:, 1:]
            
            loss = jnp.mean(jnp.square(preds_aligned - targets_aligned))
            return loss / kf_mse
            
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    print("Starting training...")
    start_time = time.time()
    for step in range(args.steps):
        key, subkey = jax.random.split(key)
        xs, ys = generate_sequences(subkey, args.batch_size, args.seq_len, A, C, args.sigma)
        
        state, loss = train_step(state, ys)
        
        if step % 50 == 0:
            wandb.log({"train_loss": float(loss), "step": step})
            
    print(f"Training finished in {time.time() - start_time:.2f}s")

    # 5. Evaluation (Rollout)
    print("Starting evaluation...")
    n_eval = args.n_eval
    eval_batch_size = 256
    n_chunks = max(1, n_eval // eval_batch_size)
    
    horizons = [2, 4, 8, 16, 32, 64]
    model_errors = {h: [] for h in horizons}
    kf_errors = {h: [] for h in horizons}
    
    get_final_state = jax.jit(jax.vmap(kalman_filter_final_state, in_axes=(0, None, None, None)))
    
    @jax.jit
    def predict_next_patch(params, seq):
        preds = model.apply({'params': params}, seq, train=False)
        return preds[:, -args.patch_size:]
    
    for i in range(n_chunks):
        if i % 5 == 0:
            print(f"Eval chunk {i}/{n_chunks}")
        key, subkey = jax.random.split(key)
        xs_eval, ys_eval = generate_sequences(subkey, eval_batch_size, 128, A, C, args.sigma)
        
        warmup = ys_eval[:, :64]
        truth = ys_eval[:, 64:]
        
        x_last = get_final_state(warmup, KF_A, KF_C, KF_Q)
        
        kf_preds_rollout = []
        x_curr = x_last
        
        for _ in range(64):
            x_curr = x_curr @ KF_A.T 
            y_pred = x_curr @ KF_C.T 
            kf_preds_rollout.append(y_pred[:, None, :])
            
        kf_rollout = jnp.concatenate(kf_preds_rollout, axis=1)
        
        current_seq = warmup
        predictions = []
        steps_generated = 0
        
        while steps_generated < 64:
            next_patch = predict_next_patch(state.params, current_seq)
            predictions.append(next_patch)
            current_seq = jnp.concatenate([current_seq, next_patch], axis=1)
            steps_generated += args.patch_size
            
        all_preds = jnp.concatenate(predictions, axis=1)
        all_preds = all_preds[:, :64]
        
        for h in horizons:
            idx = h - 1 
            if idx < 64:
                m_err = jnp.mean(jnp.square(all_preds[:, idx] - truth[:, idx]), axis=-1)
                k_err = jnp.mean(jnp.square(kf_rollout[:, idx] - truth[:, idx]), axis=-1)
                
                model_errors[h].extend(m_err.tolist())
                kf_errors[h].extend(k_err.tolist())
                
    for h in horizons:
        mean_model = np.mean(model_errors[h])
        mean_kf = np.mean(kf_errors[h])
        ratio = mean_model / mean_kf
        
        wandb.log({f"eval/score_t+{h}": ratio})
        print(f"t+{h}: Ratio={ratio:.4f} (Model={mean_model:.4f}, KF={mean_kf:.4f})")
        
    wandb.finish()

if __name__ == "__main__":
    main()
