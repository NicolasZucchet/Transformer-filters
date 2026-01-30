import jax
import jax.numpy as jnp
from jax import random
from functools import partial

def generate_system_parameters(key, dim_x, dim_y, lambda_val, structure='dense'):
    """
    Generates system parameters A and C.
    
    Args:
        structure: 'dense', 'diagonal', or 'scalar'.
            - 'dense': Random real matrix with complex conjugate eigenvalues (oscillatory).
            - 'diagonal': Real diagonal matrix with real eigenvalues (exponential).
            - 'scalar': Real scalar matrix A = s * I (exponential).
    """
    k1, k2, k3, k4 = random.split(key, 4)
    
    if structure == 'scalar':
        # A = s * I
        # Sample s from [-lambda, -lambda/2] U [lambda/2, lambda]
        sign = random.choice(k1, jnp.array([-1.0, 1.0]))
        mag = random.uniform(k2, minval=lambda_val/2, maxval=lambda_val)
        s = sign * mag
        A = s * jnp.eye(dim_x)
        
    elif structure == 'diagonal':
        # A = diag(s_i)
        # Sample s_i from [-lambda, -lambda/2] U [lambda/2, lambda]
        signs = random.choice(k1, jnp.array([-1.0, 1.0]), shape=(dim_x,))
        mags = random.uniform(k2, shape=(dim_x,), minval=lambda_val/2, maxval=lambda_val)
        A = jnp.diag(signs * mags)
        
    elif structure == 'dense':
        # Dense Real A with Complex Eigenvalues (Rotations)
        # Eigenvalues in conjugate pairs
        n_pairs = dim_x // 2
        radii = random.uniform(k1, (n_pairs,), minval=lambda_val/2, maxval=lambda_val)
        angles = random.uniform(k2, (n_pairs,), minval=0, maxval=jnp.pi) 
        
        blocks = []
        for r, th in zip(radii, angles):
            # Block for a pair of complex conjugate eigenvalues
            # [ r cos t, -r sin t ]
            # [ r sin t,  r cos t ]
            block = jnp.array([[r * jnp.cos(th), -r * jnp.sin(th)],
                               [r * jnp.sin(th),  r * jnp.cos(th)]])
            blocks.append(block)
        
        # Handle odd dimension if necessary
        if dim_x % 2 != 0:
            # Just add a real eigenvalue
            r = random.uniform(k3, minval=lambda_val/2, maxval=lambda_val)
            s = random.choice(k4, jnp.array([-1.0, 1.0]))
            blocks.append(jnp.array([[s * r]]))
            
        J = jax.scipy.linalg.block_diag(*blocks)
        
        # Random similarity transform
        X = random.normal(k3, (dim_x, dim_x))
        Q, _ = jnp.linalg.qr(X)
        A = Q @ J @ Q.T
    
    else:
        raise ValueError(f"Unknown structure: {structure}")

    # C is random real projection
    C = random.normal(k4, (dim_y, dim_x)) / jnp.sqrt(dim_x)
    
    return A, C

@partial(jax.jit, static_argnums=(1, 2))
def generate_sequences(key, batch_size, T, A, C, noise_std):
    # A, C are real
    dim_x = A.shape[0]
    dtype = A.dtype
    
    def step(x_t, k):
        # Noise
        if jnp.iscomplexobj(A):
             # Should not happen for currently supported structures, but kept for robustness
             eps = (random.normal(k, x_t.shape) + 1j * random.normal(k, x_t.shape)) * (noise_std / jnp.sqrt(2))
        else:
             eps = random.normal(k, x_t.shape) * noise_std
             
        x_next = A @ x_t + eps
        
        if jnp.iscomplexobj(C):
            y_next = (C @ x_next).real
        else:
            y_next = C @ x_next
            
        return x_next, (x_next, y_next)
    
    keys = random.split(key, batch_size)
    
    def simulate_one(k):
        ks = random.split(k, T)
        x0 = jnp.zeros(dim_x, dtype=dtype)
        
        _, (xs, ys) = jax.lax.scan(step, x0, ks)
        return xs, ys
        
    xs, ys = jax.vmap(simulate_one)(keys)
    return xs, ys