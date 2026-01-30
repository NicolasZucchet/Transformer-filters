import jax
import jax.numpy as jnp
from jax import random

def generate_system_parameters(key, dim_x, dim_y, lambda_val, diagonal=False):
    k1, k2, k3, k4 = random.split(key, 4)
    
    # Generate eigenvalues with modulus in [lambda/2, lambda]
    # We want real output y, so eigenvalues must come in conjugate pairs (or be real).
    # dim_x is typically 64 (even).
    
    n_pairs = dim_x // 2
    radii = random.uniform(k1, (n_pairs,), minval=lambda_val/2, maxval=lambda_val)
    angles = random.uniform(k2, (n_pairs,), minval=0, maxval=jnp.pi) 
    
    # Construct eigenvalues
    eigs = radii * jnp.exp(1j * angles)
    eigs_conj = jnp.conj(eigs)
    all_eigs = jnp.concatenate([eigs, eigs_conj])
    
    # If diagonal, A is diagonal complex matrix
    if diagonal:
        A = jnp.diag(all_eigs)
        
        # C must be complex such that y = Re(C x) behaves like a projection
        # Or y = C x is real?
        # If x evolves as x_{t+1} = A x_t + eps (complex), and we want y real,
        # we can define y = Re(C x).
        # To make it fair comparison with Dense case, we assume the system structure is just diagonalized.
        # We generate a random C complex.
        C = (random.normal(k3, (dim_y, dim_x)) + 1j * random.normal(k4, (dim_y, dim_x))) / jnp.sqrt(2 * dim_x)
        
    else:
        # Dense Real A
        # Build Real Jordan form first
        blocks = []
        for r, th in zip(radii, angles):
            # Block for a pair of complex conjugate eigenvalues
            # [ r cos t, -r sin t ]
            # [ r sin t,  r cos t ]
            block = jnp.array([[r * jnp.cos(th), -r * jnp.sin(th)],
                               [r * jnp.sin(th),  r * jnp.cos(th)]])
            blocks.append(block)
        
        if dim_x % 2 != 0:
            # Handle odd dimension if necessary (though default is 64)
            # Just add a real eigenvalue
            pass 
            
        J = jax.scipy.linalg.block_diag(*blocks)
        
        # Random similarity transform
        # Generate random orthogonal matrix Q
        X = random.normal(k3, (dim_x, dim_x))
        Q, _ = jnp.linalg.qr(X)
        
        A = Q @ J @ Q.T
        
        # C real
        C = random.normal(k4, (dim_y, dim_x)) / jnp.sqrt(dim_x)
        
    return A, C

def generate_sequences(key, batch_size, T, A, C, noise_std):
    # A, C can be real or complex
    dim_x = A.shape[0]
    dtype = A.dtype
    
    def step(x_t, k):
        # Noise
        # If A is complex (diagonal case), noise should be complex?
        # If we assume the underlying system is real but diagonalized, noise in z-space is P^-1 eps_real.
        # This means noise is complex with specific covariance.
        # To simplify, we'll just inject isotropic complex noise for the diagonal case, 
        # or isotropic real noise for the dense case.
        
        if jnp.iscomplexobj(A):
             # Complex noise
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
        # x1 = A x0 + eps = eps
        # y1 = C x1
        # We start loop from 0 to T-1 to generate x1...xT, y1...yT
        
        _, (xs, ys) = jax.lax.scan(step, x0, ks)
        return xs, ys
        
    xs, ys = jax.vmap(simulate_one)(keys)
    return xs, ys
