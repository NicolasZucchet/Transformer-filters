import jax
import jax.numpy as jnp

def kalman_filter(y, A, C, Q_std, R_std=1e-4):
    """
    Runs Kalman Filter.
    y: (T, dy)
    A: (dx, dx)
    C: (dy, dx)
    Q_std: scalar
    """
    T, dy = y.shape
    dx = A.shape[0]
    dtype = A.dtype

    Q = (Q_std**2) * jnp.eye(dx, dtype=dtype)
    R = (R_std**2) * jnp.eye(dy, dtype=dtype)
    
    # x0 = 0
    # x1 = A x0 + eps = eps.
    # x1 ~ N(0, Q)
    x_init = jnp.zeros(dx, dtype=dtype)
    P_init = Q
    
    def step(carry, y_t):
        x_pred, P_pred = carry
        
        # Forecast observation
        y_hat = C @ x_pred
        
        # Innovation covariance
        S = C @ P_pred @ C.T + R
        
        # Gain
        K = P_pred @ C.T @ jnp.linalg.inv(S)
        
        # Update
        innovation = y_t - y_hat
        x_post = x_pred + K @ innovation
        # P_post = (I - KC) P_pred
        I = jnp.eye(dx, dtype=dtype)
        P_post = (I - K @ C) @ P_pred
        
        # Predict next state
        x_next = A @ x_post
        P_next = A @ P_post @ A.T + Q
        
        return (x_next, P_next), (y_hat, S)
        
    _, (y_preds, Ss) = jax.lax.scan(step, (x_init, P_init), y)
    
    return y_preds, Ss

def kalman_filter_final_state(y, A, C, Q_std, R_std=1e-4):
    """
    Returns the final posterior state x_{T|T}.
    y: (T, dy)
    """
    T, dy = y.shape
    dx = A.shape[0]
    dtype = A.dtype

    Q = (Q_std**2) * jnp.eye(dx, dtype=dtype)
    R = (R_std**2) * jnp.eye(dy, dtype=dtype)
    
    x_init = jnp.zeros(dx, dtype=dtype)
    P_init = Q
    
    def step(carry, y_t):
        x_pred, P_pred = carry
        
        # Innovation covariance
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ jnp.linalg.inv(S)
        
        # Update
        y_hat = C @ x_pred
        x_post = x_pred + K @ (y_t - y_hat)
        I = jnp.eye(dx, dtype=dtype)
        P_post = (I - K @ C) @ P_pred
        
        # Predict next
        x_next = A @ x_post
        P_next = A @ P_post @ A.T + Q
        
        return (x_next, P_next), x_post
        
    _, x_posts = jax.lax.scan(step, (x_init, P_init), y)
    
    return x_posts[-1]
