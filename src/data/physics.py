"""Physics-inspired LDS datasets."""
from abc import abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm
from src.data.lds import LDSDataset


def discretize_affine(A_c, b_c, dt):
    """Discretize affine system x' = A_c x + b_c via augmented matrix exponential.

    Returns (A_d, b_d) such that x_{k+1} = A_d x_k + b_d.
    """
    n = A_c.shape[0]
    # Augmented system: [A_c, b_c; 0, 0] of size (n+1, n+1)
    aug = np.zeros((n + 1, n + 1))
    aug[:n, :n] = A_c
    aug[:n, n] = b_c
    M = expm(aug * dt)
    A_d = M[:n, :n]
    b_d = M[:n, n]
    return A_d, b_d


def build_observation_matrix(dim_x, observation="position"):
    """Build C matrix for position-only or full (position + velocity) observation.

    State layout: [q1..qn, v1..vn] where n = dim_x // 2.
    """
    if observation == "full":
        return np.eye(dim_x)
    # Default: position only
    n = dim_x // 2
    return np.eye(n, dim_x)  # selects q1..qn


class PhysicsLDSDataset(LDSDataset):
    """Base class for physics-inspired LDS datasets."""

    def __init__(self, *, config, rng: jax.Array):
        # Build continuous system (numpy)
        A_c, b_c, dim_x = self._build_continuous_system(config)

        # Discretize
        b_c_arr = b_c if b_c is not None else np.zeros(dim_x)
        A_d, b_d = discretize_affine(A_c, b_c_arr, config.dt)

        # Build observation matrix
        C = build_observation_matrix(dim_x, config.observation)
        dim_y = C.shape[0]

        # Store as JAX arrays
        self._phys_A = jnp.array(A_d, dtype=jnp.float32)
        self._phys_C = jnp.array(C, dtype=jnp.float32)
        self._phys_b = jnp.array(b_d, dtype=jnp.float32) if b_c is not None else None

        # Initialize parent (will call _setup_system)
        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            lambda_val=0.0,  # unused, overridden by _setup_system
            structure="dense",  # unused
            sigma=0.0,  # no process noise for physics
            sequence_length=config.sequence_length,
            obs_noise_std=config.obs_noise_std,
            x0_std=config.x0_std,
            eval_sequence_length=config.eval_sequence_length,
            skip_kf=config.skip_kf,
            rng=rng,
        )
        self.name = "physics"

    def _setup_system(self, key, dim_x, dim_y, lambda_val, structure):
        """Override to use physics-derived A, C, b."""
        self.A = self._phys_A
        self.C = self._phys_C
        self.b = self._phys_b

    @abstractmethod
    def _build_continuous_system(self, config):
        """Build continuous-time system matrices.

        Returns:
            (A_c, b_c, dim_x) where A_c is (dim_x, dim_x),
            b_c is (dim_x,) or None, dim_x is state dimension.
        """
        pass


class HarmonicOscillatorDataset(PhysicsLDSDataset):
    """Harmonic oscillator: dq/dt=v, dv/dt=-omega^2*q."""

    def _build_continuous_system(self, config):
        w = config.omega
        A_c = np.array([[0.0, 1.0], [-w**2, 0.0]])
        return A_c, None, 2


class DampedOscillatorDataset(PhysicsLDSDataset):
    """Damped oscillator: dq/dt=v, dv/dt=-omega^2*q - 2*zeta*omega*v."""

    def _build_continuous_system(self, config):
        w = config.omega
        z = config.zeta
        A_c = np.array([[0.0, 1.0], [-w**2, -2*z*w]])
        return A_c, None, 2


class CoupledOscillatorsDataset(PhysicsLDSDataset):
    """Coupled oscillators with tridiagonal stiffness matrix."""

    def _build_continuous_system(self, config):
        n = config.n_oscillators
        w = config.omega
        kc = config.coupling_strength

        # Tridiagonal stiffness: K_ii = omega^2 + 2*kc, K_{i,i+1} = K_{i+1,i} = -kc
        K = np.diag(np.full(n, w**2 + 2*kc))
        for i in range(n - 1):
            K[i, i+1] = -kc
            K[i+1, i] = -kc

        # State: [q1..qn, v1..vn], dynamics: dq/dt = v, dv/dt = -K q
        dim_x = 2 * n
        A_c = np.zeros((dim_x, dim_x))
        A_c[:n, n:] = np.eye(n)  # dq/dt = v
        A_c[n:, :n] = -K          # dv/dt = -K q

        return A_c, None, dim_x


class ProjectileDataset(PhysicsLDSDataset):
    """2D projectile: dx/dt=vx, dy/dt=vy, dvx/dt=0, dvy/dt=-g."""

    def _build_continuous_system(self, config):
        g = config.gravity
        # State: [x, y, vx, vy]
        A_c = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        b_c = np.array([0.0, 0.0, 0.0, -g])
        return A_c, b_c, 4
