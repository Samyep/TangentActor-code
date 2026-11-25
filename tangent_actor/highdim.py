from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hashlib,math
from functools import lru_cache
import numpy as np
import torch
from scipy.special import logsumexp
from .drivers import ConcaveSemilinearDriver
SQRT2=math.sqrt(2.0)
BASE_SEED = 17  # Reproducible example coefficients for this distribution.

class ExactEquation:
    """Small protocol shared by all analytic equations."""

    name: str
    family: str
    d: int
    T: float
    sigma: float
    mu: float | np.ndarray

    def exact_state(self, t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
        u = np.asarray(self.exact_u(t, x), dtype=np.float64)
        z = np.asarray(self.exact_z(t, x), dtype=np.float64)
        return np.concatenate((u[..., None], z), axis=-1)

    def exact_generator(self, t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
        state = self.exact_state(t, x)
        return self.generator(state[..., 0], state[..., 1:])

    @property
    def has_u_certificate(self) -> bool:
        return False

    @property
    def u_interval(self) -> tuple[float, float] | None:
        return None

    def certificate_labels(self) -> dict[str, str]:
        raise NotImplementedError

    def certificate_parameters(self) -> dict[str, Any]:
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "dimension": self.d,
            "T": self.T,
            "sigma": self.sigma,
            "mu": np.asarray(self.mu, dtype=np.float64).tolist(),
            "dtype": "float64",
            "state_convention": "z=sigma*grad(u)",
            "certificate_labels": self.certificate_labels(),
            "certificate_parameters": self.certificate_parameters(),
        }

def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


class DenseFourierDriver(ConcaveSemilinearDriver):
    """Joint f(u,z)=0.5*tanh(u)*|z|^2 with an analytic sinh transform."""

    family = "dense_fourier_joint"
    has_gradient = True
    sigma = SQRT2
    reconstruction_sign = 1.0
    # These are metadata-only guards: the preregistered actor below is
    # intentionally unbounded and does not project or sigmoid-transform r.
    r_low, r_high = -20.0, 20.0

    def __init__(
        self,
        omega: np.ndarray,
        signs: np.ndarray,
        phases: np.ndarray,
        *,
        T: float = 0.5,
        parameter_seed: tuple[int, ...] = (),
    ) -> None:
        omega = np.asarray(omega, dtype=np.float64)
        signs = np.asarray(signs, dtype=np.float64)
        phases = np.asarray(phases, dtype=np.float64)
        if omega.ndim != 2 or len(signs) != len(omega) or len(phases) != len(omega):
            raise ValueError("incompatible Dense Fourier parameters")
        self.k, self.d = omega.shape
        if self.k != 2 * self.d:
            raise ValueError("the preregistered benchmark requires k=2d")
        norms = np.linalg.norm(omega, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-14):
            raise ValueError("omega rows must be unit vectors")
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("a_j must be signs")
        self.T = float(T)
        self.name = f"dense_fourier_joint_d{self.d}_k{self.k}"
        self.omega = omega.copy()
        self.signs = signs.copy()
        self.phases = phases.copy()
        self.omega_norm_squared = np.sum(self.omega * self.omega, axis=1)
        self.scale = 2.0 / math.sqrt(2.0 * self.d)
        self.parameter_seed = tuple(int(value) for value in parameter_seed)
        self.basis_seed = -1
        # Some inherited diagnostics expect B.  It is the complete dense
        # direction matrix, not a low-rank oracle feature exposed to the actor.
        self.B = self.omega.T
        self._tensor_cache: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}

    def _tensor(self, name: str, value: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        key = (name, str(like.device), like.dtype)
        result = self._tensor_cache.get(key)
        if result is None:
            result = torch.as_tensor(value, dtype=like.dtype, device=like.device)
            self._tensor_cache[key] = result
        return result

    def _h_torch(
        self, x: torch.Tensor, attenuation: torch.Tensor | float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        omega = self._tensor("omega", self.omega, x)
        signs = self._tensor("signs", self.signs, x)
        phases = self._tensor("phases", self.phases, x)
        angle = x @ omega.T + phases
        factor = signs * attenuation
        h = 0.5 + self.scale * torch.sum(factor * torch.sin(angle), dim=-1)
        grad_h = self.scale * ((factor * torch.cos(angle)) @ omega)
        return h, grad_h

    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self._h_torch(x)
        return torch.asinh(h)

    def terminal_gradient_torch(self, x: torch.Tensor) -> torch.Tensor:
        h, grad_h = self._h_torch(x)
        return grad_h / torch.sqrt(1.0 + h.square()).unsqueeze(-1)

    def terminal_numpy(self, x: np.ndarray) -> np.ndarray:
        state = np.asarray(x, dtype=np.float64)
        angle = state @ self.omega.T + self.phases
        h = 0.5 + self.scale * np.sum(self.signs * np.sin(angle), axis=-1)
        return np.arcsinh(h)

    def terminal_gradient_numpy(self, x: np.ndarray) -> np.ndarray:
        state = np.asarray(x, dtype=np.float64)
        angle = state @ self.omega.T + self.phases
        h = 0.5 + self.scale * np.sum(self.signs * np.sin(angle), axis=-1)
        grad_h = self.scale * ((self.signs * np.cos(angle)) @ self.omega)
        return grad_h / np.sqrt(1.0 + h * h)[..., None]

    def terminal_from_coordinates_numpy(self, s: np.ndarray) -> np.ndarray:
        angle = np.asarray(s, dtype=np.float64) + self.phases
        return np.arcsinh(
            0.5 + self.scale * np.sum(self.signs * np.sin(angle), axis=-1)
        )

    def terminal_coordinate_gradient_numpy(self, s: np.ndarray) -> np.ndarray:
        angle = np.asarray(s, dtype=np.float64) + self.phases
        h = 0.5 + self.scale * np.sum(self.signs * np.sin(angle), axis=-1)
        return (
            self.scale
            * self.signs
            * np.cos(angle)
            / np.sqrt(1.0 + h * h)[..., None]
        )

    def exact_state(self, t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
        state = np.asarray(x, dtype=np.float64)
        leading = state.shape[:-1]
        time = np.broadcast_to(np.asarray(t, dtype=np.float64), leading)
        tau = np.maximum(self.T - time, 0.0)
        attenuation = np.exp(-tau[..., None] * self.omega_norm_squared)
        angle = state @ self.omega.T + self.phases
        weighted = attenuation * self.signs
        w = 0.5 + self.scale * np.sum(weighted * np.sin(angle), axis=-1)
        grad_w = self.scale * ((weighted * np.cos(angle)) @ self.omega)
        u = np.arcsinh(w)
        z = self.sigma * grad_w / np.sqrt(1.0 + w * w)[..., None]
        return np.concatenate((u[..., None], z), axis=-1)

    def exact_u(self, t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
        return self.exact_state(t, x)[..., 0]

    def exact_z(self, t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
        return self.exact_state(t, x)[..., 1:]

    def exact_u_torch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(self.T - t.reshape(-1), min=0.0)
        norms = self._tensor("omega_n2", self.omega_norm_squared, x)
        attenuation = torch.exp(-tau[:, None] * norms)
        w, _ = self._h_torch(x, attenuation)
        return torch.asinh(w)

    def driver(self, y: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        if z is None:
            raise ValueError("Dense Fourier is jointly value-gradient nonlinear")
        return 0.5 * torch.tanh(y) * torch.sum(z.square(), dim=-1)

    def driver_y(self, y: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        if z is None:
            raise ValueError("Dense Fourier is jointly value-gradient nonlinear")
        return 0.5 * (1.0 - torch.tanh(y).square()) * torch.sum(
            z.square(), dim=-1
        )

    def driver_z(self, y: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        if z is None:
            raise ValueError("Dense Fourier is jointly value-gradient nonlinear")
        return torch.tanh(y).unsqueeze(-1) * z

    def driver_reduced_numpy(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return 0.5 * np.tanh(y) * np.sum(np.square(z), axis=-1)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "d": self.d,
            "k": self.k,
            "T": self.T,
            "sigma": self.sigma,
            "has_gradient": True,
            "actor_input": "full ambient x",
            "actor_parameterization": "r=g+tau*R; q=sqrt(2)*grad(g)+tau*Q",
            "analytically_integrable_high_dimensional_stress_test": True,
            "parameter_seed": list(self.parameter_seed),
            "parameter_sha256": array_sha256(self.omega, self.signs, self.phases),
            "omega_sha256": array_sha256(self.omega),
            "signs_sha256": array_sha256(self.signs),
            "phases_sha256": array_sha256(self.phases),
            "omega_norm_max_error": float(
                np.max(np.abs(self.omega_norm_squared - 1.0))
            ),
            "oracle_projection_used": False,
        }

@dataclass(frozen=True)
class B6Equation(ExactEquation):
    """Full-rank nonseparable log-sum-exp Hopf--Cole benchmark."""

    d: int
    beta: float
    gamma: float
    T: float = 0.25
    sigma: float = SQRT2
    family: str = "b6_fullrank_lse"
    name: str = "B6_fullrank_nonseparable_hjb"

    def __post_init__(self) -> None:
        if self.d <= 0 or self.beta <= 0.0 or self.gamma < 0.0:
            raise ValueError((self.d, self.beta, self.gamma))
        count = 2 * int(self.d)
        rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, self.d, 66]))
        directions = rng.standard_normal((count, self.d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        multipliers = rng.uniform(0.5, 1.5, count)
        common = rng.standard_normal(self.d)
        common /= np.linalg.norm(common)
        vectors = (
            self.gamma * common[None, :]
            + self.beta * multipliers[:, None] * directions
        ).astype(np.float64)
        squared_norms = np.sum(vectors * vectors, axis=1)
        for value in (directions, multipliers, common, vectors, squared_norms):
            value.setflags(write=False)
        object.__setattr__(self, "K", count)
        object.__setattr__(self, "e", directions)
        object.__setattr__(self, "U", multipliers)
        object.__setattr__(self, "v", common)
        object.__setattr__(self, "b", vectors)
        object.__setattr__(self, "n2", squared_norms)
        object.__setattr__(self, "mu", 0.0)

    def _flat_logits(
        self, t: np.ndarray | float, x: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        state = np.asarray(x, dtype=np.float64)
        if state.shape[-1] != self.d:
            raise ValueError((state.shape, self.d))
        leading = state.shape[:-1]
        time = np.broadcast_to(np.asarray(t, dtype=np.float64), leading).reshape(-1)
        flat = state.reshape(-1, self.d)
        logits = flat @ self.b.T + np.maximum(self.T - time, 0.0)[:, None] * self.n2
        return logits, leading

    def terminal(self, x: np.ndarray) -> np.ndarray:
        state = np.asarray(x, dtype=np.float64)
        leading = state.shape[:-1]
        logits = state.reshape(-1, self.d) @ self.b.T
        return (-(logsumexp(logits, axis=1) - math.log(self.K))).reshape(leading)

    def terminal_gradient_z(self, x: np.ndarray) -> np.ndarray:
        state = np.asarray(x, dtype=np.float64)
        leading = state.shape[:-1]
        logits = state.reshape(-1, self.d) @ self.b.T
        weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        return (-self.sigma * (weights @ self.b)).reshape(*leading, self.d)

    def terminal_and_z(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.terminal(x), self.terminal_gradient_z(x)

    def exact_state(self, t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
        logits, leading = self._flat_logits(t, x)
        normalizer = logsumexp(logits, axis=1, keepdims=True)
        weights = np.exp(logits - normalizer)
        value = -(normalizer[:, 0] - math.log(self.K))
        z = -self.sigma * (weights @ self.b)
        return np.concatenate((value[:, None], z), axis=1).reshape(*leading, self.d + 1)

    def exact_u(self, t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
        return self.exact_state(t, x)[..., 0]

    def exact_z(self, t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
        return self.exact_state(t, x)[..., 1:]

    def probabilities(self, t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
        logits, leading = self._flat_logits(t, x)
        result = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        return result.reshape(*leading, self.K)

    def generator(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        del u
        return -0.5 * np.sum(np.square(z), axis=-1)

    @staticmethod
    def double_generator(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return -0.5 * np.sum(np.asarray(first) * np.asarray(second), axis=-1)

    @property
    def box_low(self) -> np.ndarray:
        return np.full(self.d, -np.inf)

    @property
    def box_high(self) -> np.ndarray:
        return np.full(self.d, np.inf)

    @property
    def z_ball_radius(self) -> float:
        return float("inf")

    @property
    def z_center(self) -> np.ndarray:
        return np.zeros(self.d)

    @property
    def dose_scale(self) -> float:
        return 1.0

    @property
    def picard_proxy(self) -> float:
        return float("nan")

    def certificate_labels(self) -> dict[str, str]:
        return {"none": "diagnostic-only unbounded certificate attributes"}

    def certificate_parameters(self) -> dict[str, Any]:
        return {
            "pde_id": "B6",
            "beta": self.beta,
            "gamma": self.gamma,
            "K": self.K,
            "construction_seed": [BASE_SEED, self.d, 66],
            "directions_sha256": hashlib.sha256(self.e.tobytes()).hexdigest(),
            "multipliers_sha256": hashlib.sha256(self.U.tobytes()).hexdigest(),
            "common_vector_sha256": hashlib.sha256(self.v.tobytes()).hexdigest(),
            "vectors_sha256": hashlib.sha256(self.b.tobytes()).hexdigest(),
        }


@lru_cache(maxsize=None)
def make_b6_equation(d: int, beta: float, gamma: float) -> B6Equation:
    return B6Equation(int(d), float(beta), float(gamma))


class HJBActorBenchmark:
    T = 0.25
    sigma = SQRT2
    actor_kind = "gradient"
    input_scale = 1.0
    value_bounds = None

    def __init__(self, equation: B6Equation):
        self.pde = "b6_fullrank"
        self.name = f"B6_d{equation.d}"
        self.d = equation.d
        self.equation = equation
        self._tensor_cache: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}

    def _vectors(self, x: torch.Tensor) -> torch.Tensor:
        key = ("vectors", str(x.device), x.dtype)
        value = self._tensor_cache.get(key)
        if value is None:
            value = torch.as_tensor(self.equation.b.copy(), dtype=x.dtype, device=x.device)
            self._tensor_cache[key] = value
        return value

    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        logits = x @ self._vectors(x).T
        return -(torch.logsumexp(logits, dim=-1) - math.log(self.equation.K))

    def terminal_gradient(self, x: torch.Tensor) -> torch.Tensor:
        vectors = self._vectors(x)
        return -(torch.softmax(x @ vectors.T, dim=-1) @ vectors)

    def base_drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return torch.zeros_like(x)

    def driver(self, r: torch.Tensor, q: torch.Tensor | None) -> torch.Tensor:
        del r
        if q is None:
            raise ValueError("B6 needs a Gradient Actor")
        return -0.5 * q.square().sum(-1)

    def tangent_coefficients(self, r, q):
        del r
        if q is None:
            raise ValueError("B6 needs a Gradient Actor")
        c = torch.zeros(q.shape[:-1], dtype=q.dtype, device=q.device)
        return c, -q, 0.5 * q.square().sum(-1)

    def sample_states(self, count: int, *, seed: int, purpose: str = "function"):
        if purpose not in {"function", "official"}:
            raise KeyError(purpose)
        rng = np.random.default_rng(seed)
        t = rng.uniform(0.0, np.nextafter(self.T, 0.0), int(count))
        x = rng.uniform(-1.0, 1.0, (int(count), self.d))
        return t.astype(np.float64), x.astype(np.float64)


def dense_parameters(d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct reproducible coefficients for the example equation."""
    d = int(d)
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, d, 41]))
    omega = rng.standard_normal((2 * d, d)).astype(np.float64)
    omega /= np.linalg.norm(omega, axis=1, keepdims=True)
    signs = (2 * rng.integers(0, 2, 2 * d) - 1).astype(np.float64)
    phases = rng.uniform(0.0, 2.0 * np.pi, 2 * d).astype(np.float64)
    return omega, signs, phases


@lru_cache(maxsize=None)
def make_dense_fourier(d: int) -> DenseFourierDriver:
    omega, signs, phases = dense_parameters(int(d))
    return DenseFourierDriver(
        omega,
        signs,
        phases,
        T=0.5,
        parameter_seed=(BASE_SEED, int(d), 41),
    )
