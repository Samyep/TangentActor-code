from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy.integrate import quad
from scipy.special import roots_laguerre

def hjb_log_reference(
    t: np.ndarray,
    x: np.ndarray,
    *,
    order: int = 128,
    return_gradient: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Hopf--Cole reference using scaled Gauss--Laguerre quadrature.

    ``exp(-u)=E[2/(1+||x+sqrt(2 tau) W||^2)]`` reduces exactly to a smooth
    one-dimensional integral.  The state enters only through ``||x||^2``;
    this fact is used by the reference calculator and is never exposed to an
    Actor.
    """
    times = np.asarray(t, dtype=np.float64).reshape(-1)
    states = np.asarray(x, dtype=np.float64).reshape(len(times), -1)
    tau = np.maximum(1.0 - times, 0.0)
    radius2 = np.square(states).sum(axis=1)
    nodes, weights = roots_laguerre(order)
    values = np.empty(len(times), dtype=np.float64)
    gradients = np.empty_like(states) if return_gradient else None
    terminal = tau <= 2e-14
    values[terminal] = np.log((1.0 + radius2[terminal]) / 2.0)
    if gradients is not None:
        gradients[terminal] = 2.0 * states[terminal] / (
            1.0 + radius2[terminal, None]
        )
    active = np.flatnonzero(~terminal)
    # Small batches avoid a large (#states, order) temporary for 20k-point PINN banks.
    for begin in range(0, len(active), 4096):
        index = active[begin : begin + 4096]
        local_tau = tau[index, None]
        local_r2 = radius2[index, None]
        scale = 1.0 + 2.0 * states.shape[1] * local_tau + local_r2
        s = nodes[None, :] / scale
        denominator = 1.0 + 4.0 * local_tau * s
        log_relative = (
            nodes[None, :]
            - s
            - 0.5 * states.shape[1] * np.log(denominator)
            - s * local_r2 / denominator
        )
        with np.errstate(divide="ignore"):
            log_terms = np.log(weights)[None, :] + log_relative
        log_sum = logsumexp(log_terms, axis=1)
        log_j = log_sum - np.log(scale[:, 0])
        values[index] = -math.log(2.0) - log_j
        if gradients is not None:
            normalized = np.exp(log_terms - log_sum[:, None])
            k_over_j = np.sum(normalized * s / denominator, axis=1)
            gradients[index] = 2.0 * states[index] * k_over_j[:, None]
    if gradients is None:
        return values
    return values, math.sqrt(2.0) * gradients


def hjb_log_quad_scalar(t: float, radius2: float, d: int = 100) -> float:
    tau = max(1.0 - t, 0.0)
    if tau == 0.0:
        return math.log((1.0 + radius2) / 2.0)

    def integrand(s: float) -> float:
        denominator = 1.0 + 4.0 * tau * s
        return math.exp(
            -s - 0.5 * d * math.log(denominator) - s * radius2 / denominator
        )

    integral, _ = quad(integrand, 0.0, np.inf, epsabs=2e-14, epsrel=2e-13, limit=500)
    return -math.log(2.0 * integral)
