from __future__ import annotations
import math,time
from typing import Any
import numpy as np
import torch
from torch import nn
from .joint_fk import frozen,antithetic_normals
from .joint_mcv import _weighted_rollout_chunk_mcv_labels
from .sine_drivers import SineQuadraticDriver
from .utils import synchronize

def evaluate_policy(
    driver: SineQuadraticDriver,
    actor: nn.Module,
    t: np.ndarray,
    x: np.ndarray,
    *,
    paths: int,
    time_steps: int,
    seed: int,
    device: torch.device,
    parent_batch: int = 8,
    path_chunk: int = 64,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate plain and alpha=1 MCV estimates on exactly shared paths."""
    if paths % path_chunk or path_chunk % 2:
        raise ValueError("paths must be divisible by an even path chunk")
    times = np.asarray(t, dtype=np.float64).reshape(-1)
    states = np.asarray(x, dtype=np.float64).reshape(-1, driver.d)
    plain_mean = np.empty(len(times), dtype=np.float64)
    mcv_mean = np.empty(len(times), dtype=np.float64)
    plain_variance = np.empty(len(times), dtype=np.float64)
    mcv_variance = np.empty(len(times), dtype=np.float64)
    generator = torch.Generator(device=device).manual_seed(seed)
    weight_sum = weight_square_sum = 0.0
    weight_min, weight_max = float("inf"), -float("inf")
    nonfinite = 0
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with frozen(actor):
        for begin in range(0, len(times), parent_batch):
            end = min(begin + parent_batch, len(times))
            bt = torch.as_tensor(times[begin:end], dtype=torch.float32, device=device)
            bx = torch.as_tensor(states[begin:end], dtype=torch.float32, device=device)
            plain_pairs: list[torch.Tensor] = []
            mcv_pairs: list[torch.Tensor] = []
            for _ in range(0, paths, path_chunk):
                normals = antithetic_normals(
                    (end - begin, path_chunk, time_steps, driver.d),
                    generator=generator,
                    device=device,
                )
                corrected, _, weight, martingale, diagnostics = (
                    _weighted_rollout_chunk_mcv_labels(
                        driver, actor, bt, bx, normals, need_z=False
                    )
                )
                plain = corrected + martingale
                half = path_chunk // 2
                plain_pairs.append(0.5 * (plain[:, :half] + plain[:, half:]))
                mcv_pairs.append(
                    0.5 * (corrected[:, :half] + corrected[:, half:])
                )
                weight_sum += float(weight.double().sum().item())
                weight_square_sum += float(weight.double().square().sum().item())
                weight_min = min(weight_min, float(weight.min().item()))
                weight_max = max(weight_max, float(weight.max().item()))
                nonfinite += int(diagnostics["nonfinite_count"])
            local_plain = torch.cat(plain_pairs, dim=1).double()
            local_mcv = torch.cat(mcv_pairs, dim=1).double()
            plain_mean[begin:end] = local_plain.mean(dim=1).cpu().numpy()
            mcv_mean[begin:end] = local_mcv.mean(dim=1).cpu().numpy()
            plain_variance[begin:end] = local_plain.var(
                dim=1, unbiased=True
            ).cpu().numpy()
            mcv_variance[begin:end] = local_mcv.var(
                dim=1, unbiased=True
            ).cpu().numpy()
    synchronize(device)
    elapsed = time.perf_counter() - started
    count = len(times) * paths
    if nonfinite or not np.all(np.isfinite(mcv_mean)):
        raise FloatingPointError("nonfinite tangent-policy evaluation")
    return {
        "plain": plain_mean,
        "mcv": mcv_mean,
        "plain_pair_variance": plain_variance,
        "mcv_pair_variance": mcv_variance,
    }, {
        "paths": paths,
        "antithetic_pairs": paths // 2,
        "time_steps": time_steps,
        "seed": seed,
        "mcv_alpha": 1.0,
        "mean_plain_pair_variance": float(np.mean(plain_variance)),
        "mean_mcv_pair_variance": float(np.mean(mcv_variance)),
        "variance_reduction": float(
            np.mean(plain_variance) / max(np.mean(mcv_variance), 1e-300)
        ),
        "mean_plain_mc_se": float(
            np.sqrt(np.sum(plain_variance / (paths // 2))) / len(times)
        ),
        "mean_mcv_mc_se": float(
            np.sqrt(np.sum(mcv_variance / (paths // 2))) / len(times)
        ),
        "weight_ess_fraction": float(
            weight_sum**2
            / max(count * weight_square_sum, np.finfo(float).tiny)
        ),
        "weight_min": weight_min,
        "weight_max": weight_max,
        "nonfinite_count": nonfinite,
        "wall_seconds": elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
    }
