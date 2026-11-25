from __future__ import annotations
import math,hashlib
from hashlib import sha256
from typing import Any
import numpy as np
import torch
from .utils import synchronize
def stable_seed(*parts: object) -> int:
    digest = sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")

from .joint_models import TerminalAwareActor
from .drivers import ConcaveSemilinearDriver
from .joint_mcv import mcv_tangent_labels
from .joint_fk import tangent_labels
def _state_bank(
    family: str, d: int, split: str, count: int, *, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed("icml_joint_state_bank", family, d, split, count, seed))
    t = rng.uniform(0.0, np.nextafter(0.5, 0.0), count).astype(np.float64)
    x = rng.standard_normal((count, d)).astype(np.float64)
    return t, x

def _array_hash(*arrays: np.ndarray) -> str:
    import hashlib
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()

def _predict(
    driver: ConcaveSemilinearDriver, actor: TerminalAwareActor,
    t: np.ndarray, x: np.ndarray, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    r_parts, q_parts = [], []
    actor.eval()
    with torch.no_grad():
        for begin in range(0, len(t), 1024):
            end = min(begin + 1024, len(t))
            r, q = actor.predict(
                torch.as_tensor(t[begin:end], dtype=torch.float32, device=device),
                torch.as_tensor(x[begin:end], dtype=torch.float32, device=device),
                driver.T,
            )
            if q is None:
                raise RuntimeError("joint Actor omitted q")
            r_parts.append(r.double().cpu().numpy())
            q_parts.append(q.double().cpu().numpy())
    return np.concatenate(r_parts), np.concatenate(q_parts)

def _fixed_point(
    run_id: str, actor_name: str, driver: ConcaveSemilinearDriver,
    actor: TerminalAwareActor, *, training_seed: int, device: torch.device,
    smoke: bool, estimator: str = "mcv",
) -> dict[str, Any]:
    if estimator not in {"plain", "mcv"}:
        raise ValueError(estimator)
    labeler = tangent_labels if estimator == "plain" else mcv_tangent_labels
    states, paths, steps = (16, 8, 4) if smoke else (128, 256, 32)
    t, x = _state_bank(driver.family, driver.d, "fixed_point", states, seed=training_seed)
    labels = []
    metadata = []
    for replicate in (0, 1):
        value, q, meta = labeler(
            driver, actor, t, x, paths=paths, time_steps=steps,
            seed=stable_seed(run_id, "fixed_point", actor_name, replicate),
            device=device, parent_batch=4 if smoke else 8,
            path_chunk=4 if smoke else 64,
        )
        if q is None:
            raise RuntimeError("missing q fixed-point label")
        labels.append((value, q)); metadata.append(meta)
    actor_r, actor_q = _predict(driver, actor, t, x, device)
    target_r = 0.5 * (labels[0][0] + labels[1][0])
    target_q = 0.5 * (labels[0][1] + labels[1][1])
    raw_r = float(np.mean(np.square(actor_r - target_r)))
    raw_q = float(np.mean(np.square(actor_q - target_q)))
    noise_r = float(np.mean(np.square(labels[0][0] - labels[1][0])) / 4.0)
    noise_q = float(np.mean(np.square(labels[0][1] - labels[1][1])) / 4.0)
    corrected_r = max(raw_r - noise_r, 0.0)
    corrected_q = max(raw_q - noise_q, 0.0)
    r_scale = max(float(np.std(target_r)), 1e-2)
    q_scale = max(float(np.sqrt(np.mean(np.square(target_q)))), 1e-3)
    return {
        "actor": actor_name, "estimator": estimator, "states": states, "paths_per_replication": paths,
        "time_steps": steps, "state_bank_sha256": _array_hash(t, x),
        "value_raw_rmse": math.sqrt(raw_r),
        "value_noise_corrected_rmse": math.sqrt(corrected_r),
        "gradient_raw_component_rmse": math.sqrt(raw_q),
        "gradient_raw_vector_rmse": math.sqrt(driver.d * raw_q),
        "gradient_noise_corrected_component_rmse": math.sqrt(corrected_q),
        "gradient_noise_corrected_vector_rmse": math.sqrt(driver.d * corrected_q),
        "normalized_value_inconsistency": corrected_r / r_scale**2,
        "normalized_gradient_inconsistency": corrected_q / q_scale**2,
        "normalized_total_inconsistency": corrected_r / r_scale**2 + corrected_q / q_scale**2,
        "minimum_weight_ess_fraction": min(m["weight_ess_fraction"] for m in metadata),
        "nonfinite_count": int(sum(m["nonfinite_count"] for m in metadata)),
        "ground_truth_used": False,
    }
