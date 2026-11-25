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

from .feynman_kac import generate_labels
def _purpose(family: str) -> str:
    return {"ou_gmm": "official", "allen_cahn": "native"}.get(family, "function")

def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for item in arrays:
        value = np.ascontiguousarray(item)
        digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()

def _states(family: str, benchmark, count: int, *, split: str, seed: int):
    return benchmark.sample_states(
        count, seed=stable_seed("icml_standard_states", family, benchmark.d, split, seed),
        purpose=_purpose(family),
    )

def _predict(actor, benchmark, t: np.ndarray, x: np.ndarray, device: torch.device) -> np.ndarray:
    result = []
    actor.eval()
    with torch.no_grad():
        for begin in range(0, len(t), 1024):
            end = min(begin + 1024, len(t))
            r, q = actor.predict(
                torch.as_tensor(t[begin:end], dtype=torch.float32, device=device),
                torch.as_tensor(x[begin:end], dtype=torch.float32, device=device),
            )
            target = q if benchmark.actor_kind == "gradient" else r
            if target is None:
                raise RuntimeError("Actor returned the wrong head")
            result.append(target.double().cpu().numpy())
    return np.concatenate(result)

def _fixed_point(run_id: str, family: str, benchmark, actor_name: str, actor, seed: int, device, smoke: bool, *, estimator: str = "mcv"):
    if estimator not in {"plain", "mcv"}:
        raise ValueError(estimator)
    states, paths, steps = (16, 8, 4) if smoke else (128, 256, 32)
    t, x = _states(family, benchmark, states, split="fixed_point", seed=seed)
    labels, metadata = [], []
    for replication in (0, 1):
        value, meta = generate_labels(
            benchmark, actor, t, x, paths=paths, time_steps=steps,
            seed=stable_seed(run_id, "fixed_point", actor_name, replication),
            device=device, parent_batch=4 if smoke else 8,
            path_chunk=4 if smoke else 64, estimator=estimator,
        )
        labels.append(value); metadata.append(meta)
    prediction = _predict(actor, benchmark, t, x, device)
    target = 0.5 * (labels[0] + labels[1])
    raw = float(np.mean((prediction - target) ** 2))
    noise = float(np.mean((labels[0] - labels[1]) ** 2) / 4.0)
    corrected = max(raw - noise, 0.0)
    scale = max(float(np.sqrt(np.mean(target**2))), 1e-3)
    row = {
        "actor": actor_name, "estimator": estimator, "states": states, "paths_per_replication": paths,
        "time_steps": steps, "state_bank_sha256": _array_hash(t, x),
        "raw_component_rmse": math.sqrt(raw),
        "noise_corrected_component_rmse": math.sqrt(corrected),
        "normalized_total_inconsistency": corrected / scale**2,
        "minimum_weight_ess_fraction": min(item["weight_ess_fraction"] for item in metadata),
        "nonfinite_count": int(sum(item["nonfinite_count"] for item in metadata)),
        "ground_truth_used": False,
    }
    if target.ndim == 2:
        row["raw_vector_rmse"] = math.sqrt(benchmark.d * raw)
        row["noise_corrected_vector_rmse"] = math.sqrt(benchmark.d * corrected)
    return row
