"""Load a trained actor and evaluate it on a frozen bank or reproducible sample."""

import argparse
import json

import numpy as np

from tangent_actor import evaluate, load_checkpoint
from tangent_actor.api import sample_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--bank", help="Path to your NPZ containing t and x")
    inputs.add_argument("--sample-states", type=int, metavar="N",
                        help="Draw N reproducible states for the checkpoint's PDE")
    args = parser.parse_args()
    if args.sample_states is not None and args.sample_states < 1:
        parser.error("--sample-states must be positive")

    problem, actor, payload = load_checkpoint(args.checkpoint)
    if args.bank is not None:
        with np.load(args.bank, allow_pickle=False) as arrays:
            t, x = arrays["t"], arrays["x"]
        if x.shape[1] != problem.d:
            parser.error("Bank and checkpoint have different dimensions")
    else:
        t, x = sample_states(problem, args.sample_states or 8, seed=17)

    values, metrics = evaluate(problem, actor, t, x, paths=16, time_steps=8, path_chunk=4)
    print(json.dumps({
        "checkpoint_iteration": payload["iteration"],
        "states": len(t),
        "finite": bool(np.isfinite(values["mcv"]).all()),
        "mean_mcv": float(values["mcv"].mean()),
        "metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
