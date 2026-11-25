# Training configurations

Run a configuration with `python -m tangent_actor --config configs/smoke_joint.json --output runs/joint`. Use a new output directory per run.

The `smoke_*.json` configurations run on CPU. [smoke_cosine_hjb.json](smoke_cosine_hjb.json) trains the two-dimensional [custom PDE example](../README.md#use-tangent-actor-on-your-own-pde). The `paper_*.json` files give larger training budgets and the `variant_*.json` files select algorithm variants. Training and evaluation draw local samples. To evaluate an existing checkpoint on your own states, pass `--bank /path/to/evaluation.npz` to `examples/evaluate_checkpoint.py`; the arrays are `t` of shape `(n,)` and `x` of shape `(n, dimension)`.
