# Training configurations

Run a configuration with `python -m tangent_actor --config configs/smoke_joint.json --output runs/joint`. Use a new output directory per run.

The two `smoke_*.json` configurations run on CPU. The `paper_*.json` files give larger training budgets and the `variant_*.json` files select algorithm variants. Training and evaluation draw local samples. To evaluate an existing checkpoint on your own states, pass `--bank /path/to/evaluation.npz` to `examples/evaluate_checkpoint.py`; the arrays are `t` of shape `(n,)` and `x` of shape `(n, dimension)`.

OU/GMM and B6 require your own equation inputs. Set `TANGENT_ACTOR_DATA_DIR` to their directory. For OU/GMM, provide `ou_gmm_d<D>.npz` with scalar `dimension`, `mean` of shape `(5, D)`, and a normalized `pi` of shape `(5,)`. For B6, provide `b6_frozen_parameters.json` with a list of objects containing `dimension`, positive `beta`, and nonnegative `gamma`. The public B6 and Dense-Fourier constructors use demonstration coefficients.
