# Tangent-Actor

**Learning to Linearize High-Dimensional Semilinear PDEs**

Tangent-Actor learns a value or gradient field, uses it to linearize the nonlinear PDE driver, and evaluates the resulting linear problem by Feynman–Kac sampling. The learned gradient also serves as a martingale control variate.

![Tangent-Actor method](figures/Tangent_Actor.png)

## Paper figures and evaluation data

| Evaluation | Paper coverage | Download |
| --- | --- | --- |
| Whole-function accuracy | Eight PDEs, five dimensions, four methods | [RMSE and RelMAE](evaluation/whole_function.csv) |
| Single-point accuracy | Four PDEs, five dimensions, three methods | [RMSE, RelMAE and numerical failure status](evaluation/single_point.csv) |
| Component ablation | Four variants on four 100D PDEs | [Mean RMSE and sample standard deviation](evaluation/component_ablation.csv) |

At the precision printed in the manuscript, Tangent-Actor has the lowest whole-function RMSE in 39 of 40 settings and the lowest single-point RMSE in 20 of 20. [Evaluation notes](evaluation/) describe how the paper tables and historical run summaries were obtained. [All figures](figures/) includes the method diagram, accuracy and ablation plots, solution landscapes and appendix panels.

![Four-PDE solution landscape](figures/fugu.png)

## Run the code

From a checkout, using Python 3.11 or newer:

```bash
python -m pip install -e ".[test]"
python examples/evaluate_checkpoint.py --checkpoint examples/weights/joint_d4_smoke.pt --sample-states 8
python examples/hjb_log_smoke.py
python examples/joint_smoke.py
python -m pytest -q
```

The [example checkpoint](examples/weights/joint_d4_smoke.pt) is a small 4D coupled nonconvex actor from the CPU training example. The evaluation command loads it and samples new states. To use your own test bank, substitute `--bank /path/to/your_bank.npz`. The file must contain `t` with shape `(n,)` and `x` with shape `(n, dimension)`.

## Train Tangent-Actor

From the repository root, train a 4D actor for the coupled nonconvex PDE on CPU. This configuration samples its own training and validation states.

```bash
python -m tangent_actor --config configs/smoke_joint.json --output runs/joint_4d --device cpu
```

Choose a new output directory for each run. Use `runs/joint_4d/actor.pt` for inference. It contains the actor selected by the run and the configuration needed to rebuild it. The same directory holds the per-update checkpoints (`actor_a1.pt`, `actor_a2.pt`), selection diagnostics (`summary.json`), and local evaluation states and predictions (`evaluation.npz`).

For a larger run, choose a `paper_*.json` configuration and a fresh output path. To change the training budget, edit `training_points`, `paths_per_label`, or `optimizer_steps` in a copy of the configuration. On a machine with CUDA-enabled PyTorch, use `--device cuda`.

## Deploy Tangent-Actor

On the target machine, check out the same code revision used for training. Create a `models/` directory under the checkout and copy the selected checkpoint there as `models/actor.pt`.

From the repository root, install the package and check that the actor produces finite estimates on eight sampled states. The JSON output should report `"finite": true`.

```bash
python -m pip install .
python examples/evaluate_checkpoint.py --checkpoint models/actor.pt --sample-states 8
```

In a Python application, load the actor once and reuse it across batches. This example estimates the solution at time zero and the spatial origin.

```python
import numpy as np
from tangent_actor import evaluate, load_checkpoint

problem, actor, _ = load_checkpoint("models/actor.pt", device="cpu")
t = np.array([0.0])
x = np.zeros((1, problem.d))
values, _ = evaluate(problem, actor, t, x, paths=16, time_steps=8, path_chunk=4)
print(values["mcv"])
```

For your own queries, pass times with shape `(n,)` and states with shape `(n, problem.d)`. Each entry of `values["mcv"]` estimates the PDE solution at the corresponding time and state using the martingale control variate. Adjust `paths` and `time_steps` to set the inference budget. Use at least four paths and choose an even `path_chunk` that divides `paths`.

## Use Tangent-Actor on your own PDE

The runnable `cosine_hjb` example solves a terminal-value problem on $\mathbb{R}^2$. It has zero base drift, unit diffusion, and terminal time $T=0.5$.

$$
\partial_t u + \tfrac12 \Delta u + \sqrt{1+\lVert\nabla u\rVert^2}-1=0,
\qquad u(T,x)=\tfrac12(\cos x_1+\cos x_2).
$$

### What to change for a new equation

| Where | What to change |
| --- | --- |
| [Equation class](tangent_actor/cosine_hjb.py) | Copy `CosineHJBBenchmark`, give it a new `pde` identifier, and set `T` and the constant scalar diffusion `sigma`. Implement the terminal value `terminal(x)` and its spatial gradient `terminal_gradient(x)`. Override `base_drift(t, x)` for nonzero drift and `sample_states` to sample the region where you need predictions. The example inherits zero drift and standard normal spatial samples. |
| [Driver and tangent coefficients](tangent_actor/cosine_hjb.py) | Write `driver(r, q)` with $r=u$ and $q=\sigma\nabla u$. For a smooth driver $f$, return $c=\partial_r f$, $b=\nabla_q f$, and $h=f-cr-b\cdot q$ from `tangent_coefficients`. In this example $c=0$ and $b=q/\sqrt{1+\lVert q\rVert^2}$. For a batch of $n$ states, `c` and `h` have shape `(n,)` and `b` has shape `(n, d)`. |
| [Registration](tangent_actor/api.py) | Import the class, add its identifier to `PDES`, and add a branch in `make_problem(pde, dimension)` that returns your class for that identifier. Both training and checkpoint loading use this function. |
| [Training configuration](configs/smoke_cosine_hjb.json) | Copy this JSON file, set `pde` to the registered identifier and `dimension` to the number of spatial coordinates. Start with the small CPU budget to check the full workflow. |

Use `actor_kind = "gradient"` for a driver that depends only on $q$, as above, or `"value"` for one that depends only on $r$. A driver that depends on both uses the [joint value–gradient interface](docs/ALGORITHM_AND_INTERFACES.md#joint-driver-interface). The example uses a whole-space rollout with constant scalar diffusion. Spatial boundary conditions or a state-dependent diffusion require corresponding changes to that rollout.

### Train the example and deploy it

From the repository root, install the package and train into a new directory.

```bash
python -m pip install -e .
python -m tangent_actor --config configs/smoke_cosine_hjb.json --output runs/cosine_2d --device cpu
```

On the target machine, install the same revision of your fork with `python -m pip install .` so it includes your equation and its registration. Copy `runs/cosine_2d/actor.pt` to `models/actor.pt`. The following application loads the trained actor, estimates the solution at an interior time, and checks a second query against the known terminal condition.

```python
import numpy as np
from tangent_actor import evaluate, load_checkpoint

problem, actor, _ = load_checkpoint("models/actor.pt", device="cpu")
t = np.array([0.25, problem.T])
x = np.array([[0.4, -0.2], [0.1, 0.3]])
values, _ = evaluate(problem, actor, t, x, paths=16, time_steps=8, path_chunk=4)
u = values["mcv"]
print(u[0])
np.testing.assert_allclose(u[1], np.cos(x[1]).mean(), atol=1e-6)
```

## Citation and licenses

If you use the code in research, please cite Yeping Wang, *Tangent-Actor: Learning to Linearize High-Dimensional Semilinear PDEs* (2026 manuscript). [Citation metadata](CITATION.cff).

Original code and the bundled example checkpoint are provided under the [MIT License](LICENSE). Original paper figures are available under [CC BY 4.0](LICENSES/CC-BY-4.0.txt). To reuse or adapt a figure, credit the Tangent-Actor contributors, link to the [license](https://creativecommons.org/licenses/by/4.0/) and identify any changes.
