# Algorithm and extension interfaces

Use the convention `z = sigma * grad(u)` and a backward semilinear equation with driver `f(r,z)`. At the frozen actor state `(r,z)`, each benchmark supplies tangent coefficients `(c,b,h)` satisfying

```text
f(r,z) = c*r + b.dot(z) + h.
```

The rollout uses the state-dependent tangent drift, exponential weight and source term to generate linear FK labels. Gradient labels differentiate the complete sampled payoff with respect to the initial spatial state. The MCV uses a predictable left-endpoint integrand and the unscaled Brownian increment. Its pathwise gradient differentiates the corrected payoff, including its martingale term. These are the original numerical implementations; the portable API joins them together.

Full-ambient actors receive `(t,x)` including every spatial coordinate. Terminal-aware outputs attach the exact terminal value/gradient plus a correction multiplied by `T-t`; at `t=T` the correction vanishes. Joint bounded-value actors retain the source value projection/bounded parameterization. Dense-Fourier instead retains the original unbounded terminal-aware value and gradient heads. The standard-network and no-terminal-anchor variants deliberately remove the terminal guarantee and are labeled ablations.

## Standard benchmark interface

Subclass `tangent_actor.benchmarks.Benchmark` and implement:

```python
class CustomBenchmark(Benchmark):
    d: int
    T: float
    sigma: float
    actor_kind = 'gradient'  # or 'value'
    def terminal(self, x): ...
    def terminal_gradient(self, x): ...
    def driver(self, r, q): ...
    def tangent_coefficients(self, r, q): ...  # c, b, h
    def base_drift(self, t, x): ...
    def sample_states(self, count, *, seed, purpose='function'): ...
```

Methods take PyTorch tensors except `sample_states`, which returns NumPy time and state arrays. Missing terminal gradients can use the base autograd implementation. Diffusion `sigma` is scalar in this release. For value-only drivers `q` may be `None`. Instantiate `ambient.TerminalAwareActor` and call `feynman_kac.generate_labels`, `fit_actor`, and `evaluate_policy` directly for an external benchmark; the command-line factory enumerates the nine bundled PDEs.

## Joint driver interface

Subclass `drivers.ConcaveSemilinearDriver`, supplying `terminal`, `terminal_gradient_torch`, `base_drift_torch`, `driver`, `tangent_coefficients`, `constrain_r`, `project_r_numpy`, `metadata`, and the domain attributes used by the source actor. The included concrete drivers demonstrate the complete API. Joint actor prediction is `actor.predict(t,x,T)`; standard benchmark prediction is `actor.predict(t,x)`.

`drivers.SignTransformedDriver` solves the sign-transformed jointly convex equation. The portable `evaluate` restores physical value signs; direct low-level functions operate on the working driver. B6 reads the user-supplied beta/gamma values. The public B6 and Dense-Fourier constructors use a reproducible demonstration seed. The two high-dimensional definitions retain their original equations rather than reducing to the known active coordinates.

## Evaluation semantics

`api.evaluate` returns `plain`, `mcv`, `plain_pair_variance`, and `mcv_pair_variance`. Plain and MCV estimates share the same simulated paths, with antithetic pairing inside each even-sized chunk. Variance is calculated over antithetic pair averages, not over dependent individual paths. Use at least four paths, an even `path_chunk`, and `paths % path_chunk == 0`. Paths and time steps are numerical approximation parameters, not guarantees of accuracy.

The portable `api.train` separates training, validation, selection and final evaluation streams; noisy validation labels choose fitting checkpoints. Reference-free selection uses two independent label replications with a nonnegative noise-corrected inconsistency score. It never reads an exact nonterminal solution. `no_updates` evaluates the initial actor; `no_mcv` changes labels and selector to plain. `api.save_checkpoint`/`load_checkpoint` use a portable state-dict schema with strict restoration and `weights_only=True`. `scripts/import_checkpoint.py` converts trusted legacy state dictionaries without modifying their originals.
