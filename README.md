# BLUR FOSP curvature experiment

This repository implements the four-stage experiment in the prompt. It is an
instrumented geometric test, not a claim of state-of-the-art unlearning quality.

The central distinction is simple: a small first derivative only says that the
forget objective is locally flat to first order. At a strict saddle, it can still
have a direction `v` with `v^T H_g v < 0`. A sufficiently small, correctly oriented
step along that direction lowers the forget objective even when `||grad g||` is
small. Such a checkpoint is a FOSP but not a second-order stationary point (SOSP).

## What is implemented

- `train_blur.py`: LoRA-only BLUR-NPO updates, periodic adapter checkpoints, and
  first-order statistics.
- `analyze_curvature.py`: matrix-free autograd HVPs, reorthogonalized Lanczos,
  minimum Ritz eigenpairs, automatic failure selection, CSV/JSON output, and the
  three requested plots.
- `negative_escape.py`: oriented minimum-curvature perturbations at all requested
  alpha values and before/after forget-loss measurements.
- `utils/hessian.py`: the reusable `compute_hvp`, `estimate_min_eigenvalue`,
  `flatten_parameters`, and `assign_parameters` functions.

The NPO reference policy is the frozen base model obtained by disabling its LoRA
adapter. Consequently, the run does not need a second copy of the 3B model. All
gradients, projections, Hessian-vector products, and perturbations include only
parameters with `requires_grad=True`—the LoRA subspace.

## Run

Use a CUDA machine with enough memory, authenticate with Hugging Face if Llama is
gated, and adjust `config.yaml` if necessary:

```bash
cd project
python train_blur.py --config config.yaml
python analyze_curvature.py --config config.yaml
python negative_escape.py --config config.yaml
```

Outputs are written to `results/`:

- `fosp_statistics.json` and `fosp_statistics.csv`
- `failure_summary.json`
- `failures/checkpoint_*/failure_checkpoint.pt`
- `negative_curvature_results.json` and `negative_escape.csv`
- `forget_grad_norm_vs_step.png`, `lambda_min_vs_step.png`, and
  `gradient_vs_curvature.png`

Run numerical unit tests with:

```bash
python -m unittest discover -s tests
```

## Experimental controls and interpretation

`analysis.num_loss_batches` defines one deterministic prefix of TOFU used for
every checkpoint and for the escape evaluation. This consistency is essential:
mixing stochastic minibatches can manufacture an apparent gradient/curvature gap.
Increase it for the final paper run and report the number of examples. Likewise,
increase `num_lanczos_steps` until the minimum Ritz value is stable across seeds.

The default failure rule is exactly the requested
`forget_grad_norm < 1e-3` and `lambda_min_Hg < -0.05`. These are scale-dependent
quantities, so a serious result should include threshold sensitivity and multiple
training/data/Lanczos seeds. The escape script orients the sign-ambiguous Ritz
vector so `grad(g)^T v <= 0`; negative curvature then supplies the second-order
decrease. Since extremely tiny loss differences can be near floating-point noise,
the strongest evidence is a consistent decrease across several alpha values and
seeds, ideally with a held-out forget subset as an additional evaluation.

Quantized loading is optional. Unquantized bf16 is the default because it has the
cleanest double-autograd behavior. If memory requires `quantization: 4bit`, install
bitsandbytes and verify the HVP test plus a small end-to-end run on the exact GPU
software stack before collecting results.

Keep `attn_implementation: eager` during curvature analysis. Several fused
SDPA/FlashAttention kernels implement first-order backward but not the derivative
of backward required for an HVP; using them can fail or silently make the setup
software-stack dependent.
