# Coding Plan

Iterative milestones. Each milestone ends with something runnable.

## Vocabulary

A one-line gloss per term so the rest of the document can use them
without stopping to explain.

- **NRI (Neural Relational Inference).** A model family (Kipf et al.
  2018) that watches trajectories and guesses which pairs of objects
  are interacting. We use "NRI-style encoder" to mean that kind of
  guesser.
- **GNS (Graph Network Simulator).** A model family (Sanchez-Gonzalez
  et al. 2020) that predicts the next physical state by passing
  messages along a graph. We use it as the dynamics half of the model.
- **Gumbel-Softmax.** A trick that lets gradients flow through a step
  where you would normally have to sample a discrete category.
  Needed because the encoder's edge-type head is categorical.
- **Message passing.** The computation inside a graph neural network:
  each edge computes a vector from its two endpoints, each node sums
  the incoming edge vectors and updates itself. Repeat a few times.
- **Laplacian `L = D − A`.** A matrix summary of a graph. `A` is the
  adjacency (edge weights), `D` is the diagonal of node degrees. Its
  eigenvectors encode the shape of the graph.
- **Spectral embedding / Laplacian Eigenmaps.** Take the first few
  non-trivial eigenvectors of `L`; use each node's values across those
  eigenvectors as its coordinates. If the graph was sampled from a
  smooth manifold with heat-kernel edge weights, the coordinates
  recover the manifold (Belkin & Niyogi 2003).
- **Heat kernel.** Edge weight of the form `exp(−distance² / σ²)`.
  The specific shape required for Laplacian Eigenmaps' manifold limit.
- **Spectral gap.** A jump in the sorted eigenvalues of `L`. A large
  gap after the `D`-th eigenvalue is evidence that the graph lives on
  a `D`-dimensional manifold.
- **Procrustes alignment.** Given two point clouds that may differ by
  rotation, reflection, and scale, find the best alignment and return
  the leftover error. Needed because spectral embedding has arbitrary
  rotation.
- **Rollout.** Run the model in "free" mode — feed its own predictions
  back as input for `K` steps. Exposes compounding error that a
  one-step MSE hides.
- **Permutation equivariance.** Reordering the input nodes reorders
  the output the same way but changes nothing else. A sanity property
  for any graph model.

## Stack

- **Data / simulation:**
  - Springs, gravity → custom JAX simulator. Simple enough that wrapping
    Brax is overkill; `jax.vmap + jax.jit` gives thousands of parallel
    trajectories.
  - Ant / Cheetah → Brax's built-in envs (MuJoCo-compatible MJX
    backend). Only pulled in at M6.
- **Model / training:** PyTorch. NRI and GNS reference implementations
  are PyTorch; debugging cycle is faster.
- **Interface:** flat `.npz` shards on disk. JAX process writes and
  exits; PyTorch process reads. Neither framework imports the other.
  Two separate venvs on AutoDL — see `INFRA.md`.

## M0 — Skeleton (0.5 day)

**Prereq:** read the 5–6 papers in `SUBMISSION.md` before freezing the
architecture. Two goals: don't reinvent, and pick baselines from the
literature that reviewers will expect.

- [ ] `pyproject.toml`
  - core: `numpy`, `torch`, `pyyaml`, `tqdm`, `pytest`
  - sim extra `[sim]`: `jax[cuda12]`, `brax`
  - dev: `ruff`, `pytest-cov`
- [ ] Package layout:
  ```
  fabric/
    data/        # simulators + dataset wrappers
    models/      # encoder, decoder, nri
    train/       # train loop, losses, schedules
    eval/        # spectral embedding, Procrustes, metrics
    utils/       # logging, config, seeding
  configs/       # YAML per experiment
  scripts/       # generate_data.py, train.py, verify_emergence.py
  tests/         # unit tests
  ```
- [ ] `Makefile` targets: `data`, `train`, `verify`, `test`, `lint`
- [ ] CI-light: `pytest -q` on a tiny synthetic graph

## M1 — Spring-system data (1–2 days)

- [ ] `fabric/data/springs.py`: JAX-native N-body spring simulator.
  - `step(state) -> state` is a pure function (semi-implicit Euler,
    `dt = 1e-3`, output stride `dt_out = 1e-2`).
  - `rollout = jax.jit(jax.lax.scan(step, ...))` for a single trajectory.
  - `batched_rollout = jax.vmap(rollout)` to generate `B = 1024`
    independent trajectories per call on one 4090.
  - Connectivity: Erdős–Rényi `p = 0.3` per trajectory, sampled from a
    PRNGKey so the spring topology varies across the dataset.
  - Brax is the path for M6 (Ant / Cheetah). For springs the custom
    JAX simulator is simpler than wrangling Brax's positional backend;
    keep the door open by matching Brax's `(positions, velocities)`
    state convention.
- [ ] Per timestep emit:
  - `node_features`: `[N, F_n]` — `|v_i|`, kinetic energy, mass.
    Rotation- and translation-invariant scalars only.
  - `edge_features`: `[N, N, F_e]` — `|v_i − v_j|`, `|F_ij|`
    (force magnitude). `|F_ij|` is not used by the main model path;
    it is recorded so that a sanity-check script (see M5) can compare
    our learned-graph embedding against a purely physics-based edge
    weight. Stored for every pair every step, including
    "disconnected" pairs.
  - `node_targets`: next-step `node_features`.
  - **withheld** `positions`: `[N, D]` — saved separately, never read
    by the model.
- [ ] `fabric/data/features.py` — the **only** module that touches
      positions. Builds the no-position node/edge tensors and writes
      `positions` to a separate `_withheld/` subdirectory whose loader
      raises by default.
- [ ] `fabric/data/dataset.py`: torch `Dataset`, trajectory windowing
      (`T = 49`, predict `T + 1`), 70/15/15 split by whole trajectory.
      Sharded `.npz` on disk.
- [ ] `scripts/generate_data.py`: config-driven. Two presets:
      - `springs_smoke`: `N = 5`, 200 trajectories × 100 steps. For
        unit tests and smoke training. Spectral embedding is too noisy
        at N=5 to say anything about emergence.
      - `springs_small`: `N = 20`, 1000 trajectories × 100 steps. This
        is the size the M5 acceptance bars are calibrated for.
      JAX runs here; `.npz` output is framework-neutral.
- [ ] `tests/test_springs_invariants.py`: energy conservation to 1e-3
      (catches silent simulator bugs).
- [ ] `tests/test_no_position_leak.py`: **rotation/translation
      invariance test.** Generate two datasets from the same seed but
      with the initial positions rotated + translated by a random rigid
      motion. Every tensor the model will see must be **byte-identical**
      between the two. If any feature changes, it is encoding absolute
      coordinates and must be removed. This is stricter than a
      correlation threshold and directly enforces the "no position"
      promise. **Write this before anything else in M1 — the project's
      scientific fuse.**

**Decision log entry needed:** exact list of node/edge features. Document
in `configs/springs.yaml`.

## M2 — NRI encoder (2–3 days)

**Vanilla NRI, single head.** We implement the encoder as in the
original paper — no continuous edge-weight head, no heat-kernel
parameterization, no geometric regularizer. If vanilla is enough to
make the emergence test work, we publish that. If it isn't, M7 below
adds complexity *in response to specific failures*, which is a
cleaner story for a paper anyway.

- [ ] `fabric/models/encoder.py`:
  - Node MLP → edge MLP (concat endpoints) → temporal aggregator
    (mean over window or 1D conv) → edge logits `[N, N, K]`.
  - Gumbel-Softmax samples a `K`-categorical `z_ij` per pair.
- [ ] `fabric/models/gumbel.py`: Gumbel-Softmax with annealed `tau`.
- [ ] Unit tests:
  - encoder is permutation-equivariant in node ordering
  - `z_ij` is a valid distribution along the type axis
  - as `tau → 0`, samples are hard one-hots

## M3 — GNS decoder (2–3 days)

- [ ] `fabric/models/decoder.py`:
  - Edge update: `m_ij = sum_k z_ij^k · MLP_k(h_i, h_j)` — one MLP per
    relation type, mixture weighted by `z_ij`.
  - Node update: `h_i' = MLP(h_i, sum_j m_ij)`.
  - `M = 2` message-passing steps, then a head predicting
    next `node_features`.
- [ ] `fabric/models/nri.py`: wires encoder + decoder.
      `forward(trajectory) -> {pred, z, z_logits}`.

## M4 — Training (2 days)

Three losses, nothing exotic.

- [ ] `fabric/train/losses.py`:
  - **Reconstruction** — MSE on predicted node features (one step).
  - **Rollout** — free-run `K` steps, discounted MSE.
  - **Relation KL** — KL of `z_ij` distribution to a sparse prior
    (e.g. `Cat([1−ε, ε/(K−1), …])`). Prevents uniform collapse.
- [ ] `fabric/train/trainer.py`: AMP, grad clip, cosine schedule,
      checkpoint best-val.
- [ ] `scripts/train.py` config-driven. Logs: loss components, edge
      entropy, relation-type usage histogram.
- [ ] Smoke test: 5-particle springs, single GPU, < 30 min to a sane
      reconstruction loss.

## M5 — Emergence verification (3–4 days)

The scientific payload. **One main experiment, one cheap sanity
check.** Acceptance bars assume `springs_small` (N=20); bars do not
apply to `springs_smoke` — N=5 is too small for spectral embedding to
mean anything.

### Shared tooling

- [ ] `fabric/eval/graph.py`: build adjacency `A` from `z` — for
      `K = 2` with types (none, connected), `A_ij = P(z_ij = connected)`.
      Laplacian `L = D − A` and a normalized variant.
- [ ] `fabric/eval/spectral.py`: top-`k` eigenpairs of `L`, return
      `eigenvectors[:, 1:1+D]` as coordinates (skip the trivial mode).
- [ ] `fabric/eval/procrustes.py`: orthogonal Procrustes alignment to
      the withheld ground-truth positions; report RMSE and per-axis
      variance.
- [ ] `fabric/eval/geometry.py`:
  - distance-distance correlation (emergent vs true, Spearman)
  - spectral-gap diagnostic
  - relation-strength vs true-distance correlation

### Main experiment — emergence from the learned graph

Run spectral embedding on `A` built from the encoder's `z`, compare to
withheld positions.

**Acceptance (springs, N=20):**

- Procrustes RMSE ≤ 15% of system diameter.
- Spearman(emergent dist, true dist) ≥ 0.8.
- Visible spectral gap after the `D`-th eigenvalue.
- Shuffle-`z` ablation (permute `z` at eval time): rollout MSE
  increases by ≥ 5×. If it doesn't, the decoder ignored the graph and
  any emergence reading is vacuous.

The bars are deliberately slack versus the earlier plan — vanilla
NRI+GNS has not been engineered for this test, so we should accept
"roughly recovered geometry" as a pass. Tightening comes after M7 if
escalation is needed.

### Sanity baseline (half day, independent)

`scripts/sanity_baseline.py`: take the physical edge feature `|F_ij|`,
average over time, use `exp(-α · mean_t |F_ij|)` as edge weight,
run the same spectral embedding + Procrustes pipeline. No model
training.

Purpose: **smoke-test the eval code.** If this baseline fails to
recover geometry, our `eval/*` has a bug or the simulator is broken —
fix before trusting any main-experiment numbers. If it succeeds, we
have an informal upper bound on what the learned graph could achieve.

This is infrastructure, not a headline experiment. It lives in
`scripts/`, not `fabric/eval/`.

## M6 — Harder systems (1+ week, conditional)

Only if M5 passes on springs:

- [ ] Gravitational N-body (`fabric/data/gravity.py`). Run the main
      M5 experiment on this dataset. Expect degraded Procrustes — long-
      range forces do not give heat-kernel-shaped graphs. Report honestly.
- [ ] Articulated bodies (Ant / Cheetah) are **deferred** until gravity
      is understood. If both springs and gravity work, a new milestone
      is added here for the non-Euclidean experiment.

## M7 — Escalation, only if M5 fails on springs

Failure-driven. Vanilla NRI+GNS may not be enough; if so, we add
complexity **in response to a specific observed failure**, not
preemptively. Each row of the table below names a failure mode, the
minimum addition that typically fixes it, and where it lands in the
code.

| Observed failure | Minimum addition | Where |
|---|---|---|
| `z` collapses to a single relation type | stronger sparse KL prior; slower `tau` anneal; entropy regularizer | M4 |
| Shuffle-`z` ablation does not hurt rollout | decoder ignores graph — gate messages by `z`, or raise `M` message-passing steps | M3 |
| `z` informative but embedding is noisy / geometry-free | add a continuous edge-weight head, parameterized as `w_ij = exp(-softplus(MLP))` (heat-kernel shape) | M2 |
| Continuous head learns an arbitrary shape | add the heat-kernel regularizer `L_heat = MSE(w_ij, exp(-α |F_ij|))` | M4 |
| All of the above fail | architecture is wrong — swap in a continuous Gaussian-latent graph, sparsity-inducing attention, or a different relational backbone | M2 |

**Decision rule.** Add the minimum that unblocks the next acceptance
bar. Do not stack additions preemptively — each one makes the paper's
ablation story harder.

**Paper framing.** A failure → escalation narrative reads better than
"we designed something complex and it worked." Reviewers trust the
former more.

## M8 — Write-up / submission (open-ended)

- [ ] `RESULTS.md`: plots, tables, ablations.
- [ ] Decide submission target: benchmark / leaderboard vs workshop
      paper (see `SUBMISSION.md` for venue ranking).
- [ ] For a paper path: figures via `scripts/figures/`, draft in a
      separate `paper/` tree, do not pollute `fabric/` with LaTeX.

---

## Cross-cutting decisions

1. **Framework split (locked).** JAX (custom) for springs / gravity,
   Brax for articulated bodies, PyTorch for model and training.
   Interface is `.npz` on disk.
2. **Vanilla first (locked).** M2–M5 implement plain NRI+GNS from the
   papers. Architectural additions are failure-driven, per the M7
   table — not front-loaded into the initial build.
3. **Number of relation types `K`.** Start `K = 2` for springs
   (connected / not). Sweep `K ∈ {2, 4, 8}` later.
4. **Embedding dimension `D` for the spectral test.** Match the true
   spatial dimension of the simulated system (2 or 3).
5. **What counts as "no position information".** Documented in
   `configs/*.yaml`. `|F_ij|` is allowed because it is a scalar
   invariant under global rotation and translation. Absolute or
   relative position vectors are not allowed. Enforced by
   `test_no_position_leak.py`.

## Risks tracked here, not in README

- **Vanilla NRI+GNS may fail the emergence test.** Expected — NRI
  optimizes trajectory prediction, not geometry. The response is M7
  escalation, *not* reopening M2 in a panic. Every escalation step is
  a data point for the paper's ablation story.
- **Encoder collapses** to a single relation type → stronger KL prior,
  entropy regularizer, slower `tau` anneal. Monitored via type-usage
  histogram logged in M4.
- **Decoder ignores the graph** → shuffle-`z` ablation in M5 catches
  this. If rollout MSE does not increase when `z` is permuted, the
  emergence claim is vacuous regardless of the spectral numbers.
- **Spectral embedding rotates between runs** → expected, Procrustes
  handles it; do not report un-aligned numbers.
- **Non-Euclidean true geometry** (articulated bodies) → only surfaces
  in M6+ (Ant). Do not declare failure on Procrustes-to-R³ there; that
  milestone is deliberately deferred until we understand springs and
  gravity first.
- **Data gen seed drift** → `make data` must be byte-reproducible. Add
  `tests/test_data_determinism.py` that re-runs generation and compares
  hashes.
