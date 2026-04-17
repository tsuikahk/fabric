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
  - `edge_features`: `[N, N, F_e]` — `|v_i − v_j|`, **`|F_ij|`
    (force magnitude)**. `|F_ij|` is load-bearing: Experiment A in M5
    uses its time-average as the physics-informed edge weight, so this
    feature must be stored for every pair every step, including
    "disconnected" pairs (where it will be 0 or a small number from
    indirect coupling).
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

## M2 — NRI encoder + heat-kernel edge-weight head (3–4 days)

The encoder produces **two heads** per node pair, not one:

1. **Discrete type head** — `K`-categorical, Gumbel-Softmax. NRI flavor.
   Used by the decoder to pick which interaction MLP to apply.
2. **Continuous edge-weight head** — scalar `w_ij ∈ (0, 1]`, structurally
   parameterized as

   ```
   w_ij = exp( - softplus(MLP(h_i, h_j)) )
   ```

   This is the **heat-kernel parameterization**. The exponent is
   non-negative by construction, so `w_ij` is bounded, non-negative,
   and vanishes for weakly-coupled pairs. Laplacian Eigenmaps' manifold
   limit only applies to weights of this shape; we bake the shape in so
   gradient descent cannot learn an arbitrary weight.

Both heads share the same node/edge encoder trunk:

- [ ] `fabric/models/encoder.py`:
  - Node MLP → edge MLP (concat endpoints) → temporal aggregator
    (mean over window or 1D conv) → two parallel output heads producing
    `z_logits: [N, N, K]` and `w_exponent: [N, N]`.
- [ ] `fabric/models/gumbel.py`: Gumbel-Softmax with annealed `tau`
      for the discrete head.
- [ ] `fabric/models/heat_kernel.py`: the `w_ij = exp(-softplus(·))`
      wrapper, plus a diagnostic that logs the distribution of `w` over
      pairs (to catch the mode where it collapses to all-zero or all-one).
- [ ] Unit tests:
  - encoder is permutation-equivariant in node ordering
  - `w_ij` is always in `(0, 1]`
  - `w_ij == w_ji` (the edge head must be symmetric; implement by
    averaging over endpoint orderings or by a symmetric MLP input)

## M3 — GNS decoder (3–4 days)

- [ ] `fabric/models/decoder.py`:
  - Edge update: `m_ij = w_ij · MLP_{z_ij}(h_i, h_j)` — discrete type
    `z_ij` picks which MLP, continuous weight `w_ij` scales the
    message. This couples the two encoder heads: `w_ij` controls *how
    much* a pair interacts, `z_ij` controls *how*.
  - Node update: `h_i' = MLP(h_i, sum_j m_ij)`
  - `M` message-passing steps (default `M = 2`), then a head predicting
    next `node_features`.
- [ ] `fabric/models/nri.py`: wires encoder + decoder. Signature:

      ```
      forward(trajectory, *, mask_w=False, shuffle_z=False)
          -> {pred, z, w, z_logits}
      ```

      `mask_w` zeros the edge weights, `shuffle_z` permutes the
      discrete type assignments per batch. These flags drive the M5
      ablations; they are no-ops during normal training.
- [ ] Unit test: with `mask_w=True`, rollout on any nontrivial
      trajectory has >>1 prediction MSE (sanity: the decoder must
      actually be using `w`).

## M4 — Training (2–3 days)

- [ ] `fabric/train/losses.py`:
  - **Reconstruction** — MSE on predicted node features (one step).
  - **Rollout** — free-run `K` steps, discounted MSE. Engages the
    model's use of `z, w` beyond immediate prediction.
  - **Relation KL** — KL of `z_ij` distribution to a sparse prior
    (e.g. `Cat([1−ε, ε/(K−1), …])`). Prevents uniform collapse.
  - **Heat-kernel regularizer (optional, flagged by config).** Pulls
    the learned `w_ij` toward a heat kernel of a physical edge feature
    `|F_ij|` that is already in `edge_features`:

    ```
    L_heat = MSE( w_ij, exp(-α · |F_ij|) )
    ```

    `α` is a learnable scalar. This is the "path 2" mechanism from the
    design discussion — it does not leak positions (only forces, which
    are already in the allowed features), but it does tie the
    learned graph to a geometrically meaningful one. Ablation in M5
    tests emergence **with and without** this regularizer.
- [ ] `fabric/train/trainer.py`: AMP, grad clip, cosine schedule,
      checkpoint best-val.
- [ ] `scripts/train.py` config-driven. Logs: loss components, edge
      entropy, relation-type usage histogram, `w_ij` distribution
      (mean, variance, fraction above 0.5).
- [ ] Smoke test: 5-particle springs, single GPU, < 30 min to a sane
      reconstruction loss.

## M5 — Emergence verification (3–5 days)

This is the scientific payload. Three experiments, each independently
publishable if it lands. All acceptance bars below assume
`springs_small` (N=20). Bars do not apply to the `springs_smoke`
preset — N=5 is too small for spectral embedding to mean anything.

### Shared tooling

- [ ] `fabric/eval/graph.py`: build `A` from any edge-weight source
      (physical `|F_ij|`, or learned `w_ij`, averaged over a held-out
      batch), define `L = D − A` and a normalized variant.
- [ ] `fabric/eval/spectral.py`: top-`k` eigenpairs of `L`, return
      `eigenvectors[:, 1:1+D]` as coordinates (skip the trivial mode).
- [ ] `fabric/eval/procrustes.py`: orthogonal Procrustes alignment to
      the withheld ground-truth positions; report RMSE and per-axis
      variance.
- [ ] `fabric/eval/geometry.py`:
  - distance-distance correlation (emergent vs true, Spearman)
  - spectral-gap diagnostic
  - relation-strength vs true-distance correlation
- [ ] `fabric/eval/ablations.py`: shuffle-`z`, zero-`w`, random-graph
      control. These feed into every experiment.
- [ ] `scripts/verify_emergence.py`: one CLI that takes an experiment
      id (`A`, `B`, or `C`) and prints a pass/fail table.

### Experiment A — physical-force baseline (must work)

Eval-only, no model training. For each trajectory, take the
time-average of `|F_ij|` over the window, form edge weight
`A_ij = exp(-α · mean_t |F_ij|)` where `α` is picked by minimizing
Procrustes RMSE on a validation split (single scalar grid search —
cheap). Run spectral embedding on the resulting weighted Laplacian,
compare to withheld positions.

**Purpose.** Validate that the pipeline itself (spectral embedding +
Procrustes) recovers geometry when handed a physically-meaningful
graph. If A fails, the bug is in eval code or the simulator, not in
representation learning.

**Acceptance (springs):**

- Procrustes RMSE ≤ 5% of system diameter.
- Spearman(emergent dist, true dist) ≥ 0.95.
- Clean spectral gap after `D`.

### Experiment B — learned-graph emergence (main result)

Run spectral embedding on the encoder's `w_ij` output. Compare to A
(upper bound) and to random-graph (lower bound).

**Purpose.** This is the actual fabric thesis: a model trained only on
no-position features learns a graph whose spectral embedding recovers
space.

**Acceptance (springs):**

- Procrustes RMSE ≤ 10% of system diameter.
- Spearman(emergent dist, true dist) ≥ 0.9.
- Visible spectral gap after the `D`-th eigenvalue.
- Shuffle-`z` ablation: rollout MSE increases by ≥ 5×.
- Zero-`w` ablation: rollout MSE increases by ≥ 5×.
- With-vs-without heat-kernel regularizer: report both numbers. The
  regularized run is "fabric B+"; the unregularized is "fabric B−".
  Both are interesting; B+ is the cleaner story, B− is the stronger
  claim if it still works.

### Experiment C — articulated / non-Euclidean system (scientific bonus)

Only after B passes on springs **and** gravity. Run A and B on a MuJoCo
Ant; the true "space" here is joint configuration space, *not* R³.

**Purpose.** Distinguish "the model failed" from "the system's
geometry is not Euclidean." The emergent embedding should have
dimension matching the Ant's DoF (≈ 8 for joints), not 3.

**Acceptance — deliberately weaker:**

- Spectral gap indicates a well-defined intrinsic dimension.
- Emergent coords correlate with joint angles (Spearman ≥ 0.7 per DoF).
- Emergent geodesic distances respect kinematic constraints (two
  nodes separated by a rigid link are always close regardless of R³
  distance).

Failure to reach Euclidean on Ant is **expected and interesting**, not
a defeat.

## M6 — Harder systems (1+ week, conditional)

Only after Experiments A and B pass on springs:

- [ ] Gravitational N-body (`fabric/data/gravity.py`). Run A + B.
      Expect degraded Procrustes due to long-range tails; quantify.
- [ ] MuJoCo / Brax Ant wrapper (joint-relative observations only).
      Run **Experiment C** — do not expect Euclidean emergence.
- [ ] Per-system `configs/*.yaml` and a results table in `RESULTS.md`.

## M7 — Write-up / submission (open-ended)

- [ ] `RESULTS.md`: plots, tables, ablations.
- [ ] Decide submission target: benchmark / leaderboard vs workshop
      paper (see `SUBMISSION.md` for venue ranking). Do not overload
      "A / B / C" here — those names are reserved for the M5 experiments.
- [ ] For a paper path: figures via `scripts/figures/`, draft in a
      separate `paper/` tree, do not pollute `fabric/` with LaTeX.

---

## Cross-cutting decisions

1. **Framework split (locked).** JAX (custom) for springs / gravity,
   Brax for articulated bodies, PyTorch for model and training.
   Interface is `.npz` on disk.
2. **Two encoder heads (locked).** Discrete `K`-categorical `z_ij`
   and continuous heat-kernel-shaped `w_ij`. See M2.
3. **Number of relation types `K`.** Start `K = 2` for springs
   (connected / not). Sweep `K ∈ {2, 4, 8}` later.
4. **Embedding dimension `D` for the spectral test.** Match the true
   spatial dimension of the simulated system (2 or 3) for the
   acceptance check; report `D = 1..6` for the spectral-gap plot.
   Do **not** fix `D` for Experiment C — read it off the spectral gap.
5. **What counts as "no position information".** Documented in
   `configs/*.yaml`. `|F_ij|` is allowed because force is a function
   of distance *through the physics*, not a coordinate. Absolute or
   relative position vectors are not allowed. Enforced by
   `test_no_position_leak.py`.

## Risks tracked here, not in README

- **The learned edge weight may not be a heat kernel of distance.**
  NRI/GNS optimize prediction, not geometry. Even with a continuous
  `w_ij` head, nothing in the training signal forces `w_ij` to be a
  decreasing function of distance — which is what Laplacian Eigenmaps
  needs. Three defenses, all already in the plan:
  1. Structural: parameterize `w_ij = exp(-softplus(MLP))` so the
     *shape* is heat-kernel-like even if the *content* is learned (M2).
  2. Regularizer: `L_heat` pulls `w_ij` toward `exp(-α|F_ij|)`
     (M4, flagged by config, ablated in M5 as B+ vs B−).
  3. Baseline: Experiment A bypasses learning entirely and uses
     `|F_ij|` directly (M5). If A works and B fails, we know the
     pipeline is sound and the problem is representation learning.

  If all three fail on springs, the architecture is wrong. Candidates
  to swap in: continuous Gaussian-latent graph (relational VAE),
  attention with sparsity prior, or a fully different relational
  backbone (e.g. set transformer). Decision point: after M5 on springs.

- **Encoder collapses** to a single relation type, or `w_ij` collapses
  to a constant → fix with KL prior, entropy regularizer, `tau`
  schedule; monitored via the encoder diagnostics in M2.
- **Spectral embedding rotates between runs** → expected, Procrustes
  handles it; do not report un-aligned numbers.
- **Non-Euclidean true geometry** (articulated bodies) → do not
  declare failure on Procrustes-to-R³. Run Experiment C instead.
- **Data gen seed drift** → `make data` must be byte-reproducible. Add
  a `tests/test_data_determinism.py` that re-runs generation and
  compares hashes.
