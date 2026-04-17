# Coding Plan

Iterative milestones. Each milestone ends with something runnable.

## M0 — Skeleton (0.5 day)

- [ ] `pyproject.toml` (PyTorch path; JAX/Brax optional extra)
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

- [ ] `fabric/data/springs.py`: deterministic N-body spring simulator
      (start in NumPy; port to Brax later if data generation is the
      bottleneck).
- [ ] Per timestep emit:
  - `node_features`: `[N, F_n]` — velocity magnitude, KE, mass
    (no positions, no absolute reference frame)
  - `edge_features`: `[N, N, F_e]` — relative speed, instantaneous
    interaction magnitude (no relative position vectors)
  - `node_targets`: next-step `node_features`
  - **withheld** `positions`: `[N, D]` — saved separately, never read
    by the model
- [ ] `fabric/data/dataset.py`: trajectory windowing, train/val/test split,
      sharded `.npz` on disk.
- [ ] `scripts/generate_data.py` + a `tests/test_springs_invariants.py`
      checking energy conservation to catch silent simulator bugs.

**Decision log entry needed:** exact list of node/edge features. Document
in `configs/springs.yaml`.

## M2 — NRI encoder (3–4 days)

- [ ] `fabric/models/encoder.py`:
  - Node MLP → edge MLP (concat endpoints) → temporal aggregator
    (mean over window or 1D conv) → edge logits
    `[N, N, K]` for `K` relation types.
- [ ] `fabric/models/gumbel.py`: Gumbel-Softmax with annealed `tau`.
- [ ] Unit test: encoder is permutation-equivariant in node ordering.

## M3 — GNS decoder (3–4 days)

- [ ] `fabric/models/decoder.py`:
  - Edge update: `m_ij = MLP_k(h_i, h_j)` selected by relation type `z_ij`
  - Node update: `h_i' = MLP(h_i, sum_j m_ij)`
  - K message-passing steps, then a head predicting next `node_features`.
- [ ] `fabric/models/nri.py`: wires encoder + decoder; exposes
      `forward(trajectory) -> (pred, z, edge_logits)`.

## M4 — Training (2–3 days)

- [ ] `fabric/train/losses.py`:
  - one-step reconstruction (MSE on node features)
  - KL to a uniform / sparse prior on edge categorical
  - rollout loss: free-run K steps, MSE accumulated with a discount
- [ ] `fabric/train/trainer.py`: AMP, grad clip, cosine schedule, checkpoint
      best-val.
- [ ] `scripts/train.py` driven by a config; logs loss, KL, edge entropy,
      relation-type usage histogram.
- [ ] Smoke test: 5-particle springs, single GPU, < 30 min to a sane
      reconstruction loss.

## M5 — Emergence verification (3–5 days)

This is the scientific payload. Implement carefully.

- [ ] `fabric/eval/graph.py`: build `A` from `z` (mean over a held-out
      batch), define `L = D − A` (and a normalized variant).
- [ ] `fabric/eval/spectral.py`: top-`k` eigenpairs of `L`, return
      coordinates `eigenvectors[:, 1:1+D]` (skip the trivial mode).
- [ ] `fabric/eval/procrustes.py`: orthogonal Procrustes alignment to the
      withheld ground-truth positions; report RMSE and per-axis variance.
- [ ] `fabric/eval/geometry.py`:
  - distance-distance correlation (emergent vs true, Spearman)
  - spectral-gap diagnostic
  - relation-strength vs true-distance correlation
- [ ] `scripts/verify_emergence.py`: one CLI that runs all of the above
      against a checkpoint and prints a pass/fail table.

**Acceptance bar for the spring system:**

- Procrustes RMSE ≤ 10% of system diameter.
- Spearman(emergent dist, true dist) ≥ 0.9.
- Visible spectral gap after the `D`-th eigenvalue.

## M6 — Harder systems (1+ week, conditional)

Only after M5 passes on springs:

- [ ] Gravitational N-body (`fabric/data/gravity.py`).
- [ ] MuJoCo Ant / Cheetah wrapper (joint-relative observations only).
- [ ] Per-system `configs/*.yaml` and a results table in `RESULTS.md`.

## M7 — Write-up / submission (open-ended)

- [ ] `RESULTS.md`: plots, tables, ablations.
- [ ] Decide path A (benchmark) vs path B (workshop paper).
- [ ] If B: figures via `scripts/figures/`, draft in a separate `paper/`
      tree, do not pollute `fabric/` with LaTeX.

---

## Cross-cutting decisions to make before M2

1. **Framework.** PyTorch (faster to iterate, easier debugging) vs JAX
   (faster data + cleaner vmap over relation types). Default: PyTorch
   for the model, NumPy for the spring simulator. Revisit only if data
   generation is slow.
2. **Number of relation types `K`.** Start `K = 2` for springs
   (connected / not). Sweep `K ∈ {2, 4, 8}` later.
3. **Embedding dimension `D` for the spectral test.** Match the true
   spatial dimension of the simulated system (2 or 3) for the
   acceptance check; report `D = 1..6` for the spectral-gap plot.
4. **What counts as "no position information".** Document the exact
   feature list in the config; have a unit test that asserts no
   absolute-coordinate field leaks into model input tensors.

## Risks tracked here, not in README

- Encoder collapses to a single relation type → fix with KL prior /
  entropy regularizer / `tau` schedule.
- Decoder learns the dynamics without using `z` → ablation: zero out
  `z` at eval and check rollout degrades.
- Spectral embedding rotates between runs → expected; Procrustes handles it.
- Non-Euclidean true geometry (e.g. articulated bodies on a manifold)
  → distinguish via the geometry diagnostics in M5 before declaring failure.
