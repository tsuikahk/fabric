# Submission Notes

Meta-decisions for the paper / submission side of the project. Not for
code. Updated as we learn more.

## Target venues (ranked by fit)

| Venue | Type | Deadline | Notes |
|---|---|---|---|
| **NeurReps** (NeurIPS workshop) | Workshop | ~Sep | "Symmetry and Geometry in Neural Representations" — exact topical fit |
| **GRaM** (ICML workshop) | Workshop | ~May | "Geometry-grounded Representation Learning" — also exact fit |
| **ICLR** main | Main conf | ~Oct | If springs + gravity + Ant all pass M5 |
| **NeurIPS** main | Main conf | ~May | Fallback to main; broader audience |

Strategy: aim workshop first (NeurReps most likely), use reviewer
feedback to decide whether to extend to main conference.

Workshops are **non-archival** — a workshop paper can be extended and
resubmitted to a main conference. Double-submission rules apply; check
each venue.

## Acceptance bar by result

- Springs M5 passes only → workshop paper, honest scope.
- Springs + gravitational N-body pass → workshop strong, main conf short paper possible.
- Springs + gravity + Ant pass → main conf competitive.
- + theory (when does spectral embedding equal true metric?) → strong main conf.

## Reading list (M0 prerequisite)

Read before architecture is frozen. Goal: don't reinvent, learn the
vocabulary, pick the right baselines.

- [ ] Kipf et al. 2018, *Neural Relational Inference for Interacting Systems*
- [ ] Sanchez-Gonzalez et al. 2020, *Learning to Simulate Complex Physics with Graph Networks* (GNS)
- [ ] Battaglia et al. 2018, *Relational inductive biases, deep learning, and graph networks*
- [ ] Cranmer et al. 2020, *Discovering Symbolic Models from Deep Learning with Inductive Biases*
- [ ] Belkin & Niyogi 2003, *Laplacian Eigenmaps for Dimensionality Reduction*
- [ ] One of: Hashimoto et al. on AdS/deep learning, or Yi-Zhuang You's holographic ML papers

Add notes per paper as: key claim, what they measure, what they don't
measure that we do.

## Baselines to compare against

- **NRI vanilla** (discrete edge types, no geometry test): we should
  reproduce its trajectory-prediction numbers as a sanity check, then
  add our emergence test to their graph.
- **GNS with ground-truth positions**: upper bound on prediction error.
- **Random edge graph + GNS**: lower bound, shows relation learning matters.
- **Ablation: shuffle z at eval time**: if rollout loss does NOT
  explode, the model ignored the relation graph — emergence claim is
  void.

## Required ablations for paper

1. Shuffle-z ablation (above).
2. `K` sweep: `K ∈ {1, 2, 4, 8}` relation types, does emergence quality track `K`?
3. Edge-weight head on/off: does adding continuous coupling strength
   improve Procrustes RMSE?
4. Graph sparsity prior: KL weight sweep, does a sparser graph give a
   cleaner spectral gap?
5. Feature ablations: remove each node/edge feature, see which are
   necessary for emergence.

## Open questions for the paper

- Under what formal conditions does the leading Laplacian eigenspace
  of the learned graph recover the true metric? (Cite Belkin & Niyogi
  for the manifold limit; what breaks in finite N?)
- Is the holographic framing load-bearing or decorative? If decorative,
  drop it — reviewers punish over-claiming.
- How does emergence quality scale with N? Plot Procrustes RMSE vs N.

## Submission infrastructure

Filled in when we get there.

- arXiv account: TODO
- OpenReview account: TODO
- HuggingFace model hub org: TODO
- Code release: this repo, make a `v1.0-workshop` tag at submission
- Reproducibility: `scripts/reproduce.sh` must regenerate every figure
  from a single `make`

## Author info

- Primary author: (user)
- Affiliation: TBD
- Contact: TBD
