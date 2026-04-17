# fabric

**Final task: test whether spatial geometry emerges from a purely relational
representation of a physical system.**

## Hypothesis

The Cartesian coordinates of bodies in a physical system are not primitive.
They are an *emergent* description of a more fundamental object: the graph of
pairwise relations (forces, interactions, correlations) between the bodies.
If this is true, a model trained only on local node features and pairwise
*relational* observations — never on absolute positions — should still recover
the geometry of the system as a derived quantity of its learned relation graph.

## Setup

We train a Neural Relational Inference (NRI) style model on multi-body
simulations from Brax / MuJoCo. The model has two halves:

1. **Relation encoder.** Given a trajectory of node-local features and
   pairwise relational features (relative velocity, contact force, …),
   infer a discrete latent relation graph `z` over node pairs via
   Gumbel-Softmax.
2. **Relational dynamics decoder.** A Graph Network Simulator (GNS) that
   does message passing on `z` to predict the next-step node state.

Absolute coordinates are deliberately withheld from both encoder and decoder.
They are kept only as a held-out ground truth for the emergence test.

## Emergence test

After training, the learned relation graph defines a Laplacian
`L = D − A`. We take the leading non-trivial eigenvectors of `L` as
**emergent coordinates** of the nodes, and compare them to the withheld
ground-truth coordinates after Procrustes alignment.

The hypothesis is supported if:

- Procrustes-aligned reconstruction error is small.
- Pairwise distances in the emergent embedding correlate with true distances.
- The Laplacian spectrum shows a clean spectral gap consistent with a
  low-dimensional manifold.

## Stages

1. **Spring systems** — should emerge as Euclidean. Sanity check.
2. **Gravitational N-body** — non-uniform interaction strength.
3. **Articulated bodies (MuJoCo Ant / Cheetah)** — constraints and contacts.

## Failure mode of interest

A dense, structureless relation graph that yields no clean spectral
embedding. We need experiments that distinguish "the model failed to learn"
from "this system's relational structure genuinely is non-Euclidean."

## Status

Scaffolding stage. See `PLAN.md` for the coding plan.
