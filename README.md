# fabric

**Final task: test whether spatial geometry emerges from a purely relational
representation of a physical system.**

## Hypothesis

The Cartesian coordinates of bodies in a physical system are not primitive.
They are a *derived* description of something more fundamental: the web of
pairwise relations — forces, couplings, correlations — between those bodies.

If this is right, then a model that only ever sees pairwise *relational*
quantities (never positions) should still be able to reconstruct the
geometry of the system from what it has learned about those relations.

## Inspiration, not claim

The spirit of the project echoes the holographic principle in theoretical
physics — the idea that bulk geometry can be reconstructed from data living
only on the boundary. We borrow the *idea*, not the mathematics. Nothing in
this repository proves or depends on AdS/CFT duality; we only note that
"space is a derived quantity of relations" has been taken seriously by
physicists, not only by philosophers. The experiment here is a small,
classical-mechanics probe in that spirit.

## Setup in plain language

1. Run a physics simulation of `N` bodies connected by springs (later:
   gravity, articulated joints).
2. Strip the output: the model receives only quantities that are invariant
   under global rotation and translation — speeds, kinetic energies,
   force magnitudes between pairs. Absolute positions are hidden.
3. Train a two-part model:
   - **A relation-guesser** that decides, from trajectory observations,
     which pairs of bodies are coupled and how strongly. *(In ML
     terminology, an NRI-style encoder with both a discrete edge-type
     head and a continuous edge-weight head.)*
   - **A dynamics predictor** that advances the system one step at a
     time, using only the guessed coupling graph, not positions.
     *(In ML terminology, a Graph Network Simulator decoder.)*
4. The guessed coupling graph is the object of study.

## The emergence test

Take the coupling graph the model has learned. From it, build a matrix
called the graph Laplacian (a standard object in graph theory).
The eigenvectors of this matrix, read row by row, give every body a
position in a low-dimensional space. We call these **emergent coordinates**.

Compare the emergent coordinates to the true positions that were hidden
from the model all along. If they match (after a rotation / scale
alignment — "Procrustes"), then space, in this setup, was indeed a
derived quantity of the relations the model learned.

The hypothesis is supported if:

- Emergent coordinates align with true coordinates up to low error.
- Distances in the emergent space track true distances.
- The sorted eigenvalues of the Laplacian show a clear jump at the true
  dimension of the system — a sign that the graph is sampling a
  low-dimensional manifold, not an arbitrary high-dimensional blob.

## Systems, easiest to hardest

1. **Spring networks** — should give back Euclidean space. Sanity check.
2. **Gravitational N-body** — long-range forces, noisier.
3. **Articulated bodies (Ant / Cheetah)** — the "true space" is not
   Euclidean at all; it is a configuration manifold of joint angles.
   Here we are looking for the right non-Euclidean geometry, not
   a failure to recover R³.

## Failure mode worth preparing for

A dense, structureless coupling graph that yields no clean emergent
geometry. We need the experiments to distinguish "the model failed to
learn" from "this system's relations genuinely do not encode a simple
geometry."

## Status

Scaffolding stage. See:

- `PLAN.md` — milestones and acceptance criteria.
- `INFRA.md` — where code, data, and checkpoints actually live.
- `SUBMISSION.md` — target venues, reading list, required ablations.
