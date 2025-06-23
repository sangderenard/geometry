# vecdec

A lightweight, vectorized Discrete Exterior Calculus (DEC) toolkit in C++ using Eigen.

Designed for real-world scientific computation, graphics, beam physics, and simulation.

## Philosophy
- Vectorized, modern C++
- Eigen-first, no STL-heavy overhead
- GPU extension ready
- Test-driven

## Timeline
The repository has steadily grown from simple node utilities into a flexible
geometry engine.  Major milestones include:

- **Initial scaffold** – base data types and early DAG/DEC sketches.
- **Node linking utilities** – bi-directional relationships and activation
  metadata for graph traversal.
- **Graph operations** – generic `GraphOps` interface and geneology search/sort
  capabilities.
- **Neural network and DAG** – manifest structures hinting at delayed
  computation graphs.
- **Locking guidelines** – concurrency primitives and lock bank design.
- **Operator declarations** – feature-wise arithmetic on nodes and set
  operations on geneologies.
- **Emergent stencil vision** – establishing expectations for scalable physics
  modeling.

## Module Overview
The code base is organised into small, self‑contained headers.  Below is an
outline of the main pieces, their pure mathematical roots and practical roles.

| Module | Maths background | Layperson description |
|-------|-----------------|-----------------------|
| `utils.h` | Graph theory & set theory | Defines the `Node` structure with links, features and exposures, supporting traversal and set‑like operations. |
| `graph_ops.h` | Graph algorithms, category theory | Abstracts common graph operations (push/pop, slicing, search) across containers like genealogies or DAGs. |
| `dag.h`/`dag.c` | Directed acyclic graphs | Provides nodes for computation graphs, forming the basis of delayed transforms. |
| `neuralnetwork` structures | Functional analysis on graphs | Skeleton for connecting DAG levels into trainable networks. |
| `parametric_domain.h` | Topology & manifolds | Describes continuous domains via parameter ranges, periodicity and boundary conditions, enabling analytic shapes. |
| `granular_domain.hpp` | Measure theory | Placeholder for discretisations of space with varying resolution. |
| `mesh.hpp` & `meshgrid.hpp` | Discrete geometry | Mesh containers and grid generators for DEC operations. |
| `metric_tensor.hpp` | Differential geometry | Tools for defining metrics on manifolds. |
| `relational_vector.h` | Graph metrics | Compute shortest paths through relational graphs and project them into continuous space. |
| `modspace.h` | Analysis | Generates modulated parameter ranges (non‑linear spacing). |

These components combine to express shapes common in engineering (lines,
surfaces, volumes) and arbitrary parametrised forms.  They are designed to work
with future DEC kernels so that continuous descriptions can be discretised on
demand.

## Mathematical Perspective
At its heart, the project treats geometry as a collection of sets and relations:
graphs express connectivity, while parameter spaces model continuous manifolds.
By keeping everything vectorised, these abstractions scale to large simulations
or learning tasks.  The delayed computation graphs allow transformations to be
scheduled and optimised before execution, bridging discrete mesh operations with
continuous formulas.
