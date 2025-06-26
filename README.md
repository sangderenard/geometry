# Guardian Geometry Engine

A foundational, high-performance C++ engine for vectorized geometry processing and scientific computation, built on the principles of Discrete Exterior Calculus (DEC).

Designed from the ground up for demanding applications in simulation, computational physics, computer graphics, and machine learning, this library provides the intricate machinery needed to model complex systems.

## Design Philosophy
The engine is built on a set of core principles that prioritize performance, mathematical rigor, and future-readiness.

- **Vectorization & Performance:** At its core, the library is designed for vectorization. By leveraging Eigen for tensor operations and enforcing a vectorized-first approach, it achieves the performance required for large-scale simulations and data processing tasks.

- **Mathematical Abstraction:** We treat geometry as a collection of sets and relations. Graphs express connectivity, while parametric domains model continuous manifolds. This allows complex physical and abstract systems to be described with mathematical clarity.

- **Delayed Computation:** The engine is built around a Directed Acyclic Graph (DAG) that enables delayed computation. Transformations and operations are scheduled and optimized as a complete pipeline before execution, bridging the gap between discrete mesh operations and continuous analytical formulas.

- **Generalized Operators:** The API moves beyond simple grid-based methods. It supports generalized, n-dimensional **interaction fields** and **relational vector spaces**, allowing for the creation of meshless, adaptive, or emergent numerical methods suitable for highly non-Euclidean or abstract problem spaces.

- **Graph-Native Signal Processing:** The engine includes a library of graph kernels and a sophisticated spline interpolation system. This allows for advanced, graph-native signal processing tasks like feature diffusion, smoothing, and data reconstruction directly on the geometric structures, bypassing the need to project to a regular grid.

- **Extensibility & Agent Readiness:** With clear modular boundaries and composable, mesh-agnostic operations, the architecture is designed for extensibility. It anticipates integration with autonomous agents and future hardware like GPUs, providing a robust platform for next-generation computational tools.

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

## Architecture
The library is structured in layers, from low-level backend execution to a high-level user-facing API. This design promotes modularity and allows different parts of the system to be developed and optimized independently.

```text
┌──────────────────────────────────────────┐
│  Application / User Code                 │
│  • Maximum efficiency in complex math    │
|  * Advanced differential rendering       │
│  • Numerous convenience utilities        │
|  * Built-in learning                     |
|  * Graph-native performance              |
└──────────────────────────────────────────┘
                 ▲
                 │
┌──────────────────────────────────────────┐
│  High-Level API Layer                    │
│  • Guardian dynamic primitives           │
|  * Cross-domain utilities                │
|  * Efficient rendering engine            │
│  • Responsive and interactive geometries │
│  • Classic physics simulations vectorized│
└──────────────────────────────────────────┘
                 ▲
                 │
┌───────────────  ───────────────────────────┐
│  Core Domain Modules                       │
|  * guardian_environment.h/c  (Delocalizing)|
│  • utils.h/c           (Dynamic Primitives)│
│  • parametric_domain.h/c (Lossless Domains)│
|  * parametric_transform.h/c (Continuous Geometry)
|  * dag.h/c (Wrap Counting Directional Graphs)
│  • graph_ops.h/c                (graph API)│
|  * metric_tensor.h/c       (N-D Capability)│
│  • laplace_beltrami.h/c      (Core Physics)│
|  * eigensolver.h/c (Eigenmode Decomposition)
│  • kernels.h/c              (Graph Kernels)│
│  • interpolator.h/c  (Reticulating Splines)│
│  • differentiator.h/c          (DEC+engine)│
|  * double_buffer.h/c (generic db byte arrays)
|  * diff_print.h/c (differential buffer updating)
└────────────────  ──────────────────────────┘
                 ▲
                 │
┌───────────────    ───────────────────────────┐
│  Backend Execution Layer                     │
│  • Eigen tensor broadcasts                   │
│  • ONNX graph executor                       │
│  • Batch deployed OpenGL/GLSL compute shaders│
└───────────────────    ───────────────────────┘
```

## Module Overview
The code base is organised into small, self‑contained headers.  Below is an
outline of the main pieces, their pure mathematical roots and practical roles.

| Module | Maths background | Layperson description |
|---------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `guardian_platform.h`     | Operating Systems               | A clean abstraction layer over the host OS, providing portable interfaces for threading, mutexes, file I/O, and high-precision timing.                             |
| `guardian_renderer.h`     | Computer Graphics               | An extensible rendering frontend for visualizing models, with backends for both terminal ASCII art and graphical windows via OpenGL.                               |
| `utils.h`                 | Graph theory & set theory       | Core graph primitives defining the `Node` as a fundamental unit of data and connectivity, with support for feature vectors and relational links.                   |
| `graph_ops.h`             | Graph algorithms, category theory | Abstracts common graph operations (push/pop, slicing, search) across containers like genealogies or DAGs.                                                        |
| `dag.h`/`dag.c`           | Directed acyclic graphs         | Implements the Directed Acyclic Graph (DAG) for building deferred computation pipelines, essential for automatic differentiation and complex model execution.      |
| `parametric_domain.h`     | Topology & manifolds            | A sophisticated engine for defining N-dimensional manifolds with fine-grained control over boundaries (inclusive, exclusive, infinite), periodicity, and discontinuities. |
| `granular_domain.hpp`     | Measure theory                  | Placeholder for discretisations of space with varying resolution.                                                                                              |
| `mesh.hpp` & `meshgrid.hpp` | Discrete geometry               | Mesh containers and grid generators for DEC operations.                                                                                                        |
| `laplace_beltrami.h/cpp`  | Differential Geometry           | Implements the Laplace-Beltrami operator, the cornerstone of the physics engine. It provides the geometric foundation for simulating fields and dynamics on curved manifolds. |
| `kernels.h/cpp`           | Functional Analysis             | A library of graph-native kernels (e.g., heat, Gaussian) for performing convolution, diffusion, and feature extraction directly on geometric graphs.              |
| `interpolator.h/cpp`      | Spline Theory, Numerical Analysis | A sophisticated interpolation engine for "reticulating splines," enabling smooth reconstruction of data and fields across discrete points on a manifold or graph. |
| `modspace.h`              | Analysis                        | Generates modulated parameter ranges (non‑linear spacing).                                                                                                     |
| `stencil.h`               | Numerical methods, field theory | A powerful API for defining local operators. Supports not just classical grid stencils but generalized, n-dimensional interaction fields for emergent numerical methods. |

These components combine to express shapes common in engineering (lines,
surfaces, volumes) and arbitrary parametrised forms.  They are designed to work
with future DEC kernels so that continuous descriptions can be discretised on
demand.

## Contributing
This project is a platform for next-generation geometry, simulation, and learning systems. Contributors are encouraged to design with the following principles in mind:

- **Embrace Functional & Composable Design:** Prefer pure functions and operations that can be chained together.

- **Think in Vectors:** Ensure new algorithms are vectorized to leverage the core engine's performance.

- **Be Mesh-Agnostic:** Where possible, design operations that support both structured and unstructured data.

- **Leverage Advanced APIs:** Use the stencil and relational APIs to express novel numerical, physical, or learning operations.

- **Document with Context:** Provide both the mathematical background and the practical role of new modules.
