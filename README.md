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
- **Iterative solver** – modular engine for extrapolating simulations with
  controllable convergence parameters.
- **Emergence framework** – cellular division and growth parameters for graphs.
- **Auxin diffusion** – plant-inspired environment exploration algorithms.
- **Pidgeon solver** – probabilistic approach to the pigeonhole problem.
- **Synthesizer core** – simple signal generation utilities.

## Architecture
The library is structured in layers, from low-level backend execution to a high-level user-facing API. This design promotes modularity and allows different parts of the system to be developed and optimized independently.

```text
╔══════════════════════════════════════════════════════════════════╗
║ Application / User Code                                          ║
║ • Maximum efficiency in complex math                             ║
║ • Advanced differential rendering                                ║
║ • Numerous convenience utilities                                 ║
║ • Built-in learning                                              ║
║ • Graph-native performance                                       ║
║ • hull - ASCII/2D/3D hull editor and visualizer                  ║
║ • oscilliscope - console or 2D waveform viewer                   ║
║ • vid2ascii - buffer to ASCII renderer                           ║
║ • fitfile GPS visualizer - 3D activity mapper                    ║
║ • rigidbodysim - iterative solver for physics models             ║
║ • flowsimulator - DEC & Laplace field tool                       ║
║ • crt simulator - advanced CRT renderer                          ║
╚══════════════════════════════════════════════════════════════════╝
                     ▲
                     │
╔══════════════════════════════════════════════════════════════════╗
║ High-Level API Layer                                             ║
║ • Guardian dynamic primitives                                    ║
║ • Cross-domain utilities                                         ║
║ • Efficient rendering engine                                     ║
║ • Responsive and interactive geometries                          ║
║ • Classic physics simulations vectorized                         ║
╚══════════════════════════════════════════════════════════════════╝
                     ▲
                     │
╔══════════════════════════════════════════════════════════════════╗
║ Core Domain Modules                                              ║
║ • guardian_platform.h/c           (Delocalizer)                  ║
║ • guardian_platform_extended.h    (Input & rendering)            ║
║ • cross_process_api.h/c           (Cross-process messaging)      ║
║ • utils.h/c                       (Dynamic Primitives)           ║
║ • execution_graph.h/c             (Task orchestrator)            ║
║ • parametric_domain.h/c           (Lossless Domains)             ║
║ • parametric_transform.h/c        (Continuous Geometry)          ║
║ • dec.h/c                         (Differential Edge Calculus)   ║
║ • dag.h/c                         (Wrap Counting Directional Graphs) ║
║ • graph_ops.h/c                   (graph API)                    ║
║ • stencil.h/.c                    (N-D interaction map)          ║
║ • geneology.h/c                   (Advanced relationship tracking) ║
║ • metric_tensor.h/c               (N-D Capability)               ║
║ • laplace_beltrami.h/c            (Core Physics)                 ║
║ • eigensolver.h/c                 (Eigenmode Decomposition)      ║
║ • kernels.h/c                     (Graph Kernels)                ║
║ • interpolator.h/c                (Reticulating Splines)         ║
║ • differentiator.h/c              (difference/integration engine)║
║ • double_buffer.h/c               (generic db byte arrays)       ║
║ • diff_print.h/c                  (differential buffer updating) ║
║ • iterative_solver.h/c            (iterative convergence engine) ║
║ • histogram_normalization.h/c     (histogram aware normalization) ║
║ • envelopes.h/c                   (parametric and quantized ADSR) ║
║ • fft.h/c                         (FFT capabilities)             ║
║ • synthesizer.h/c                 (signal generation core)       ║
║ • pidgeon_solver.h/c              (probabilistic pigeonhole solver) ║
║ • auxin_diffusion.h/c             (simulated plant exploration)  ║
╚══════════════════════════════════════════════════════════════════╝
                     ▲
                     │
╔══════════════════════════════════════════════════════════════════╗
║ Backend Execution Layer                                          ║
║ • Eigen tensor broadcasts                                        ║
║ • ONNX graph executor                                            ║
║ • Batch deployed OpenGL/GLSL compute shaders                     ║
╚══════════════════════════════════════════════════════════════════╝
```

## Module Overview
The code base is organised into small, self‑contained headers.  Below is an
outline of the main pieces, their pure mathematical roots and practical roles.

| Module | Maths background | Layperson description |
|---------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `guardian_platform.h`     | Operating Systems               | Delocalizes OS features with portable threading, I/O, and timing, forming the base for input and rendering layers. |
| `guardian_platform_extended.h` | Device Interfaces            | Enumerates input devices and captures cross-platform events to enable integrated rendering pipelines. |
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
| `iterative_solver.h`      | Numerical analysis             | Simple iterative solver for extrapolating simulations and driving convergence-controlled updates. |
| `histogram_normalization.h` | Statistics                     | Suite of advanced methods for histogram-based data normalization. |
| `execution_graph.h`       | Scheduler theory               | Executes dependent tasks in an ordered graph. |
| `envelopes.h`             | Signal processing              | Parametric and quantized ADSR or custom envelope modeling. |
| `fft.h`                   | Harmonic analysis              | Lightweight FFT implementation for spectral transforms. |
| `cross_process_api.h`     | Operating Systems              | Cross-process messaging and synchronization primitives. |
| `synthesizer.h`           | Signal processing              | Core oscillator for simple waveform generation. |
| `pidgeon_solver.h`        | Probability theory             | Estimates collision likelihood when mapping many items into few buckets. |
| `auxin_diffusion.h`       | Biological modeling            | Simulates resource-driven growth using auxin-like diffusion. |

These components combine to express shapes common in engineering (lines,
surfaces, volumes) and arbitrary parametrised forms.  They are designed to work
with future DEC kernels so that continuous descriptions can be discretised on
demand.

## Neural Network Integration
The engine bundles a lightweight neural network system built on top of the DAG
infrastructure.  Each network registers a collection of DAGs as differentiable
steps, with user-provided forward and backward callbacks.  Custom functions can
be appended through the `NeuralNetworkFunctionRepo`, enabling gradient-based
learning and experimental architectures that operate directly on geometric data.

## Contributing
This project is a platform for next-generation geometry, simulation, and learning systems. Contributors are encouraged to design with the following principles in mind:

- **Embrace Functional & Composable Design:** Prefer pure functions and operations that can be chained together.

- **Think in Vectors:** Ensure new algorithms are vectorized to leverage the core engine's performance.

- **Be Mesh-Agnostic:** Where possible, design operations that support both structured and unstructured data.

- **Leverage Advanced APIs:** Use the stencil and relational APIs to express novel numerical, physical, or learning operations.

- **Document with Context:** Provide both the mathematical background and the practical role of new modules.
