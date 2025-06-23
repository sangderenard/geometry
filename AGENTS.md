# AGENTS

This repo is designed with autonomous and assistive agent integration in mind.

## Key Expectations
- All operations are vectorizable and composable.
- DEC operations support both structured and unstructured meshes.
- C++11+ Eigen-only for baseline. GPU extensions to come.
- Modular, functional, and mesh-agnostic design is preferred.

## Project Snapshot
Agents collaborating in this repository should be aware of the trajectory so far:

- **Early utilities** laid out the `Node` container and linking helpers.
- **GraphOps layer** introduced generic push/pop interfaces across graphs.
- **Geneology and DAG** structures established a basis for delayed computation.
- **Lock bank and concurrency guidelines** highlight parallel execution goals.
- **Operator declarations** hint at arithmetic on graphs and sets.
- **Parametric domains** expose continuous ranges to marry discrete meshes with analytic geometry.
- **Generalized stencils and interaction fields** enable local operators and sampling patterns based on physical, neural, or abstract rules, supporting meshless and adaptive computation.
- **Relational vector spaces** allow shortest-path and local-projection computations in arbitrary relational graphs, supporting advanced analysis in non-Euclidean or discontinuous spaces.

## Advanced Agent Guidance
- **Vectorization:** Design all new operations to be vectorizable, supporting batch and parallel execution.
- **Composability:** Favor functional, composable interfaces that can be chained or scheduled in delayed computation graphs.
- **Mesh Agnosticism:** Ensure algorithms work for both structured (grid) and unstructured (graph, meshless) domains.
- **Extensibility:** Use the advanced stencil and relational APIs to express new numerical, physical, or learning operations. Document new modules with both mathematical and practical context.
- **Agent Collaboration:** Structure code and documentation so that both human and autonomous agents can contribute, extend, and optimize the system. Design for future GPU and distributed extensions.

## Architectural Vision
This project is a platform for next-generation geometry, simulation, and learning systems. The scaffolding, abstractions, and documentation are as important as the finished code. Contributors are encouraged to prototype and scale new mathematical and computational ideas, leveraging the advanced faculties of the system.
