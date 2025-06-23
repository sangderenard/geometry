# AGENTS

This repo is designed with autonomous and assistive agent integration in mind.

## Key Expectations
- All operations are vectorizable.
- DEC operations support structured and unstructured meshes.
- C++11+ Eigen-only for baseline. GPU extensions to come.

## Project Snapshot
Agents collaborating in this repository should be aware of the trajectory so far:

- **Early utilities** laid out the `Node` container and linking helpers.
- **GraphOps layer** introduced generic push/pop interfaces across graphs.
- **Geneology and DAG** structures established a basis for delayed computation.
- **Lock bank and concurrency guidelines** highlight parallel execution goals.
- **Operator declarations** hint at arithmetic on graphs and sets.
- **Parametric domains** expose continuous ranges to marry discrete meshes with analytic geometry.

Every new contribution should keep operations vectorizable and consider both structured and unstructured meshes. Where possible, prefer functional designs that compose naturally with the DAG and neural network scaffolds.
