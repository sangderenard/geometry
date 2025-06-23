#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <tuple>
#include <memory>

namespace DEC {

using namespace Eigen;
using Scalar = double;
using Index = int;

using VectorX = Matrix<Scalar, Dynamic, 1>;
using MatrixX = Matrix<Scalar, Dynamic, Dynamic>;
using SparseMatrixX = SparseMatrix<Scalar>;

struct Mesh {
    MatrixX V;            // Vertex positions (n x 3)
    MatrixXi F;           // Face indices (m x 3)
    std::vector<std::vector<Index>> edges; // Edges per vertex or per face, topology only
};

class DECSystem {
public:
    explicit DECSystem(const Mesh& mesh);

    // Operators
    SparseMatrixX d0() const; // 0-form to 1-form exterior derivative
    SparseMatrixX d1() const; // 1-form to 2-form exterior derivative

    SparseMatrixX star0() const; // Hodge star on 0-forms
    SparseMatrixX star1() const; // Hodge star on 1-forms
    SparseMatrixX star2() const; // Hodge star on 2-forms

    SparseMatrixX laplace0() const; // Hodge Laplacian on 0-forms
    SparseMatrixX laplace1() const; // Hodge Laplacian on 1-forms

    // Scalar field processing
    VectorX gradient(const VectorX& scalar_field) const;
    VectorX divergence(const VectorX& vector_field_1form) const;
    VectorX curl(const VectorX& vector_field_1form) const;

    // Field solvers
    VectorX solve_poisson_0form(const VectorX& rhs) const;
    VectorX solve_poisson_1form(const VectorX& rhs) const;

    // Vector transport
    MatrixX integrate_vector_field(const VectorX& vector_1form, const Vector3d& seed_point, double step_size, int steps) const;

    // Hash surface exit
    MatrixX exit_hashmap_from_field(const VectorX& vector_field_1form, int oversample_rate = 4) const;

    // GPU toggles
    void enable_gpu_acceleration(bool use_gpu);

    // Debug / profiling
    void dump_operator_info() const;

private:
    Mesh mesh_;
    SparseMatrixX d0_, d1_, star0_, star1_, star2_;
    void compute_operators();
};

} // namespace DEC
