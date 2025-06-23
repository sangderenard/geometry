#ifndef PARAMETRIC_DOMAIN_H
#define PARAMETRIC_DOMAIN_H

#include <stddef.h>
#include <stdbool.h>

#define PD_MAX_DIM 16

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PD_BOUNDARY_INCLUSIVE,
    PD_BOUNDARY_EXCLUSIVE,
    PD_BOUNDARY_POS_INFINITY,
    PD_BOUNDARY_NEG_INFINITY
} BoundaryType;

typedef enum {
    PD_PERIODIC_NONE,
    PD_PERIODIC_SIMPLE,
    PD_PERIODIC_MIRROR,
    PD_PERIODIC_CYCLIC,
    PD_PERIODIC_POLARIZED
} PeriodicityType;

typedef enum {
    PD_BC_DIRICHLET,
    PD_BC_NEUMANN,
    PD_BC_CUSTOM
} BoundaryConditionType;

typedef struct DiscontinuityNode {
    double position;
    BoundaryType type;
    struct DiscontinuityNode* next;
    struct DiscontinuityNode* prev;
} DiscontinuityNode;

typedef struct AxisDescriptor {
    double start;
    double end;
    bool periodic;
    PeriodicityType periodicity_type;
    BoundaryType start_boundary;
    BoundaryType end_boundary;

    BoundaryConditionType bc_type;
    void (*bc_function)(double*, size_t); // Pointer to BC function
    DiscontinuityNode* discontinuities;

    double (*extrapolate_neg)(double);
    double (*extrapolate_pos)(double);
} AxisDescriptor;

typedef struct ParametricDomain {
    size_t dim;
    AxisDescriptor axes[PD_MAX_DIM];
} ParametricDomain;

ParametricDomain* parametric_domain_create(size_t dimensions);
void parametric_domain_destroy(ParametricDomain* pd);

void parametric_domain_set_axis_range(ParametricDomain* pd, size_t axis, double start, double end);
void parametric_domain_set_periodic(ParametricDomain* pd, size_t axis, bool periodic, PeriodicityType type);
void parametric_domain_set_boundaries(ParametricDomain* pd, size_t axis, BoundaryType start, BoundaryType end);
void parametric_domain_set_boundary_condition(ParametricDomain* pd, size_t axis, BoundaryConditionType bc_type, void (*func)(double*, size_t));
void parametric_domain_add_discontinuity(ParametricDomain* pd, size_t axis, double position, BoundaryType type);
void parametric_domain_set_extrapolation(ParametricDomain* pd, size_t axis, double (*neg)(double), double (*pos)(double));

bool parametric_domain_contains_point(const ParametricDomain* pd, const double* params);
void parametric_domain_clamp_point(const ParametricDomain* pd, double* params);
void parametric_domain_wrap_point(const ParametricDomain* pd, double* params);

#ifdef __cplusplus
}
#endif

#endif // PARAMETRIC_DOMAIN_H
