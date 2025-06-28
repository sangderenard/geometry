#ifndef PARAMETRIC_DOMAIN_H
#define PARAMETRIC_DOMAIN_H

#include <stddef.h>
#include "geometry/types.h"

#ifdef __cplusplus
extern "C" {
#endif


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
