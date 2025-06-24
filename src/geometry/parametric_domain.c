#include "geometry/parametric_domain.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ParametricDomain* parametric_domain_create(size_t dimensions) {
    if (dimensions == 0 || dimensions > PD_MAX_DIM) return NULL;
    ParametricDomain* pd = (ParametricDomain*)malloc(sizeof(ParametricDomain));
    if (!pd) return NULL;
    memset(pd, 0, sizeof(ParametricDomain));
    pd->dim = dimensions;

    for (size_t i = 0; i < dimensions; ++i) {
        pd->axes[i].start = 0.0;
        pd->axes[i].end = 1.0;
        pd->axes[i].periodic = false;
        pd->axes[i].periodicity_type = PD_PERIODIC_NONE;
        pd->axes[i].start_boundary = PD_BOUNDARY_INCLUSIVE;
        pd->axes[i].end_boundary = PD_BOUNDARY_INCLUSIVE;
        pd->axes[i].bc_type = PD_BC_DIRICHLET;
        pd->axes[i].bc_function = NULL;
        pd->axes[i].discontinuities = NULL;
        pd->axes[i].extrapolate_neg = NULL;
        pd->axes[i].extrapolate_pos = NULL;
    }

    return pd;
}

void parametric_domain_destroy(ParametricDomain* pd) {
    if (!pd) return;
    for (size_t i = 0; i < pd->dim; ++i) {
        DiscontinuityNode* node = pd->axes[i].discontinuities;
        while (node) {
            DiscontinuityNode* next = node->next;
            free(node);
            node = next;
        }
    }
    free(pd);
}

void parametric_domain_set_axis_range(ParametricDomain* pd, size_t axis, double start, double end) {
    if (!pd || axis >= pd->dim) return;
    pd->axes[axis].start = start;
    pd->axes[axis].end = end;
}

void parametric_domain_set_periodic(ParametricDomain* pd, size_t axis, bool periodic, PeriodicityType type) {
    if (!pd || axis >= pd->dim) return;
    pd->axes[axis].periodic = periodic;
    pd->axes[axis].periodicity_type = periodic ? type : PD_PERIODIC_NONE;
}

void parametric_domain_set_boundaries(ParametricDomain* pd, size_t axis, BoundaryType start, BoundaryType end) {
    if (!pd || axis >= pd->dim) return;
    pd->axes[axis].start_boundary = start;
    pd->axes[axis].end_boundary = end;
}

void parametric_domain_set_boundary_condition(ParametricDomain* pd, size_t axis, BoundaryConditionType bc_type, void (*func)(double*, size_t)) {
    if (!pd || axis >= pd->dim) return;
    pd->axes[axis].bc_type = bc_type;
    pd->axes[axis].bc_function = func;
}

void parametric_domain_add_discontinuity(ParametricDomain* pd, size_t axis, double position, BoundaryType type) {
    if (!pd || axis >= pd->dim) return;
    DiscontinuityNode* node = (DiscontinuityNode*)malloc(sizeof(DiscontinuityNode));
    node->position = position;
    node->type = type;
    node->next = pd->axes[axis].discontinuities;
    node->prev = NULL;
    if (node->next) node->next->prev = node;
    pd->axes[axis].discontinuities = node;
}

void parametric_domain_set_extrapolation(ParametricDomain* pd, size_t axis, double (*neg)(double), double (*pos)(double)) {
    if (!pd || axis >= pd->dim) return;
    pd->axes[axis].extrapolate_neg = neg;
    pd->axes[axis].extrapolate_pos = pos;
}

bool parametric_domain_contains_point(const ParametricDomain* pd, const double* params) {
    if (!pd || !params) return false;
    for (size_t i = 0; i < pd->dim; i++) {
        double p = params[i], s = pd->axes[i].start, e = pd->axes[i].end;
        bool start_ok = (pd->axes[i].start_boundary == PD_BOUNDARY_INCLUSIVE) ? p >= s : p > s;
        bool end_ok = (pd->axes[i].end_boundary == PD_BOUNDARY_INCLUSIVE) ? p <= e : p < e;
        if (!start_ok || !end_ok) return false;
    }
    return true;
}

void parametric_domain_clamp_point(const ParametricDomain* pd, double* params) {
    if (!pd || !params) return;
    for (size_t i = 0; i < pd->dim; i++) {
        if (params[i] < pd->axes[i].start) params[i] = pd->axes[i].start;
        if (params[i] > pd->axes[i].end) params[i] = pd->axes[i].end;
    }
}

void parametric_domain_wrap_point(const ParametricDomain* pd, double* params) {
    if (!pd || !params) return;
    for (size_t i = 0; i < pd->dim; i++) {
        if (!pd->axes[i].periodic) continue;
        double s = pd->axes[i].start, e = pd->axes[i].end, r = e - s;
        if (r <= 0.0) continue;
        switch (pd->axes[i].periodicity_type) {
            case PD_PERIODIC_SIMPLE:
            case PD_PERIODIC_CYCLIC:
                while (params[i] < s) params[i] += r;
                while (params[i] >= e) params[i] -= r;
                break;
            case PD_PERIODIC_MIRROR:
                {
                    double norm = (params[i] - s) / r;
                    int cycles = (int)floor(norm);
                    double frac = norm - cycles;
                    params[i] = (cycles % 2 == 0) ? s + frac * r : e - frac * r;
                }
                break;
            case PD_PERIODIC_POLARIZED:
                params[i] = s + 0.5 * r * (1 - cos((params[i] - s) * 2 * 3.14159265358979323846 / r));
                break;
            default: break;
        }
    }
}
