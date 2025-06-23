#include "geometry/stencil.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Generalized Stencil API: Descriptor and Field Types
 *
 * This API supports the construction of stencils in arbitrary dimensions, with pole (direction/offset) arrangements determined by user-supplied geometry or by iterative solvers.
 *
 * Field Types:
 * - Rectangular (Cartesian): Poles are axis-aligned, e.g. 6-point, 14-point, or 27-point stencils in 3D.
 * - Interaction Field (formerly "polar"): Poles are arranged according to a set of states with defined mutual interactions (attractive, repulsive, etc.), not to be confused with polar coordinates. This is a generalization where each pole's position and weight may be determined by a field of mutual influences, not just geometric distance.
 *
 * Terminology:
 * - "Interaction Field" is used here to describe a set of states (poles) with defined mutual reactions, as opposed to "polar field" which may be confused with polar coordinates.
 *
 * Examples:
 * 1. Rectangular: A 3D 27-point stencil (all neighbors in a 3x3x3 cube).
 * 2. Interaction Field: A 6-pole arrangement where each pole's position is iteratively solved based on attractive/repulsive forces from the other poles, e.g. for meshless PDEs or neural field models.
 * 3. Custom: User supplies N poles in D dimensions, with a state resolver that encodes arbitrary interaction rules (e.g., energy minimization, equilibrium, or even learned arrangements).
 *
 * The state resolver callback allows contributors to define the rules for pole placement and adjustment, enabling the stencil geometry to be the solution to a state equation in n-dimensional continuous physics or other interaction-based systems.
 */

// --- Legacy 3D Stencil API (compatibility) ---
Stencil* stencil_create_rectangular(int radius) {
    // TODO: Implement legacy 3D rectangular stencil
    return NULL;
}

Stencil* stencil_create_polar(int radius) {
    // TODO: Implement legacy 3D polar stencil
    return NULL;
}

void stencil_destroy(Stencil* s) {
    if (!s) return;
    free(s->points);
    free(s);
}

// --- Generalized N-Dimensional Stencil API ---
GeneralStencil* stencil_create_custom(size_t dims, size_t count, const int* offsets, const double* weights, StencilStateResolver resolver, void* resolver_data) {
    // Allocate and copy user-provided offsets and weights
    GeneralStencil* s = (GeneralStencil*)calloc(1, sizeof(GeneralStencil));
    s->poles = (StencilPole*)calloc(count, sizeof(StencilPole));
    s->count = count;
    s->dims = dims;
    s->resolver = resolver;
    s->resolver_data = resolver_data;
    for (size_t i = 0; i < count; ++i) {
        s->poles[i].offsets = (int*)calloc(dims, sizeof(int));
        memcpy(s->poles[i].offsets, offsets + i * dims, dims * sizeof(int));
        s->poles[i].dims = dims;
        s->poles[i].weight = weights ? weights[i] : 1.0;
    }
    // Optionally, run the resolver to reach equilibrium
    if (resolver) {
        int equilibrium = 0;
        int max_iter = 1000; // Arbitrary max iterations
        for (int iter = 0; iter < max_iter && !equilibrium; ++iter) {
            equilibrium = resolver(s->poles, s->count, s->dims, s->resolver_data);
        }
    }
    return s;
}

static size_t pow_size(size_t base, size_t exp) {
    size_t r = 1;
    for (size_t i = 0; i < exp; ++i) r *= base;
    return r;
}

static int is_zero_vec(const int* v, size_t dims) {
    for (size_t i = 0; i < dims; ++i)
        if (v[i] != 0) return 0;
    return 1;
}

GeneralStencil* stencil_create_rectangular_nd(size_t dims, int radius,
                                              RectangularStencilType type) {
    if (dims == 0 || radius <= 0) return NULL;

    if (type == RECT_STENCIL_DEFAULT) {
        if (dims == 2)
            type = RECT_STENCIL_AXIS_ALIGNED_WITH_CENTER; // 5-point
        else
            type = RECT_STENCIL_AXIS_ALIGNED; // 6-point in 3D, 8-point in 4D
    }

    size_t count = 0;
    switch (type) {
        case RECT_STENCIL_AXIS_ALIGNED:
            count = dims * 2 * radius;
            break;
        case RECT_STENCIL_AXIS_ALIGNED_WITH_CENTER:
            count = dims * 2 * radius + 1;
            break;
        case RECT_STENCIL_FULL:
            count = pow_size(2 * radius + 1, dims) - 1;
            break;
        case RECT_STENCIL_FULL_WITH_CENTER:
            count = pow_size(2 * radius + 1, dims);
            break;
        case RECT_STENCIL_14_POINT_3D:
            if (dims != 3 || radius != 1) return NULL;
            count = 14;
            break;
        case RECT_STENCIL_24_CELL_4D:
            if (dims != 4 || radius != 1) return NULL;
            count = 24;
            break;
        default:
            return NULL;
    }

    GeneralStencil* s = (GeneralStencil*)calloc(1, sizeof(GeneralStencil));
    s->poles = (StencilPole*)calloc(count, sizeof(StencilPole));
    s->count = count;
    s->dims = dims;

    size_t idx = 0;

    if (type == RECT_STENCIL_AXIS_ALIGNED ||
        type == RECT_STENCIL_AXIS_ALIGNED_WITH_CENTER) {
        for (size_t d = 0; d < dims; ++d) {
            for (int r = 1; r <= radius; ++r) {
                s->poles[idx].offsets = (int*)calloc(dims, sizeof(int));
                s->poles[idx].dims = dims;
                s->poles[idx].offsets[d] = r;
                s->poles[idx++].weight = 1.0;

                s->poles[idx].offsets = (int*)calloc(dims, sizeof(int));
                s->poles[idx].dims = dims;
                s->poles[idx].offsets[d] = -r;
                s->poles[idx++].weight = 1.0;
            }
        }
        if (type == RECT_STENCIL_AXIS_ALIGNED_WITH_CENTER) {
            s->poles[idx].offsets = (int*)calloc(dims, sizeof(int));
            s->poles[idx].dims = dims;
            s->poles[idx].weight = 1.0;
            idx++;
        }
    } else if (type == RECT_STENCIL_FULL ||
               type == RECT_STENCIL_FULL_WITH_CENTER) {
        int* cur = (int*)calloc(dims, sizeof(int));
        for (size_t i = 0; i < dims; ++i) cur[i] = -radius;
        int done = 0;
        while (!done) {
            if (type == RECT_STENCIL_FULL_WITH_CENTER || !is_zero_vec(cur, dims)) {
                s->poles[idx].offsets = (int*)calloc(dims, sizeof(int));
                s->poles[idx].dims = dims;
                for (size_t d = 0; d < dims; ++d)
                    s->poles[idx].offsets[d] = cur[d];
                s->poles[idx++].weight = 1.0;
            }
            for (size_t d = 0; d < dims; ++d) {
                cur[d]++;
                if (cur[d] <= radius) break;
                cur[d] = -radius;
                if (d == dims - 1) done = 1;
            }
        }
        free(cur);
    } else if (type == RECT_STENCIL_14_POINT_3D) {
        static const int dirs[6][3] = {
            {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}
        };
        for (int i = 0; i < 6; ++i) {
            s->poles[idx].offsets = (int*)calloc(3, sizeof(int));
            s->poles[idx].dims = 3;
            memcpy(s->poles[idx].offsets, dirs[i], 3*sizeof(int));
            s->poles[idx++].weight = 1.0;
        }
        static const int diags[8][3] = {
            {1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},
            {-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}
        };
        for (int i = 0; i < 8; ++i) {
            s->poles[idx].offsets = (int*)calloc(3, sizeof(int));
            s->poles[idx].dims = 3;
            memcpy(s->poles[idx].offsets, diags[i], 3*sizeof(int));
            s->poles[idx++].weight = 1.0;
        }
    } else if (type == RECT_STENCIL_24_CELL_4D) {
        int pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
        for (int p = 0; p < 6; ++p) {
            for (int sx = -1; sx <= 1; sx += 2) {
                for (int sy = -1; sy <= 1; sy += 2) {
                    s->poles[idx].offsets = (int*)calloc(4, sizeof(int));
                    s->poles[idx].dims = 4;
                    s->poles[idx].offsets[pairs[p][0]] = sx;
                    s->poles[idx].offsets[pairs[p][1]] = sy;
                    s->poles[idx++].weight = 1.0;
                }
            }
        }
    }

    return s;
}

GeneralStencil* stencil_create_polar_nd(size_t dims, int radius) {
    // TODO: Implement D-dimensional polar stencil
    return NULL;
}

void stencil_destroy_general(GeneralStencil* s) {
    if (!s) return;
    for (size_t i = 0; i < s->count; ++i) {
        free(s->poles[i].offsets);
    }
    free(s->poles);
    free(s);
}
