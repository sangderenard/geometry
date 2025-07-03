#ifndef GEOMETRY_STENCIL_H
#define GEOMETRY_STENCIL_H

#include <stddef.h>
#include "geometry/types.h"
#ifdef __cplusplus
extern "C" {
#endif

// --- Legacy 3D Stencil API (for compatibility) ---


Stencil* stencil_create_rectangular(int radius);
Stencil* stencil_create_polar(int radius);
void stencil_destroy(Stencil* s);

// --- Generalized N-Dimensional Stencil API ---

typedef struct GeneralStencil GeneralStencil;

// State resolver callback for pole placement/adjustment
// Should update the offsets/weights in the poles array based on a force field or energy function
// Returns 1 if equilibrium is reached, 0 otherwise
// (User can use user_data for parameters, force field, etc.)
typedef int (*StencilStateResolver)(StencilPole* poles, size_t count, size_t dims, void* user_data);


// Create a custom N-pole stencil in D dimensions (user provides offsets and weights, can provide a state resolver)
GeneralStencil* stencil_create_custom(size_t dims, size_t count, const int* offsets, const double* weights, StencilStateResolver resolver, void* resolver_data);



// Create a D-dimensional rectangular stencil of given radius and pattern
GeneralStencil* stencil_create_rectangular_nd(size_t dims, int radius,
                                              RectangularStencilType type);

// Create a D-dimensional polar (equidistant) stencil of given radius (Euclidean distance)
GeneralStencil* stencil_create_polar_nd(size_t dims, int radius);

// Destroy a GeneralStencil and free memory
void stencil_destroy_general(GeneralStencil* s);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_STENCIL_H
