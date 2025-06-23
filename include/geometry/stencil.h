#ifndef GEOMETRY_STENCIL_H
#define GEOMETRY_STENCIL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Legacy 3D Stencil API (for compatibility) ---
typedef struct {
    int dx, dy, dz;   // Offset from center
    double weight;    // Weight for this point
} StencilPoint;

typedef struct {
    StencilPoint* points;
    size_t count;
} Stencil;

Stencil* stencil_create_rectangular(int radius);
Stencil* stencil_create_polar(int radius);
void stencil_destroy(Stencil* s);

// --- Generalized N-Dimensional Stencil API ---

typedef struct GeneralStencil GeneralStencil;

typedef struct {
    int* offsets;      // [dims] array: e.g. {dx, dy, dz, ...}
    size_t dims;       // Number of dimensions
    double weight;     // Weight for this pole
} StencilPole;

// State resolver callback for pole placement/adjustment
// Should update the offsets/weights in the poles array based on a force field or energy function
// Returns 1 if equilibrium is reached, 0 otherwise
// (User can use user_data for parameters, force field, etc.)
typedef int (*StencilStateResolver)(StencilPole* poles, size_t count, size_t dims, void* user_data);

struct GeneralStencil {
    StencilPole* poles;
    size_t count;
    size_t dims;
    // Optional: reference to a state resolver for iterative equilibrium
    StencilStateResolver resolver;
    void* resolver_data;
};

// Create a custom N-pole stencil in D dimensions (user provides offsets and weights, can provide a state resolver)
GeneralStencil* stencil_create_custom(size_t dims, size_t count, const int* offsets, const double* weights, StencilStateResolver resolver, void* resolver_data);

// Common rectangular stencil variants
typedef enum {
    // Default axis-aligned cross (6 point in 3D, 5 point in 2D, etc.)
    RECT_STENCIL_DEFAULT = 0,
    // Axis-aligned neighbours only (2*d points)
    RECT_STENCIL_AXIS_ALIGNED,
    // Axis-aligned neighbours plus centre (2*d + 1)
    RECT_STENCIL_AXIS_ALIGNED_WITH_CENTER,
    // Full box without centre ( (2*radius+1)^d - 1 )
    RECT_STENCIL_FULL,
    // Full box including centre ( (2*radius+1)^d )
    RECT_STENCIL_FULL_WITH_CENTER,
    // Classical 14 point stencil in 3D (axis + face diagonals)
    RECT_STENCIL_14_POINT_3D,
    // 24-cell vertices in 4D (\u00b11,\u00b11,0,0 permutations)
    RECT_STENCIL_24_CELL_4D
} RectangularStencilType;

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
