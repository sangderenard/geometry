#include "geometry/pidgeon_solver.h"
#include <math.h>

/* Approximate probability of at least one collision using basic factorial logic */
double pidgeon_solver_probability(size_t items, size_t buckets) {
    if (buckets == 0) return 1.0;
    if (items == 0) return 0.0;
    if (items > buckets) return 1.0;

    double p = 1.0;
    for (size_t i = 0; i < items; ++i) {
        p *= (double)(buckets - i) / (double)buckets;
    }
    return 1.0 - p;
}
