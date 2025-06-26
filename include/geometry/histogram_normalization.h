#ifndef GEOMETRY_HISTOGRAM_NORMALIZATION_H
#define GEOMETRY_HISTOGRAM_NORMALIZATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HNORM_MAX_BINS 256

typedef struct {
    int bins;
} HistogramNormParams;

void histogram_normalize(const double* data, size_t count,
                         const HistogramNormParams* params,
                         double* out);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_HISTOGRAM_NORMALIZATION_H
