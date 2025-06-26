#include "geometry/histogram_normalization.h"

void histogram_normalize(const double* data, size_t count,
                         const HistogramNormParams* params,
                         double* out) {
    if (!data || !out || count == 0) return;
    int bins = params ? params->bins : HNORM_MAX_BINS;
    if (bins <= 0 || bins > HNORM_MAX_BINS) bins = HNORM_MAX_BINS;

    double hist[HNORM_MAX_BINS] = {0};
    double cdf[HNORM_MAX_BINS] = {0};
    double minv = data[0];
    double maxv = data[0];
    for (size_t i = 1; i < count; ++i) {
        if (data[i] < minv) minv = data[i];
        if (data[i] > maxv) maxv = data[i];
    }
    double range = maxv - minv;
    if (range <= 0) {
        for (size_t i = 0; i < count; ++i) out[i] = 0;
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        int idx = (int)((data[i] - minv) / range * (bins - 1));
        if (idx < 0) idx = 0;
        if (idx >= bins) idx = bins - 1;
        hist[idx] += 1.0;
    }
    double cum = 0;
    for (int i = 0; i < bins; ++i) {
        cum += hist[i];
        cdf[i] = cum;
    }
    double total = cdf[bins - 1];
    for (size_t i = 0; i < count; ++i) {
        int idx = (int)((data[i] - minv) / range * (bins - 1));
        if (idx < 0) idx = 0;
        if (idx >= bins) idx = bins - 1;
        out[i] = cdf[idx] / total;
    }
}
