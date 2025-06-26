#include "geometry/fft.h"
#include <math.h>

void fft_naive(size_t n, const double* in_r, const double* in_i,
               double* out_r, double* out_i) {
    for (size_t k = 0; k < n; ++k) {
        double sum_r = 0.0;
        double sum_i = 0.0;
        for (size_t t = 0; t < n; ++t) {
            double angle = -2.0 * M_PI * t * k / n;
            double c = cos(angle);
            double s = sin(angle);
            double xr = in_r[t];
            double xi = in_i ? in_i[t] : 0.0;
            sum_r += xr * c - xi * s;
            sum_i += xr * s + xi * c;
        }
        out_r[k] = sum_r;
        out_i[k] = sum_i;
    }
}
