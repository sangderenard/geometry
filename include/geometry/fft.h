#ifndef GEOMETRY_FFT_H
#define GEOMETRY_FFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void fft_naive(size_t n, const double* in_real, const double* in_imag,
               double* out_real, double* out_imag);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_FFT_H
