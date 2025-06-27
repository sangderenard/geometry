#ifndef SIMD_CONFIG_H
#define SIMD_CONFIG_H

#if defined(__AVX2__)
#define SIMD_COMPILE_AVX2 1
#else
#define SIMD_COMPILE_AVX2 0
#endif

#if defined(__SSE2__)
#define SIMD_COMPILE_SSE2 1
#else
#define SIMD_COMPILE_SSE2 0
#endif

#endif // SIMD_CONFIG_H
