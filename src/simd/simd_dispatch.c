#include "simd/simd_dispatch.h"

void simd_add_u8_avx2(simd_u8* dst, const simd_u8* src, size_t count);
void simd_add_u8_sse2(simd_u8* dst, const simd_u8* src, size_t count);
void simd_add_u8_fallback(simd_u8* dst, const simd_u8* src, size_t count);

void simd_add_u16_avx2(simd_u16*, const simd_u16*, size_t);
void simd_add_u16_sse2(simd_u16*, const simd_u16*, size_t);
void simd_add_u16_fallback(simd_u16*, const simd_u16*, size_t);

void simd_add_s8_avx2(simd_s8*, const simd_s8*, size_t);
void simd_add_s8_sse2(simd_s8*, const simd_s8*, size_t);
void simd_add_s8_fallback(simd_s8*, const simd_s8*, size_t);

void simd_add_f32_avx2(simd_f32*, const simd_f32*, size_t);
void simd_add_f32_sse2(simd_f32*, const simd_f32*, size_t);
void simd_add_f32_fallback(simd_f32*, const simd_f32*, size_t);

void simd_add_f64_avx2(simd_f64*, const simd_f64*, size_t);
void simd_add_f64_sse2(simd_f64*, const simd_f64*, size_t);
void simd_add_f64_fallback(simd_f64*, const simd_f64*, size_t);

void simd_and_u8_avx2(simd_u8*, const simd_u8*, size_t);
void simd_and_u8_sse2(simd_u8*, const simd_u8*, size_t);
void simd_and_u8_fallback(simd_u8*, const simd_u8*, size_t);

void simd_or_u8_avx2(simd_u8*, const simd_u8*, size_t);
void simd_or_u8_sse2(simd_u8*, const simd_u8*, size_t);
void simd_or_u8_fallback(simd_u8*, const simd_u8*, size_t);

void simd_xor_u8_avx2(simd_u8*, const simd_u8*, size_t);
void simd_xor_u8_sse2(simd_u8*, const simd_u8*, size_t);
void simd_xor_u8_fallback(simd_u8*, const simd_u8*, size_t);

void simd_memcpy_avx2(void*, const void*, size_t);
void simd_memcpy_sse2(void*, const void*, size_t);
void simd_memcpy_fallback(void*, const void*, size_t);

void simd_memset_avx2(void*, int, size_t);
void simd_memset_sse2(void*, int, size_t);
void simd_memset_fallback(void*, int, size_t);

simd_add_u8_fn   simd_add_u8_impl   = simd_add_u8_fallback;
simd_add_u16_fn  simd_add_u16_impl  = simd_add_u16_fallback;
simd_add_s8_fn   simd_add_s8_impl   = simd_add_s8_fallback;
simd_add_f32_fn  simd_add_f32_impl  = simd_add_f32_fallback;
simd_add_f64_fn  simd_add_f64_impl  = simd_add_f64_fallback;
simd_and_u8_fn   simd_and_u8_impl   = simd_and_u8_fallback;
simd_or_u8_fn    simd_or_u8_impl    = simd_or_u8_fallback;
simd_xor_u8_fn   simd_xor_u8_impl   = simd_xor_u8_fallback;
simd_memcpy_fn   simd_memcpy_impl = simd_memcpy_fallback;
simd_memset_fn   simd_memset_impl = simd_memset_fallback;

#if defined(_MSC_VER)
#include <intrin.h>
static int cpu_supports_avx2(void) {
    int info[4];
    __cpuidex(info, 0, 0);
    if (info[0] < 7) return 0;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
}
static int cpu_supports_sse2(void) {
    int info[4];
    __cpuidex(info, 0, 0);
    if (info[0] < 1) return 0;
    __cpuidex(info, 1, 0);
    return (info[3] & (1 << 26)) != 0;
}
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static int cpu_supports_avx2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_max(0, 0) || __get_cpuid_max(0,0) < 7) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 5)) != 0;
}
static int cpu_supports_sse2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_max(0, 0) || __get_cpuid_max(0,0) < 1) return 0;
    __cpuid(1, eax, ebx, ecx, edx);
    return (edx & (1 << 26)) != 0;
}
#else
static int cpu_supports_avx2(void) { return 0; }
static int cpu_supports_sse2(void) { return 0; }
#endif

void simd_init_dispatch(void) {
    if (cpu_supports_avx2()) {
        simd_add_u8_impl   = simd_add_u8_avx2;
        simd_add_u16_impl  = simd_add_u16_avx2;
        simd_add_s8_impl   = simd_add_s8_avx2;
        simd_add_f32_impl  = simd_add_f32_avx2;
        simd_add_f64_impl  = simd_add_f64_avx2;
        simd_and_u8_impl   = simd_and_u8_avx2;
        simd_or_u8_impl    = simd_or_u8_avx2;
        simd_xor_u8_impl   = simd_xor_u8_avx2;
        simd_memcpy_impl = simd_memcpy_avx2;
        simd_memset_impl = simd_memset_avx2;
    } else if (cpu_supports_sse2()) {
        simd_add_u8_impl   = simd_add_u8_sse2;
        simd_add_u16_impl  = simd_add_u16_sse2;
        simd_add_s8_impl   = simd_add_s8_sse2;
        simd_add_f32_impl  = simd_add_f32_sse2;
        simd_add_f64_impl  = simd_add_f64_sse2;
        simd_and_u8_impl   = simd_and_u8_sse2;
        simd_or_u8_impl    = simd_or_u8_sse2;
        simd_xor_u8_impl   = simd_xor_u8_sse2;
        simd_memcpy_impl = simd_memcpy_sse2;
        simd_memset_impl = simd_memset_sse2;
    }
}
