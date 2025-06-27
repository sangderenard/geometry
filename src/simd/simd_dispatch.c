#include "simd_dispatch.h"

void simd_add_u8_avx2(uint8_t* dst, const uint8_t* src, size_t count);
void simd_add_u8_sse2(uint8_t* dst, const uint8_t* src, size_t count);
void simd_add_u8_fallback(uint8_t* dst, const uint8_t* src, size_t count);

simd_add_u8_fn simd_add_u8 = simd_add_u8_fallback;

#if defined(_MSC_VER)
#include <intrin.h>
static int cpu_supports_avx2() {
    int info[4];
    __cpuidex(info, 0, 0);
    if (info[0] < 7) return 0;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5));
}
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static int cpu_supports_avx2() {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_max(0, 0)) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 5));
}
#else
static int cpu_supports_avx2() { return 0; }
#endif

void simd_init_dispatch() {
    if (cpu_supports_avx2()) {
        simd_add_u8 = simd_add_u8_avx2;
    } else {
        simd_add_u8 = simd_add_u8_sse2;
    }
}