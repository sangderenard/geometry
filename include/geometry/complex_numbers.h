#ifndef GEOMETRY_COMPLEX_NUMBERS_H
#define GEOMETRY_COMPLEX_NUMBERS_H

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Simple complex number represented as a 2D vector (real, imag).
 * Operations are implemented using Euler's formula
 *     e^{iθ} = cos(θ) + i sin(θ)
 * so that complex multiplication corresponds to rotating and
 * scaling the underlying vector.
 */
typedef struct {
    double real;
    double imag;
} ComplexNumber;

// Construct from Cartesian components
static inline ComplexNumber complex_from_cartesian(double real, double imag) {
    ComplexNumber c; c.real = real; c.imag = imag; return c;
}

// Construct from polar coordinates using Euler's formula
static inline ComplexNumber complex_from_polar(double r, double theta) {
    ComplexNumber c; c.real = r * cos(theta); c.imag = r * sin(theta); return c;
}

// Add two complex numbers
static inline ComplexNumber complex_add(ComplexNumber a, ComplexNumber b) {
    return complex_from_cartesian(a.real + b.real, a.imag + b.imag);
}

// Multiply two complex numbers
static inline ComplexNumber complex_mul(ComplexNumber a, ComplexNumber b) {
    return complex_from_cartesian(a.real * b.real - a.imag * b.imag,
                                  a.real * b.imag + a.imag * b.real);
}

// Subtract two complex numbers
static inline ComplexNumber complex_sub(ComplexNumber a, ComplexNumber b) {
    return complex_from_cartesian(a.real - b.real, a.imag - b.imag);
}

// Divide two complex numbers
static inline ComplexNumber complex_div(ComplexNumber a, ComplexNumber b) {
    double denom = b.real * b.real + b.imag * b.imag;
    return complex_from_cartesian(
        (a.real * b.real + a.imag * b.imag) / denom,
        (a.imag * b.real - a.real * b.imag) / denom);
}

// Complex conjugate
static inline ComplexNumber complex_conj(ComplexNumber a) {
    return complex_from_cartesian(a.real, -a.imag);
}

// Magnitude (absolute value)
static inline double complex_abs(ComplexNumber a) {
    return sqrt(a.real * a.real + a.imag * a.imag);
}

// Convert to a raw 2D vector [real, imag]
static inline void complex_to_vector(const ComplexNumber* c, double out[2]) {
    out[0] = c->real; out[1] = c->imag;
}

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_COMPLEX_NUMBERS_H
