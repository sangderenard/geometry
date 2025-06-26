#include "geometry/complex_numbers.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

int main(void) {
    ComplexNumber a = complex_from_cartesian(1.0, 1.0);
    ComplexNumber b = complex_from_polar(1.0, M_PI / 2.0);
    ComplexNumber sum = complex_add(a, b);
    assert(fabs(sum.real - 1.0) < 1e-9); // real component
    assert(fabs(sum.imag - 2.0) < 1e-9); // imag component

    ComplexNumber prod = complex_mul(a, b);
    // a*(i) -> (1+i)*(0+1i) = -1+1i
    assert(fabs(prod.real + 1.0) < 1e-9);
    assert(fabs(prod.imag - 1.0) < 1e-9);

    ComplexNumber diff = complex_sub(a, b);
    assert(fabs(diff.real - 1.0) < 1e-9); // 1 - 0
    assert(fabs(diff.imag - 0.0) < 1e-9); // 1 - 1

    ComplexNumber quot = complex_div(a, b);
    assert(fabs(quot.real - 1.0) < 1e-9); // (1+i)/i = 1 - i*(1)
    assert(fabs(quot.imag + 1.0) < 1e-9);

    ComplexNumber conj = complex_conj(a);
    assert(conj.real == a.real && conj.imag == -a.imag);
    assert(fabs(complex_abs(a) - sqrt(2.0)) < 1e-9);

    double vec[2];
    complex_to_vector(&prod, vec);
    assert(fabs(vec[0] + 1.0) < 1e-9 && fabs(vec[1] - 1.0) < 1e-9);

    printf("complex_numbers_test passed\n");
    return 0;
}
