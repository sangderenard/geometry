#ifndef MODSPACE_H
#define MODSPACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

// A modulation function: takes [0,1] and maps it to [0,1] non-linearly
typedef double (*ModulationFn)(double t);

// ModSpace struct
typedef struct {
    double start;
    double end;
    size_t num;
    ModulationFn modulation;  // NULL if linear
} ModSpace;

// Allocates and fills array of values
double* modspace_generate(const ModSpace* space);

#ifdef __cplusplus
}
#endif

#endif
