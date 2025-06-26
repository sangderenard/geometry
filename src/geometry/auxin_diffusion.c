#include "geometry/auxin_diffusion.h"

/* Simple exponential diffusion plus accumulation */
double auxin_diffuse(AuxinState* state, double input) {
    if (!state) return 0.0;
    state->concentration += input;
    state->concentration *= state->diffusion_rate;
    return state->concentration;
}
