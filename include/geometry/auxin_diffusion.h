#ifndef GEOMETRY_AUXIN_DIFFUSION_H
#define GEOMETRY_AUXIN_DIFFUSION_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AuxinState {
    double concentration;
    double diffusion_rate;
} AuxinState;

double auxin_diffuse(AuxinState* state, double input);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_AUXIN_DIFFUSION_H
