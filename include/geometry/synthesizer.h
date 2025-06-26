#ifndef GEOMETRY_SYNTHESIZER_H
#define GEOMETRY_SYNTHESIZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef enum {
    WAVE_SINE,
    WAVE_SAW,
    WAVE_SQUARE
} SynthWave;

typedef struct Synthesizer {
    double phase;
    double sample_rate;
    double frequency;
    SynthWave wave;
} Synthesizer;

void synth_init(Synthesizer* s, double sample_rate);
double synth_step(Synthesizer* s);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_SYNTHESIZER_H
