#include "geometry/synthesizer.h"
#include <math.h>

void synth_init(Synthesizer* s, double sample_rate) {
    if (!s) return;
    s->phase = 0.0;
    s->sample_rate = sample_rate;
    s->frequency = 440.0;
    s->wave = WAVE_SINE;
}

double synth_step(Synthesizer* s) {
    if (!s) return 0.0;
    double value = 0.0;
    double t = s->phase;
    switch (s->wave) {
    case WAVE_SINE:   value = sin(t * 2.0 * M_PI); break;
    case WAVE_SAW:    value = 2.0 * (t - floor(t + 0.5)); break;
    case WAVE_SQUARE: value = (fmod(t, 1.0) < 0.5) ? 1.0 : -1.0; break;
    }
    s->phase += s->frequency / s->sample_rate;
    if (s->phase > 1.0)
        s->phase -= 1.0;
    return value;
}
