#include "geometry/envelopes.h"

double envelope_value(const EnvelopeADSR* env, double t, double dur) {
    if (!env || dur <= 0) return 0.0;
    double a = env->attack;
    double d = env->decay;
    double s = env->sustain;
    double r = env->release;
    double level = env->sustain_level;
    if (t < a) {
        return (a == 0.0) ? 1.0 : (t / a);
    }
    t -= a;
    if (t < d) {
        return 1.0 - (1.0 - level) * (t / (d == 0.0 ? 1.0 : d));
    }
    t -= d;
    double sustain_time = dur - (a + d + r);
    if (sustain_time < 0) sustain_time = 0;
    if (t < sustain_time) {
        return level;
    }
    t -= sustain_time;
    if (t < r) {
        return level * (1.0 - t / (r == 0.0 ? 1.0 : r));
    }
    return 0.0;
}
