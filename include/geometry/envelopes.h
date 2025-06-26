#ifndef GEOMETRY_ENVELOPES_H
#define GEOMETRY_ENVELOPES_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double attack;
    double decay;
    double sustain;
    double release;
    double sustain_level;
} EnvelopeADSR;

double envelope_value(const EnvelopeADSR* env, double time, double duration);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_ENVELOPES_H
