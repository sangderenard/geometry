#include "geometry/iterative_solver.h"

void iterative_solver_init(IterativeSolverState* state,
                           int max_iterations,
                           double tolerance,
                           double step_scale) {
    if (!state) return;
    state->max_iterations = max_iterations;
    state->tolerance = tolerance;
    state->step_scale = step_scale;
    state->current_iteration = 0;
}

void iterative_solver_step(IterativeSolverState* state,
                           IterativeStepFn fn,
                           void* user) {
    if (!state || !fn) return;
    if (state->current_iteration >= state->max_iterations)
        return;

    fn(state, user);
    state->current_iteration++;
}
