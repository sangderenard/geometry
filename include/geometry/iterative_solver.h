#ifndef GEOMETRY_ITERATIVE_SOLVER_H
#define GEOMETRY_ITERATIVE_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IterativeSolverState {
    int max_iterations;
    double tolerance;
    double step_scale;
    int current_iteration;
} IterativeSolverState;

typedef void (*IterativeStepFn)(IterativeSolverState* state, void* user);

void iterative_solver_init(IterativeSolverState* state,
                           int max_iterations,
                           double tolerance,
                           double step_scale);

void iterative_solver_step(IterativeSolverState* state,
                           IterativeStepFn fn,
                           void* user);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_ITERATIVE_SOLVER_H
