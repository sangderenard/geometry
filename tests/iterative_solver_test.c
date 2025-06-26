#include "geometry/iterative_solver.h"
#include <assert.h>
#include <stdio.h>

static void step_fn(IterativeSolverState* state, void* user) {
    (void)state;
    int* count = (int*)user;
    (*count)++;
}

int main(void) {
    IterativeSolverState solver;
    iterative_solver_init(&solver, 3, 1e-4, 1.0);
    int counter = 0;
    iterative_solver_step(&solver, step_fn, &counter);
    iterative_solver_step(&solver, step_fn, &counter);
    iterative_solver_step(&solver, step_fn, &counter);
    iterative_solver_step(&solver, step_fn, &counter); // should not increment
    assert(counter == 3);
    printf("iterative_solver_test passed\n");
    return 0;
}
