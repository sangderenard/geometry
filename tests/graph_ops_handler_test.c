#include "geometry/graph_ops_handler.h"
#include <assert.h>
#include <stdio.h>

static int push_called = 0;
static int pop_called = 0;

static void dummy_push(Node* n, Node* c) {
    (void)n; (void)c; push_called++;
}

static Node* dummy_pop(Node* n) {
    pop_called++; return n;
}

int main(void) {
    GraphOps ops = {0};
    ops.push = dummy_push;
    ops.pop = dummy_pop;
    GraphOpsHandler h;
    assert(graph_ops_handler_init(&h, &ops) == GRAPH_OPS_OK);

    Node node = {0};
    Node child = {0};

    assert(graph_ops_push(&h, &node, &child) == GRAPH_OPS_OK);
    assert(push_called == 1);

    Node* ret = graph_ops_pop(&h, &node);
    assert(ret == &node && pop_called == 1);

    /* Call a missing op */
    ret = graph_ops_shift(&h, &node);
    assert(ret == NULL);

    printf("graph_ops_handler_test passed\n");
    return 0;
}
