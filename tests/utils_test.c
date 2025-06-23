#include "geometry/utils.h"
#include <assert.h>
#include <stdio.h>

static void forward_add(Node* self, void* data) {
    int* val = (int*)data;
    (*val)++;
}

static void backward_add(Node* self, void* data) {
    int* val = (int*)data;
    (*val)++;
}

int main(void) {
    Node* a = node_create();
    Node* b = node_create();

    // Add relation and features
    size_t rel_idx = node_add_relation(a, 1, forward_add, backward_add);
    assert(rel_idx == 0);
    assert(node_get_relation(a, rel_idx) != NULL);

    size_t feat_idx = node_add_feature(a, "feat1");
    assert(feat_idx == 0);
    assert(node_get_feature(a, feat_idx) != NULL);

    size_t exp_idx = node_add_exposure(a, forward_add, backward_add);
    assert(exp_idx == 0);
    assert(node_get_exposure(a, exp_idx) != NULL);

    // b mirrors relation for backward gathers
    node_add_relation(b, 1, forward_add, backward_add);

    // Link nodes bidirectionally
    size_t link_idx = node_add_bidirectional_link(a, b, 0);
    assert(link_idx == 0);
    assert(node_get_forward_link(a, link_idx)->node == b);
    assert(node_get_backward_link(b, link_idx)->node == a);

    int count = 0;
    node_scatter_to_siblings(a, &count);
    assert(count == 1); // b incremented

    count = 0;
    node_gather_from_siblings(b, &count);
    assert(count == 1); // gather from a

    count = 0;
    node_scatter_to_descendants(a, &count);
    assert(count == 1); // only one descendant

    count = 0;
    node_gather_from_ancestors(b, &count);
    assert(count == 1);

    node_destroy(a);
    node_destroy(b);

    printf("utils_test passed\n");
    return 0;
}
