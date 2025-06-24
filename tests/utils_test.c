#include "geometry/utils.h"
#include "geometry/guardian.h"
#include "geometry/stencil.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void forward_add(Node* self, void* data) {
    int* val = (int*)data;
    (*val)++;
}

static void backward_add(Node* self, void* data) {
    int* val = (int*)data;
    (*val)++;
}

// Helper to create a random rectangular stencil
GeneralStencil* random_rect_stencil(size_t dims, int radius) {
    RectangularStencilType type = (rand() % 2) ? RECT_STENCIL_AXIS_ALIGNED : RECT_STENCIL_AXIS_ALIGNED_WITH_CENTER;
    return stencil_create_rectangular_nd(dims, radius, type);
}

void test_quaternion_operators() {
    Quaternion q1 = {1, 0, 0, 0};
    Quaternion q2 = {0, 1, 0, 0};
    Quaternion q3 = quaternion_add(q1, q2);
    assert(q3.w == 1 && q3.x == 1 && q3.y == 0 && q3.z == 0);
    Quaternion q4 = quaternion_mul(q1, q2);
    assert(q4.w == 0 && q4.x == 1 && q4.y == 0 && q4.z == 0);
    Quaternion q5 = quaternion_conjugate(q2);
    assert(q5.w == 0 && q5.x == -1 && q5.y == 0 && q5.z == 0);
    Quaternion q6 = quaternion_normalize((Quaternion){0,2,0,0});
    assert(fabs(q6.x - 1.0) < 1e-8);
    double axis[3], angle;
    quaternion_to_axis_angle(q1, axis, &angle);
    assert(fabs(angle) < 1e-8);
    Quaternion q7 = quaternion_from_axis_angle((double[]){0,0,1}, M_PI/2);
    double roll, pitch, yaw;
    quaternion_to_euler(q7, &roll, &pitch, &yaw);
    Quaternion q8 = quaternion_from_euler(roll, pitch, yaw);
    assert(quaternion_is_normalized(q8, 1e-8));
    double m[3][3];
    quaternion_to_matrix(q7, m);
    Quaternion q9 = quaternion_from_matrix(m);
    assert(quaternion_is_normalized(q9, 1e-8));
    assert(quaternion_angle_between(q7, q9) < 1e-6);
    Quaternion q10 = quaternion_slerp(q1, q7, 0.5);
    assert(quaternion_is_normalized(q10, 1e-8));
}

void test_quaternion_history() {
    QuaternionHistory* hist = quaternion_history_create(4, NULL, NULL);
    Quaternion q1 = {1,0,0,0};
    Quaternion q2 = quaternion_from_axis_angle((double[]){0,0,1}, M_PI/4);
    Quaternion q3 = quaternion_from_axis_angle((double[]){0,0,1}, M_PI/2);
    quaternion_history_add(hist, q1);
    quaternion_history_add(hist, q2);
    quaternion_history_add(hist, q3);
    assert(hist->count == 3);
    assert(quaternion_history_is_smooth(hist, 1.0));
    Quaternion avg = quaternion_history_average(hist);
    assert(quaternion_is_normalized(avg, 1e-8));
    quaternion_history_destroy(hist);
}

void test_quaternion_edge_cases() {
    Quaternion q180 = quaternion_from_axis_angle((double[]){1,0,0}, M_PI);
    double axis[3], angle;
    quaternion_to_axis_angle(q180, axis, &angle);
    assert(fabs(angle - M_PI) < 1e-8);
    Quaternion qflip1 = {0,1,0,0};
    Quaternion qflip2 = {0,-1,0,0};
    assert(quaternion_angle_between(qflip1, qflip2) < 1e-8);
}

int main(void) {
    // Original test
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

    // Node locking API
    Node* lock_node = node_create();
    node_lock(lock_node);
    assert(node_is_locked(lock_node));
    assert(!node_trylock(lock_node));
    node_unlock(lock_node);
    assert(!node_is_locked(lock_node));
    node_destroy(lock_node);

    // TokenGuardian basic usage
    TokenGuardian* guard = guardian_create();
    unsigned long thread_token = guardian_register_thread(guard);
    unsigned long mem_token;
    void* block = guardian_alloc(guard, 16, &mem_token);
    assert(block != NULL);
    const char* msg = "hello";
    guardian_send(guard, thread_token, thread_token, msg, strlen(msg)+1);
    char buf[16];
    size_t n = guardian_receive(guard, thread_token, buf, sizeof(buf));
    assert(n == strlen(msg)+1 && strcmp(buf, msg) == 0);
    guardian_free(guard, mem_token);
    guardian_unregister_thread(guard, thread_token);
    guardian_destroy(guard);

    printf("utils_test passed\n");

    // New test for random node pairs
    srand((unsigned)time(NULL));
    size_t num_pairs = 10;
    for (size_t i = 0; i < num_pairs; ++i) {
        size_t dims_a = 2 + rand() % 2; // 2D or 3D
        size_t dims_b = 2 + rand() % 2;
        int radius_a = 1 + rand() % 2;
        int radius_b = 1 + rand() % 2;
        GeneralStencil* sa = random_rect_stencil(dims_a, radius_a);
        GeneralStencil* sb = random_rect_stencil(dims_b, radius_b);
        StencilSet* set_a = stencilset_wrap_single(sa);
        StencilSet* set_b = stencilset_wrap_single(sb);
        // Try all pole pairs (just first for demo)
        size_t pole_a = 0, pole_b = 0;
        StencilRelation rel;
        int reltype = stencilset_negotiate_bond(set_a, pole_a, set_b, pole_b, &rel);
        printf("Pair %zu: dims_a=%zu, dims_b=%zu, reltype=%d\n", i, dims_a, dims_b, reltype);
        // Cleanup
        stencil_destroy_general(sa);
        stencil_destroy_general(sb);
        free(set_a->stencils); free(set_a->relation[0]); free(set_a->relation); free(set_a);
        free(set_b->stencils); free(set_b->relation[0]); free(set_b->relation); free(set_b);
    }

    test_quaternion_operators();
    test_quaternion_history();
    test_quaternion_edge_cases();
    printf("All quaternion tests passed!\n");

    return 0;
}
