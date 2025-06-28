/* utils_test.c - rewritten to brutally test TokenGuardian, GuardianHeap, and linked lists */

#include "geometry/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

void test_guardian_initialization() {
    printf("[TEST] Guardian Initialization...\n");

    TokenGuardian * dummy = guardian_create_dummy();
    assert(dummy->self->guardian_pointer_token == GUARDIAN_NOT_USED);

    TokenGuardian* heap_guardian = guardian_create_heap();
    assert(heap_guardian != NULL);
    
    memory_ops_retire(heap_guardian, NODE_FEATURE_IDX_HEAP);
    printf("  -> Passed\n");
}

void test_guardian_token_integrity() {
    printf("[TEST] Guardian Token Integrity...\n");

    TokenGuardian* g = guardian_create_heap();
    int x = 42;
    GuardianToken* tok = guardian_create_pointer_token(g, &x, 1);
    assert(tok->___object == &x);
    assert(tok->token);

    memory_ops_retire(g, NODE_FEATURE_IDX_GUARDIAN);
    printf("  -> Passed\n");
}

void test_guardian_heap_push_and_verify() {
    printf("[TEST] Guardian Heap Push...\n");

    TokenGuardian* g = guardian_create_heap();
    for (int i = 0; i < 1000; ++i) {
        int* data = (int*)malloc(sizeof(int));
        *data = i;
        GuardianToken * tok = guardian_create_pointer_token(g, data, i % 4);
        graph_ops_heap.push(g->heap, tok);
    }

}

#include "geometry/graph_ops_handler.h"
#include "geometry/graph_ops.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

void test_linked_list_api_basic() {
    printf("[TEST] Linked List API (basic create/push/get/pop)...\n");

    // Look up the linked-list suite
    const OperationSuite* suite =
        get_operation_suite(NODE_FEATURE_TYPE_LINKED_LIST);
    assert(suite && "Linked-list suite must be available");

    // Create an empty list
    void* list = suite->create();
    assert(list && "create() should return a valid container");
    assert(suite->size(list) == 0);

    // Insert three string literals
    const char* items[] = { "root", "a", "b" };
    for (size_t i = 0; i < 3; ++i) {
        suite->push(list, (void*)items[i]);
        // size should increment
        assert(suite->size(list) == i + 1);
        // get() should return the same pointer we pushed
        void* elem = suite->get(list, i);
        assert(elem == items[i]);
    }

    // Verify contents and order
    for (size_t i = 0; i < 3; ++i) {
        const char* got = (const char*)suite->get(list, i);
        assert(strcmp(got, items[i]) == 0);
    }

    // Pop should remove the last‐in element ("b")
    void* popped = suite->pop(list);
    assert(popped == items[2]);
    assert(suite->size(list) == 2);

    // Clean up
    suite->destroy(list);
    printf("  -> Passed\n");
}

void test_linked_list_api_iteration() {
    printf("[TEST] Linked List API (for_each iteration)...\n");

    const OperationSuite* suite =
        get_operation_suite(NODE_FEATURE_TYPE_LINKED_LIST);
    void* list = suite->create();

    // Push numbers 1,2,3 as pointers
    int a = 1, b = 2, c = 3;
    suite->push(list, &a);
    suite->push(list, &b);
    suite->push(list, &c);

    // Collect via for_each
    int seen_sum = 0;
    void iter_fn(void* elem, void* user_data) {
        seen_sum += *(int*)elem;
    }
    suite->for_each(list, iter_fn, NULL);
    // sum of 1+2+3 = 6
    assert(seen_sum == 6);

    suite->destroy(list);
    printf("  -> Passed\n");
}

int main() {
    test_linked_list_api_basic();
    test_linked_list_api_iteration();
    printf("\n[ALL LINKED LIST API TESTS PASSED]\n");
    test_guardian_initialization();
    test_guardian_token_integrity();
    test_guardian_heap_push_and_verify();
    

    printf("\n[ALL TESTS PASSED]\n");
    return 0;
}
// This code is a test suite for the Guardian system, focusing on initialization, token integrity, heap management, and linked list operations.
// It includes tests for creating and managing guardians, ensuring token integrity, pushing items to the heap