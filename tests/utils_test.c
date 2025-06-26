/* utils_test.c - rewritten to brutally test TokenGuardian, GuardianHeap, and linked lists */

#include "geometry/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

void test_guardian_initialization() {
    printf("[TEST] Guardian Initialization...\n");

    TokenGuardian dummy = guardian_create_dummy();
    assert(dummy.state == GUARDIAN_NOT_USED);

    TokenGuardian* heap_guardian = guardian_create_heap();
    assert(heap_guardian != NULL);
    assert(heap_guardian->state == GUARDIAN_HEAP);

    guardian_destroy(heap_guardian);
    printf("  -> Passed\n");
}

void test_guardian_token_integrity() {
    printf("[TEST] Guardian Token Integrity...\n");

    TokenGuardian* g = guardian_create_heap();
    int x = 42;
    GuardianToken tok = guardian_create_pointer_token(g, &x, 1);
    assert(tok.ptr == &x);
    assert(tok.valid == true);

    guardian_destroy(g);
    printf("  -> Passed\n");
}

void test_guardian_heap_push_and_verify() {
    printf("[TEST] Guardian Heap Push...\n");

    TokenGuardian* g = guardian_create_heap();
    for (int i = 0; i < 1000; ++i) {
        int* data = (int*)malloc(sizeof(int));
        *data = i;
        GuardianToken tok = guardian_create_pointer_token(g, data, i % 4);
        guardian_heap_push(g->heap, tok);
    }

    GuardianHeap* heap = g->heap;
    LinkedListItem* iter = heap->head;
    int count = 0;
    while (iter) {
        assert(iter->token.valid);
        assert(iter->token.ptr != NULL);
        iter = iter->next;
        count++;
    }
    assert(count == 1000);

    guardian_destroy(g);
    printf("  -> Passed\n");
}

void test_linked_list_no_guardian() {
    printf("[TEST] Linked List Without Guardian...\n");

    LinkedListItem* head = linked_list_item_create(NULL, (void*)"root", 0, (GuardianToken){0});
    LinkedListItem* a = linked_list_item_create(NULL, (void*)"a", 0, (GuardianToken){0});
    LinkedListItem* b = linked_list_item_create(NULL, (void*)"b", 0, (GuardianToken){0});

    head->next = a;
    a->prev = head;
    a->next = b;
    b->prev = a;

    assert(strcmp((char*)head->ptr, "root") == 0);
    assert(strcmp((char*)head->next->ptr, "a") == 0);
    assert(strcmp((char*)head->next->next->ptr, "b") == 0);

    free(b);
    free(a);
    free(head);
    printf("  -> Passed\n");
}

void test_mixed_guardian_linkage() {
    printf("[TEST] Mixed Guardian / Manual List...\n");

    TokenGuardian* g = guardian_create_heap();
    LinkedListItem* a = linked_list_item_create(g, (void*)"a", 0, guardian_create_pointer_token(g, (void*)"a", 0));
    LinkedListItem* b = linked_list_item_create(NULL, (void*)"b", 0, (GuardianToken){0});

    a->next = b;
    b->prev = a;

    assert(strcmp((char*)a->ptr, "a") == 0);
    assert(strcmp((char*)b->ptr, "b") == 0);

    guardian_destroy(g);
    free(b);
    printf("  -> Passed\n");
}

int main() {
    test_guardian_initialization();
    test_guardian_token_integrity();
    test_guardian_heap_push_and_verify();
    test_linked_list_no_guardian();
    test_mixed_guardian_linkage();

    printf("\n[ALL TESTS PASSED]\n");
    return 0;
}
// This code is a test suite for the Guardian system, focusing on initialization, token integrity, heap management, and linked list operations.
// It includes tests for creating and managing guardians, ensuring token integrity, pushing items to the heap