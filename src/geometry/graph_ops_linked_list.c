#include "geometry/graph_ops_linked_list.h"
#include "geometry/graph_ops.h"
#include "assembly_backend/memory_ops.h"
#include <stdlib.h>

typedef struct GuardianLinkedList GuardianLinkedList;

// Create an empty list
static void* ll_create(void) {
    GuardianLinkedList* list = instantiate_on_input_cache(NODE_FEATURE_IDX_LINKED_LIST, 1, false);
    if (!list) {
        // Handle allocation failure
        return NULL;
    }
    return (void*)list->self->guardian_pointer_token;
}

// Destroy the list and free nodes
static void ll_destroy(void* container) {
    LinkedList* list = (LinkedList*)container;
    LLNode* node = list->head;
    while (node) {
        LLNode* next = node->next;
        free(node);
        node = next;
    }
    free(list);
}

// Push element to the end
static void ll_push(void* container, void* element) {
    LinkedList* list = (LinkedList*)container;
    LLNode* node = (LLNode*)malloc(sizeof(LLNode));
    node->data = element;
    node->next = NULL;
    if (!list->head) {
        list->head = node;
    } else {
        LLNode* curr = list->head;
        while (curr->next) curr = curr->next;
        curr->next = node;
    }
    list->size++;
}

// Pop element from the end
static void* ll_pop(void* container) {
    LinkedList* list = (LinkedList*)container;
    if (!list->head) return NULL;
    LLNode* curr = list->head;
    LLNode* prev = NULL;
    while (curr->next) {
        prev = curr;
        curr = curr->next;
    }
    void* data = curr->data;
    if (prev) prev->next = NULL;
    else list->head = NULL;
    free(curr);
    list->size--;
    return data;
}

// Get element at index
static void* ll_get(void* container, size_t index) {
    LinkedList* list = (LinkedList*)container;
    LLNode* curr = list->head;
    size_t i = 0;
    while (curr && i < index) {
        curr = curr->next;
        i++;
    }
    return curr ? curr->data : NULL;
}

// Get size of the list
static size_t ll_size(const void* container) {
    const LinkedList* list = (const LinkedList*)container;
    return list->size;
}

// Iterate over each element
static void ll_for_each(void* container, void (*fn)(void*, void*), void* user_data) {
    LinkedList* list = (LinkedList*)container;
    LLNode* curr = list->head;
    while (curr) {
        fn(curr->data, user_data);
        curr = curr->next;
    }
}

// Stub suite wiring only the needed linked-list ops
const OperationSuite graph_ops_linked_list = {
    .create   = ll_create,
    .destroy  = ll_destroy,
    .push     = ll_push,
    .pop      = ll_pop,
    .get      = ll_get,
    .size     = ll_size,
    .for_each = ll_for_each
};
