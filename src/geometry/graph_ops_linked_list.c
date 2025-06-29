#include "geometry/graph_ops_linked_list.h"
#include "assembly_backend/memory_ops.h"
#include <stdlib.h>


// Create an empty list
static void* ll_create(void) {
    GuardianLinkedList* list = instantiate_on_input_cache(NODE_FEATURE_IDX_LINKED_LIST, 1, false);
    if (!list) {
        // Handle allocation failure
        return NULL;
    }
    return (void*)list;
}

// Destroy the list and free nodes
static void ll_destroy(void* container) {
    return memory_ops_retire(container, NODE_FEATURE_IDX_LINKED_LIST);
}

// Push element to the end
static void ll_push(void* container, void* element) {
    GuardianLinkedList* list = (GuardianLinkedList*)container;
    GuardianPointerToken* node = (GuardianPointerToken*)element;

    // Clear any existing links on the new node
    memops_set_prev_on_link(node, NULL);
    memops_set_next_on_link(node, NULL);

    if (list->size == 0) {
        // Empty list: node is both head (left) and tail (right)
        list->left  = node;
        list->right = node;
    } else {
        // Non-empty: append after tail
        GuardianPointerToken* tail = list->right;
        memops_set_next_on_link(tail, node);
        memops_set_prev_on_link(node, tail);
        list->right = node;
    }

    list->size++;
}

// Pop element from the end
static void* ll_pop(void* container) {
    GuardianPointerToken* return_token = NULL;
    GuardianLinkedList* list = (GuardianLinkedList*)container;
    if (!list->left) return NULL;
    GuardianLinkNode * right_link = memops_get_pointer_from_token(list->right);
    if (!right_link) {
        return_token = list->left;
    }
    else {
        return_token = list->right;
    }
    list->right = right_link->prev;
    if (list->right) {
        memops_set_next_on_link(list->right, NULL); // Clear next link of new tail
    } else {
        list->left = NULL; // List is now empty
    }

    list->size--;
    if (list->size < 0) {
        // Handle error: size should not be negative
        list->size = 0;
    }
    return return_token;
}

// Get element at index
static void* ll_get(void* container, size_t index) {
    GuardianLinkedList* list = (GuardianLinkedList*)container;
    if (index >= list->size) {
        // Handle error: index out of bounds
        return NULL;
    }else if (index < 0) {
        // Handle error: negative index
        return NULL;
    }else if (index == 0) {
        // Special case for head
        return list->left;
    }else if (index == list->size - 1) {
        // Special case for tail
        return list->right;
    }else if (index > list->size / 2) {
        // If index is in the second half, start from tail
        GuardianPointerToken * next_link = list->right;
        GuardianPointerToken * return_link = memops_hop_link(next_link, -(int)(list->size - index - 1));
        if (!return_link) {
            // Handle error: unable to hop to the requested index
            return NULL;
        }
        return return_link;
    } else if (index < list->size / 2) {
        // If index is in the first half, start from head
        GuardianPointerToken * next_link = list->left;
        GuardianPointerToken * return_link = memops_hop_link(next_link, (int)index);
        if (!return_link) {
            // Handle error: unable to hop to the requested index
            return NULL;
        }
        return return_link;
    }
    return NULL;
    
}

// Get size of the list
static size_t ll_size(const void* container) {
    const GuardianLinkedList* list = (const GuardianLinkedList*)container;
    if (!list) {
        // Handle error: container is NULL
        return 0;
    }
    return list->size;
}

// Iterate over each element
static void * ll_for_each(void* container, void * (*fn)(void*, void*), void* user_data, boolean inplace) {
    GuardianLinkedList* list = (GuardianLinkedList*)container;
    GuardianLinkNode* curr = memops_get_pointer_from_token(list->left);
    if (!curr) {
        // Handle error: list is empty
        return;
    }
    while (curr) {
        if (inplace) {
            // If inplace, modify the current node directly
            curr->payload = fn(curr->payload, user_data);
        } else {
            // If not inplace, pass the pointer to the function
            fn(curr->payload, user_data);
        }
        
        curr = memops_get_pointer_from_token(curr->next);
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
