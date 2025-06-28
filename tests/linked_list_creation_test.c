#include "geometry/utils.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    /* simple payloads */
    const char* items[3] = {"a", "b", "c"};
    GuardianLinkedList* list = guardian_create_linked_list(NULL, 3, NODE_FEATURE_TYPE_POINTER, (void**)items);
    assert(list);
    assert(list->size == 3);
    printf("linked_list_creation_test passed\n");
    return 0;
}
