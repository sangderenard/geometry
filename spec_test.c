#include <stdio.h>
#include <inttypes.h>
#include "speculative.h"

int main(void) {
    size_t rows = 3, cols = 3;
    double matrix_a[9] = {1,2,3,4,5,6,7,8,9};
    double matrix_b[9] = {9,8,7,6,5,4,3,2,1};
    double matrix_c[9] = {0};

    MatMulAddFuseInstructionSetNNodeHeader *node = create_matmul_add_fuse_instruction_set_n_node(
        N, cols, rows, cols, rows, cols, rows,
        (ParameterValue*)matrix_a,
        (ParameterValue*)matrix_b,
        (ParameterValue*)matrix_c
    );

    if (!node) {
        printf("Node creation failed (error code %d).\n", speculative_error);
        switch (speculative_error) {
            case SPEC_ERR_NONE:
                printf("No error reported (SPEC_ERR_NONE).\n");
                break;
            case SPEC_ERR_ALLOC_HEADER:
                printf("Failed to allocate node header (SPEC_ERR_ALLOC_HEADER).\n");
                break;
            case SPEC_ERR_ALLOC_NODE:
                printf("Failed to allocate node storage (SPEC_ERR_ALLOC_NODE).\n");
                break;
            default:
                printf("Unknown error code.\n");
        }
        return 1;
    }

    printf("Node header: A %zux%zu, B %zux%zu, C %zux%zu\n",
        node->matrix_a_cols, node->matrix_a_rows,
        node->matrix_b_cols, node->matrix_b_rows,
        node->matrix_c_cols, node->matrix_c_rows);

    // Now dump only the operator plan and dataset
    WordData *words = (WordData*) node->node;
    printf("Operator plan flags: 0x%x, operator_type: %" PRIu64 "\n",
        words[0].operator_operand_plan.flags,
        words[0].operator_operand_plan.operator_type);

    printf("Dataset subsequent_rows: %u, next_dataset_ref: %" PRIu64 "\n",
        words[1].dataset.subsequent_rows,
        words[1].dataset.next_dataset_ref);

    return 0;
}
