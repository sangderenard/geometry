#include "speculative.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>

size_t calculate_tree_node_requirements(size_t leaves, size_t b)
{
    size_t internals = 0;

    while (leaves > 1) {                // while we have >1 nodes to group
        leaves = (leaves + b - 1) / b;  // ceil divide  leaves / b
        internals += leaves;            // these become the new internal layer
    }
    return internals;                   // root included here?  exclude if not
}
size_t * matmul_get_matrix_index_remapping(
    size_t matrix_a_cols, size_t matrix_b_cols, size_t matrix_b_rows, boolean tertiary)
{
    const size_t sidm_multiplier = 8;
    const size_t operands = tertiary ? 3 : 2;
    size_t stripes = (matrix_a_cols * matrix_b_rows) / sidm_multiplier;

    size_t * indices_remapping = (size_t *)mg_alloc(
        sizeof(size_t) * stripes * operands * sidm_multiplier);

    for (size_t stripe = 0; stripe < stripes; ++stripe) {
        for (size_t lane = 0; lane < sidm_multiplier; ++lane) {
            size_t base_index = stripe * operands * sidm_multiplier;

            // A indices
            indices_remapping[base_index + 0 * sidm_multiplier + lane] =
                (stripe * sidm_multiplier + lane) % (matrix_a_cols * matrix_b_rows);

            // B indices
            indices_remapping[base_index + 1 * sidm_multiplier + lane] =
                (stripe * sidm_multiplier + lane) % (matrix_b_rows * matrix_b_cols);

            // C indices if tertiary
            if (tertiary) {
                indices_remapping[base_index + 2 * sidm_multiplier + lane] =
                    (stripe * sidm_multiplier + lane) % (matrix_b_cols * matrix_b_rows); 
            }
        }
    }

    return indices_remapping;
}

// Error variable
SpeculativeError speculative_error = SPEC_ERR_NONE;

MatMulAddFuseInstructionSetNNodeHeader *create_matmul_add_fuse_instruction_set_n_node(
    size_t subdivision_pointers, size_t matrix_a_cols, size_t matrix_a_rows,
    size_t matrix_b_cols, size_t matrix_b_rows, size_t matrix_c_cols, size_t matrix_c_rows,
    ParameterValue* matrix_a_flat, ParameterValue* matrix_b_flat, ParameterValue* matrix_c_flat){

    speculative_error = SPEC_ERR_NONE;
    MatMulAddFuseInstructionSetNNodeHeader *node_header = (MatMulAddFuseInstructionSetNNodeHeader *)mg_alloc(sizeof(MatMulAddFuseInstructionSetNNodeHeader));
    if (!node_header) {
        speculative_error = SPEC_ERR_ALLOC_HEADER;
        return NULL;
    }

    node_header->subdivision_pointers = N;
    node_header->matrix_a_cols = matrix_a_cols;
    node_header->matrix_a_rows = matrix_a_rows;
    node_header->matrix_b_cols = matrix_b_cols;
    node_header->matrix_b_rows = matrix_b_rows;
    node_header->matrix_c_cols = matrix_c_cols;
    node_header->matrix_c_rows = matrix_c_rows;

    // determine how many operations are going to be performed for the entire matrix multiplication
    // use this to determine how many parameters are needed and thus if the node is large enough
    size_t total_operations = matrix_a_rows * matrix_b_cols * matrix_b_rows;
    size_t total_operands = (matrix_c_flat) ? total_operations * 3 / 2 : total_operations * 2; // if C is provided, we have 3 operands, otherwise 2
    size_t first_line_operands_capacity = 24; // 12 words of parameters, each word is 2 operands, so 24 operands
    size_t first_page_operands_capacity = 16 * first_line_operands_capacity; // 16 lines of 24 operands each
    size_t first_book_operands_capacity = BOOK_SIZE;
    size_t total_additional_parameter_books;
    size_t first_book_operands = 0; // number of operands that fit in the first book
    size_t first_line_operands = 0;
    size_t first_page_operands = 0;
    if (total_operands <= first_line_operands_capacity) {
        first_line_operands_capacity = total_operands; // all operands fit in the first line
        first_page_operands = total_operands;
        first_book_operands = total_operands; // all operands fit in the first book
        total_additional_parameter_books = 0;
    } else if (total_operands <= first_page_operands_capacity) {
        first_line_operands = first_line_operands_capacity; // all operands fit in the first line
        first_page_operands = total_operands; // all operands fit in the first line
        first_book_operands = total_operands; // all operands fit in the first book
        total_additional_parameter_books = 0; // no additional books needed
    } else if (total_operands <= first_book_operands_capacity){
        first_line_operands = first_line_operands_capacity; // only the first line fits in the first book
        first_page_operands = first_page_operands_capacity; // all operands fit in the first page
        first_book_operands = total_operands; // only the first line fits in the first book
        total_additional_parameter_books = 0;
    } else {
        first_line_operands = first_line_operands_capacity; // first line fits in the first book
        first_page_operands = first_page_operands_capacity; // all operands fit in the first page
        first_book_operands = first_book_operands_capacity; // remaining operands fit in the first book
        total_additional_parameter_books = (total_operands - first_book_operands_capacity) / BOOK_SIZE;
        if ((total_operands - first_book_operands_capacity) % BOOK_SIZE > 0) {
            total_additional_parameter_books++; // if there are remaining operands, we need an additional book
        }
    }

    size_t n_tree_nodes = (1 + total_additional_parameter_books) + calculate_tree_node_requirements(1 + total_additional_parameter_books, N);
    size_t tail_parameters = total_operands - first_book_operands - (total_additional_parameter_books * BOOK_SIZE); // remaining parameters after the first book and additional books
    // Diagnostic: print allocation parameters for debugging simd_aligned_space
    printf("DEBUG: total_operations=%zu, total_operands=%zu, first_line_operands=%zu, BOOK_SIZE=%zu, first_book_operands=%zu, total_additional_parameter_books=%zu, n_tree_nodes=%zu, alloc_size=%zu\n",
           total_operations, total_operands, first_line_operands, (size_t)BOOK_SIZE, first_book_operands, total_additional_parameter_books, n_tree_nodes,
           n_tree_nodes * sizeof(MatMulAddFuseInstructionSetNNode));

    // Allocate memory for the node
    WordData * simd_aligned_space = (WordData *)mg_alloc(sizeof(MatMulAddFuseInstructionSetNNode) * n_tree_nodes);
    if (!simd_aligned_space) {
        speculative_error = SPEC_ERR_ALLOC_NODE;
        mg_free(node_header);
        return NULL;
    }

    node_header->node = (MatMulAddFuseInstructionSetNNode *)simd_aligned_space;

    if (!node_header->node) {
        speculative_error = SPEC_ERR_ALLOC_NODE_ASSIGNMENT;
        mg_free(node_header);
        return NULL;
    }

    printf("DEBUG: Allocated node at %p with size %zu bytes\n", simd_aligned_space, sizeof(MatMulAddFuseInstructionSetNNode) * n_tree_nodes);
    *simd_aligned_space = (WordData) {
        .operator_operand_plan = {.operator_type = OPERATOR_TYPE_64_3_ADD_MULTIPLY, .flags = 0},
    };

    
    WordData * reader = (WordData *)((char *)simd_aligned_space + sizeof(WordData));
    if (total_additional_parameter_books > 0) {

        *reader = (WordData) {
            .dataset = {
                .subsequent_rows = 8 * 16 * total_additional_parameter_books + first_book_operands / 2, // 8 pages of 16 rows each, plus the first book's rows
            },
        };
        WordData * head = reader;
        head[0].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_a_flat[0]), *(uint64_t *)(char *)&(matrix_a_flat[1]), // rows and columns of matrix A
            },
        };
        head[1].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_a_flat[2]), *(uint64_t *)(char *)&(matrix_a_flat[3]), // rows and columns of matrix B
            },
        };
        head[2].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_a_flat[4]), *(uint64_t *)(char *)&(matrix_a_flat[5]), // rows and columns of matrix C
            },
        };
        head[3].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_a_flat[6]), *(uint64_t *)(char *)&(matrix_a_flat[7]), // first row of matrix A
            },
        };
        head[4].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_b_flat[0]), *(uint64_t *)(char *)&(matrix_b_flat[1]), // first row of matrix B
            },
        };
        head[5].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_b_flat[2]), *(uint64_t *)(char *)&(matrix_b_flat[3]), // first row of matrix C
            },
        };
        head[6].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_b_flat[4]), *(uint64_t *)(char *)&(matrix_b_flat[5]), // second row of matrix A
            },
        };
        head[7].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_b_flat[6]), *(uint64_t *)(char *)&(matrix_b_flat[7]), // second row of matrix B
            },
        };
        head[8].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_c_flat[0]), *(uint64_t *)(char *)&(matrix_c_flat[1]), // second row of matrix C
            },
        };
        head[9].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_c_flat[2]), *(uint64_t *)(char *)&(matrix_c_flat[3]), // third row of matrix A
            },
        };
        head[10].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_c_flat[4]), *(uint64_t *)(char *)&(matrix_c_flat[5]), // third row of matrix B
            },
        };
        head[11].parameter_word = (ParameterWord) {
            .p = {
            *(uint64_t *)(char *)&(matrix_c_flat[6]), *(uint64_t *)(char *)&(matrix_c_flat[7]), // third row of matrix C
            },
        };
    } else if (!matrix_c_flat){
        // these next two need correction to interleave up to 8 parameters from each matrix
        // which in this case of no C matrix, means we interleave an additional 4
        // parameters from both A and B matrices, for a total of 24 parameters
        size_t total_words = total_operands / 2; // each word is 2 operands, so we divide by 2
        size_t words_per_matrix = total_words / 2;
        for(int i = 0; i < 2; i++){
            while( words_per_matrix > 0 ) {
                *reader = (WordData) {
                    .parameter_word = {
                        .p = {
                            *(uint64_t *)(char *)&(matrix_a_flat[0]), *(uint64_t *)(char *)&(matrix_a_flat[1]), // rows and columns of matrix A
                        },
                    },
                };
                reader = (WordData *)((char *)reader + sizeof(ParameterWord));
            }
        }
    } else {
        // if we have a C matrix, we interleave 3 parameters from each matrix
        // so we need to do some quick integer math to determine how many
        // parameters can be grouped and what is truly orphaned
        // this is done by taking the total operands and dividing by 3
        printf("DEBUG: total_operands=%zu, first_line_operands=%zu, first_page_operands=%zu, first_book_operands=%zu\n",
               total_operands, first_line_operands, first_page_operands, first_book_operands);
        while( total_operands > 2 ) {
            *reader = (WordData) {
                .parameter_word = {
                    .p = {
                        *(uint64_t *)(char *)&(matrix_a_flat[0]), *(uint64_t *)(char *)&(matrix_a_flat[1]), // rows and columns of matrix A
                    },
                },
            };
            reader = (WordData *)((char *)reader + sizeof(ParameterWord));
            *reader = (WordData) {
                .parameter_word = {
                    .p = {
                        *(uint64_t *)(char *)&(matrix_b_flat[0]), *(uint64_t *)(char *)&(matrix_b_flat[1]), // rows and columns of matrix B
                    },
                },
            };
            reader = (WordData *)((char *)reader + sizeof(ParameterWord));
            *reader = (WordData) {
                .parameter_word = {
                    .p = {
                        *(uint64_t *)(char *)&(matrix_c_flat[0]), *(uint64_t *)(char *)&(matrix_c_flat[1]), // rows and columns of matrix C
                    },
                },
            };
            reader = (WordData *)((char *)reader + sizeof(ParameterWord));
            total_operands -= 3;
            printf("DEBUG: Remaining total_operands=%zu\n", total_operands);
        }
    } 






    WordData * tail = NULL;
    if ( tail_parameters > 0 ) {
        tail = (WordData *)mg_alloc(sizeof(WordData) * (tail_parameters+1)/2);
    }

  
    uint64_t submatrix_start_x = 0;
    uint64_t submatrix_start_y = 0;
    uint64_t submatrix_end_x = matrix_a_cols - 1;
    uint64_t submatrix_end_y = matrix_a_rows - 1;
    reader = (WordData *)((char *)reader + sizeof(WordData));
    reader->parameter_word.p[0] = submatrix_start_y;
    reader->parameter_word.p[1] = submatrix_end_y;
    reader = (WordData *)((char *)reader + sizeof(ParameterWord));
    reader->parameter_word.p[0] = submatrix_start_x;
    reader->parameter_word.p[1] = submatrix_end_x;

    // for later memory optimization, we pack the data as local instructions, matrix A, B, and C, for A * B + C in 8 parameters inteleaved
    // this is accomplished by filling ParameterTwelveWord with maxtrix data while instructions provide local coordinate data in the hot loop
    return node_header;
}
