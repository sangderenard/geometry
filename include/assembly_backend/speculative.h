#pragma once

#include "geometry/types.h"
#include "assembly_backend/core.h"
#include <stdint.h>

// -------------------------------------------
// Enums — symbolic, don't assume forced size
// -------------------------------------------

// Flags packed as bits in a uint8_t
typedef enum {
    REGISTER_WORD_FLAG_INTEGER             = 0x01,
    REGISTER_WORD_FLAG_REFERENCE           = 0x02,
    REGISTER_WORD_FLAG_IN_PLACE            = 0x04,
    REGISTER_WORD_FLAG_SCATTER             = 0x08,
    REGISTER_WORD_FLAG_GATHER_UNFULLFILLED = 0x10,
    REGISTER_WORD_FLAG_GATHER_FULLFILLED   = 0x20,
    REGISTER_WORD_FLAG_BURN_AFTER_READING  = 0x40,
    REGISTER_WORD_FLAG_OPERATOR_CODE       = 0x80
} RegisterWordFlag;

#define REGISTER_WORD_FLAG_GATHER_ANY \
    (REGISTER_WORD_FLAG_GATHER_UNFULLFILLED | REGISTER_WORD_FLAG_GATHER_FULLFILLED)

typedef enum {
    OPERATOR_TYPE_64_2_ADD_MULTIPLY  = 0x00,
    OPERATOR_TYPE_32_3_ADD_MULTIPLY  = 0x01,
    OPERATOR_TYPE_32_4_TRIGONOMETRIC = 0x02,
    OPERATOR_TYPE_64_3_ADD_MULTIPLY  = 0x03,
} OperatorType;

typedef enum {
    OPERATOR_TYPE_DEFAULT  = 0x00,
    OPERATOR_TYPE_EXTERNAL = 0x01
} OperatorSubstituteFlags;

// ----------------------------------------------
// Packed structs — sizes and field order matter
// ----------------------------------------------

// 16 bytes: instruction metadata word
typedef struct {
    uint8_t  flags;           // 0: RegisterWordFlag
    uint8_t  unused_0;        // 1: reserved
    uint8_t  unused_1;        // 2: reserved
    uint8_t  unused_2;        // 3: reserved
    uint8_t  operand_word;    // 4: originally, the location of the operand, now unused
    uint8_t  value_word;      // 5: originally, the location of the value, now unused
    uint8_t  operator_word;   // 6: originally, the location of the operator, now unused
    uint8_t  unused_3;        // 7: reserved, explicit pad for alignment
    uint64_t operator_type;   // 8–15: OperatorType (8 bytes)
} OperatorOperandPlanWord;

// 16 bytes: dataset description
typedef struct {
    uint16_t subsequent_rows;
    uint16_t pad0;
    uint32_t pad1;
    uint64_t next_dataset_ref;
} Dataset;

// 16 bytes: raw word
typedef struct {
    uint64_t raw_first_value;
    uint64_t raw_second_value;
} DatasetRawWord;

// 16 bytes: split words (for various use cases)
typedef struct { uint64_t input_output_val, diff_val; } DiffSplitWord;
typedef struct { uint64_t input_output_val, integrated_val; } IntegratedSplitWord;
typedef struct { uint64_t input_val, output_val; } InputOutputSplitWord;

typedef struct {
    uint64_t p[2];
} ParameterWord;

// 16 bytes: union of all row types
typedef union {
    Dataset            dataset;
    DatasetRawWord     dataset_raw;
    OperatorOperandPlanWord operator_operand_plan;
    DiffSplitWord      diff_split;
    IntegratedSplitWord integrated_split;
    InputOutputSplitWord input_output_split;
    ParameterWord      parameter_word; // 2 words, 16 bytes
} WordData;

// ------------------------------------
// 256 bytes: 16 x 16-byte words = Line
// ------------------------------------
typedef struct {
    WordData word[16];
} Line;

// 4096 bytes: 16 x 256-byte lines = Page
typedef struct {
    Line line[16];
} Page;

// -----------------------------------------
// Parameter block layouts, word granularity
// -----------------------------------------

typedef struct {
    uint64_t p[8];
} ParameterFourWord;



typedef struct {
    ParameterWord p[16];
} ParameterLine;

typedef struct {
    ParameterFourWord f[2]; // 8 words = 64 bytes
} ParameterEightWord;

typedef struct {
    ParameterFourWord f[3]; // 12 words = 96 bytes
} ParameterTwelveWord;

// ParameterPage: 16 x ParameterLine (each line: 16 x 2 x 8 = 256 bytes, page: 4096 bytes)
typedef struct {
    ParameterLine line[16];
} ParameterPage;

// ParameterBook: 8 x ParameterPage (each page: 4096 bytes, book: 32768 bytes)
typedef struct {
    ParameterPage page[8];
} ParameterBook;

// ---------------------------------------------------
// MatMulAddFuseInstruction-specific layouts
// ---------------------------------------------------
typedef union {
    uint64_t integer;
    double   floating_point;
} ParameterValue;
typedef struct {
    OperatorOperandPlanWord operator_operand_plan_row;
    Dataset dataset_row;
    uint64_t submatrix_row_start, submatrix_row_end;
    uint64_t submatrix_col_start, submatrix_col_end;
} MatMulAddFuseConfiguration;

// Each line: 16 x 16 = 256 bytes (metadata + parameter)
typedef struct {
    MatMulAddFuseConfiguration metadata;
    ParameterTwelveWord parameter_00; // 192 bytes
    // Padding if needed for alignment
} MatMulAddFuseInstructionLine;

// 16 lines/page
typedef struct {
    MatMulAddFuseInstructionLine instruction_line;
    ParameterLine parameters_lines[15];
} MatMulAddFuseInstructionPage;

// 8 pages/book
typedef struct {
    MatMulAddFuseInstructionPage instruction_page;
    ParameterPage parameters_pages[7];
} MatMulAddFuseInstructionBook;

// For quad node: union of parameter/instruction view
typedef union {
    MatMulAddFuseInstructionBook instructions;
    ParameterBook parameters;
} MatMulAddFuseInstructionsOrParameters;

// ------------------------------------------
// Quad node structure (for tree expansion)
// ------------------------------------------

struct MatMulAddFuseInstructionSetQuadNode; // forward declare
#define N 4 // Number of expansions in a quad node
#define BOOK_SIZE 2048 // 2048 words = 32768 bytes
typedef struct MatMulAddFuseInstructionSetNNode {
    struct MatMulAddFuseInstructionSetNNode *expansion[N/2*2]; // even only
    ParameterWord parameters[BOOK_SIZE-N/2]; // 2048 words = 32768 bytes
} ParameterBookWithNPointerHeader;

typedef union {
    MatMulAddFuseInstructionsOrParameters instructions;
    ParameterBookWithNPointerHeader parameters;
} MatMulAddFuseInstructionSetNNode;

// ---------------------------------------------------
// the first 16 bytes will tell us the operator type, flags, and other metadata
// next, the dataset line will describe whether there is another dataset after the current one
// then we have the coordinates inside the matrix if this instruction is for a larger matrix multiplication
// such as if it is a leaf node in a table
// these submatrix coordinates are 256 bits and are largely padding
// immediately following these coordinates we have twelve "words" of parameters
// which equates to 16 * 12 = 192 bytes and 24 actual parameters
// after that, it's presumed we needed more than one page of data, and would be dealing with
// a book of parameters, which follows the same structure as the MatMulAddFuseInstructionLine
// but no longer prepending instructions on each new page
// If more than one book is necessary, and the dataset didn't indicate a skip to a new dataset,
// additional books can be created that are only parameters, without the instruction metadata.
// And finally, a node can either be this data structure, or else pointers to other nodes
// Plus parameter space for purposes to be determined later in keeping with not being leaves.
// ---------------------------------------------------

typedef struct {
    MatMulAddFuseInstructionSetNNode *node; // Pointer to the node
    size_t subdivision_pointers; // Number of pointers in the node
    size_t matrix_a_cols; // Number of columns in the node
    size_t matrix_a_rows; // Number of rows in the node
    size_t matrix_b_cols; // Number of columns in the node
    size_t matrix_b_rows; // Number of rows in the node
    size_t matrix_c_cols; // Number of columns in the result matrix
    size_t matrix_c_rows; // Number of rows in the result matrix
} MatMulAddFuseInstructionSetNNodeHeader;

MatMulAddFuseInstructionSetNNodeHeader *create_matmul_add_fuse_instruction_set_n_node(
    size_t subdivision_pointers, size_t matrix_a_cols, size_t matrix_a_rows,
    size_t matrix_b_cols, size_t matrix_b_rows, size_t matrix_c_cols, size_t matrix_c_rows,
    ParameterValue* matrix_a_flat, ParameterValue* matrix_b_flat, ParameterValue* matrix_c_flat);

// ---------------------------------------------------
// Error codes for MatMulAddFuseInstructionSetNNodeHeader creation
// ---------------------------------------------------
typedef enum {
    SPEC_ERR_NONE = 0,            // No error
    SPEC_ERR_ALLOC_HEADER,        // Failed to allocate node header
    SPEC_ERR_ALLOC_NODE,           // Failed to allocate node storage
    SPEC_ERR_ALLOC_NODE_ASSIGNMENT, // Failed to assign node storage
} SpeculativeError;

// Last error encountered during speculative node creation
extern SpeculativeError speculative_error;

// ---------------------------------------------------
// Instruction page generator
// ---------------------------------------------------

// ---------------------------------------------------
// Static asserts for development safety
// ---------------------------------------------------

_Static_assert(sizeof(OperatorOperandPlanWord) == 16, "OperatorOperandPlanWord must be 16 bytes");
_Static_assert(sizeof(Dataset) == 16, "Dataset must be 16 bytes");
_Static_assert(sizeof(DatasetRawWord) == 16, "DatasetRawWord must be 16 bytes");
_Static_assert(sizeof(WordData) == 16, "WordData must be 16 bytes");
_Static_assert(sizeof(Line) == 256, "Line must be 256 bytes");
_Static_assert(sizeof(Page) == 4096, "Page must be 4096 bytes");
_Static_assert(sizeof(MatMulAddFuseInstructionPage) == sizeof(Page), "MatMulAddFuseInstructionPage must match Page size");
_Static_assert(sizeof(MatMulAddFuseInstructionBook) == sizeof(ParameterBook), "MatMulAddFuseInstructionBook must match ParameterBook size");
_Static_assert(sizeof(ParameterBook) == sizeof(MatMulAddFuseInstructionBook), "ParameterBook and MatMulAddFuseInstructionBook size mismatch");

