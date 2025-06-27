#ifndef ASSEMBLY_BACKEND_MEMORY_OPS_H
#define ASSEMBLY_BACKEND_MEMORY_OPS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#pragma pack(push,1)
#endif
typedef struct DiffBlockHeader {
    uint32_t magic;
    uint8_t  version;
    uint16_t type_id;
    uint8_t  flags;
    uint32_t payload_bytes;
    uint32_t pointer_index_offset;
    uint32_t metadata_offset;
    uint16_t stride;
    uint64_t block_id;
} 
#if defined(__GNUC__) || defined(__clang__)
__attribute__((packed))
#endif
DiffBlockHeader;
#ifdef _MSC_VER
#pragma pack(pop)
#endif

// Memory block history size for tracking last N resizes
#define MEMORY_BLOCK_HISTORY 8

typedef enum {
    MEM_BLOCK_FLAG_UNINITIALIZED          = 0x00,
    MEM_BLOCK_FLAG_DYNAMIC       = 0x01,
    MEM_BLOCK_FLAG_FIXED    = 0x02,
    MEM_BLOCK_FLAG_ENCODED     = 0x04,
    MEM_BLOCK_FLAG_MANAGED    = 0x08,
} MemBlockFlags;

typedef enum {
    MEM_BLOCK_HISTORY_TYPE_ROLLING = 0, // Rolling memory block
    MEM_BLOCK_HISTORY_TYPE_UNIQUE_ROLLING = 1, // Unique rolling memory block
    MEM_BLOCK_HISTORY_TYPE_ORDERS_OF_AVERAGES = 2, // elements of the window update every n^mth step
    MEM_BLOCK_HISTORY_TYPE_OUTLIERS = 3, // Outlier detection memory block
} MemBlockHistoryType;

typedef enum {
    // --- Classical deterministic encodings ---

    BIT_ENCODING_SCHEME_BINARY = 0,  
    // Standard binary encoding. Fastest hardware support.
    // Use case: raw performance, integer math, type-safe boundaries.
    // SIMD: Supports vectorized loads, masks, gathers natively.

    BIT_ENCODING_SCHEME_GREY_CODE = 1,  
    // Each successive value differs in only one bit.
    // Use case: Minimizes Hamming distance — ideal for gradient-compatible encodings.
    // SIMD: Differentiation-friendly, good for minimizing mutation distances in training loops.

    BIT_ENCODING_SCHEME_HUFFMAN = 2,  
    // Entropy-based compression based on frequency.
    // Use case: Reduces bit usage at cost of decoding speed.
    // SIMD: SIMD lookup tables possible, but requires decoding tree traversal.

    BIT_ENCODING_SCHEME_ELIAS_GAMMA = 3,
    // Unary-prefixed binary encoding for positive integers.
    // Use case: Extremely compact for small integers.
    // SIMD: Inefficient for parallel decode due to variable length.

    BIT_ENCODING_SCHEME_FIBONACCI = 4,
    // Uses Fibonacci numbers as base encoding units.
    // Use case: Compact, self-terminating, useful for sequences with known statistical decline.
    // SIMD: Difficult to decode in parallel due to carry dependencies.

    // --- Signed & Differential-optimized schemes ---

    BIT_ENCODING_SCHEME_ZIGZAG = 5,  
    // Maps signed integers to unsigned integers (used in Protobuf).
    // Use case: Works well when delta-encoding small positive and negative deltas.
    // SIMD: Ideal for packed delta streams.

    BIT_ENCODING_SCHEME_DELTA = 6,
    // Stores differences between successive values.
    // Use case: Time series, memory walks, data flow encoding.
    // SIMD: Works beautifully with prefix sum & AVX register chaining.

    BIT_ENCODING_SCHEME_BURROWS_WHEELER_TRANSFORM = 7,
    // Transforms data to be more compressible.
    // Use case: Combined with RLE or Huffman for data entropy reduction.
    // SIMD: Memory-heavy, great for block shuffling before compression.

    BIT_ENCODING_SCHEME_RUN_LENGTH = 8,
    // Replaces consecutive repeated values with value-count pairs.
    // Use case: Sparse or repetitive memory regions.
    // SIMD: Can be accelerated with compress/decompress intrinsics.

    // --- Multi-bit symbol map encodings ---

    BIT_ENCODING_SCHEME_TERNARY_BALANCED = 9,
    // Base-3 signed encoding (-1, 0, 1) with zero-centered logic.
    // Use case: Ideal for logic simulation, ternary arithmetic, or symbolic computation.
    // SIMD: Ternary lookup with AVX shuffle lanes.

    BIT_ENCODING_SCHEME_PERMUTATION = 10,
    // Encodes relative order or ranking of items.
    // Use case: Graph node reordering, probabilistic memory structures.
    // SIMD: Custom gather/scatter required.

    BIT_ENCODING_SCHEME_BITWISE_FOURIER = 11,
    // Represents values via their frequency-domain signature.
    // Use case: Bit-domain convolution, rhythm-like diff analysis.
    // SIMD: FFT + logic domain introspection.

    // --- Experimental / Neuroplastic / Adaptive schemes ---

    BIT_ENCODING_SCHEME_NEURAL_FIELD = 12,
    // Uses internal floating representation with differentiable parameters.
    // Use case: Model-aware compression and data reshaping.
    // SIMD: Requires on-the-fly learned functions or LUTs.

    BIT_ENCODING_SCHEME_CONTEXTUAL_BLOCK = 13,
    // Encodes using surrounding context to pick encoding dictionary.
    // Use case: Adaptive memory layout compression.
    // SIMD: Requires embedded context header per block.

    BIT_ENCODING_SCHEME_WAVELET_PACK = 14,
    // Encodes with low/high-frequency bit signatures (Haar or Daubechies).
    // Use case: Noise filtering, error resilience.
    // SIMD: Tree-based AVX execution using lanes.

    // Future-protected range for custom experimentals
    BIT_ENCODING_SCHEME_CUSTOM_RESERVED = 15

} BitEncodingScheme;

typedef enum {
    DIFF_TAPE_FORMAT_FLAG_NONE = 0x00, // No diff tape format
    DIFF_TAPE_FORMAT_FLAG_UNTIMED = 0x01, // Untimed diff tape format
    DIFF_TAPE_FORMAT_FLAG_TIMED_US = 0x02, // Timed diff tape format in microseconds
    DIFF_TAPE_FORMAT_FLAG_TIMED_NS = 0x04, // Timed diff tape format in nanoseconds
    DIFF_TAPE_FORMAT_FLAG_TIMED_MS = 0x08, // Timed diff tape format in milliseconds
    DIFF_TAPE_FORMAT_FLAG_TIMED_S = 0x10, // Timed diff tape format in seconds
    DIFF_TAPE_FORMAT_FLAG_GLOBAL_ITERATION = 0x20, // Global iteration-based diff tape format
    DIFF_TAPE_FORMAT_FLAG_LOCAL_ITERATION = 0x40, // Local iteration
    DIFF_TAPE_FORMAT_FLAG_FULL = 0x80, // Full diff tape format with all metadata
    DIFF_TAPE_FORMAT_FLAG_COMPRESSED = 0x100, // Compressed diff
    
} DiffTapeFormat;

// A header for a dynamic n*2^m *64B span
typedef struct {
    float n;            // initial n dimension (integer)
    float m;            // initial m exponent (integer)
    float growth_n;     // growth counter for n dimension
    float growth_m;     // growth counter for m exponent
    float decay_n;      // decay factor applied on shrink decisions
    float decay_m;      // decay factor applied on shrink decisions
    uint64_t timestamp_ns[MEMORY_BLOCK_HISTORY]; // timestamp of last resize in nanoseconds
    size_t payload_bytes[MEMORY_BLOCK_HISTORY];     // actual payload size in bytes
    void* diff_tape; // pointer to the diff tape for this span
    DiffTapeFormat diff_tape_flags; // is this span using a diff tape?
    BitEncodingScheme encoding_scheme; // Encoding scheme used for this span
    MemBlockHistoryType history_type; // Type of history tracking
    MemBlockFlags flags; // Flags for the memory block
} MemSpanHeader;

void* mg_encode_block(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags);
void* mg_decode_block(const void* encoded_block, size_t* out_payload_size);
void  mg_tensor_compare_64x64(const void* block_a, const void* block_b, float* out_diff_tensor);
const DiffBlockHeader* mg_peek_header(const void* block);

// SIDM-backed allocator interface
void* mg_alloc(size_t size);
void  mg_free(void* ptr);
// Allocate a span of size (floor(init_n) * 2^floor(init_m) * 64), with growth/decay factors
void* memops_span_alloc(float init_n, float init_m, float decay_factor);
// Free a span previously allocated with memops_span_alloc
void  memops_span_free(void* span_ptr);

#ifdef __cplusplus
}
#endif

#endif /* ASSEMBLY_BACKEND_MEMORY_OPS_H */
