import ctypes
import string
from typing import Union
from enum import IntFlag, auto

from sympy import Integer

# Left‐wall flags
class LeftWallFlags(IntFlag):
    LOCK       = auto()
    ELASTIC    = auto()
    PERMEABLE  = auto()
    REFLECTIVE = auto()

# Right‐wall flags
class RightWallFlags(IntFlag):
    LOCK       = auto()
    ELASTIC    = auto()
    PERMEABLE  = auto()
    REFLECTIVE = auto()

# Whole‐cell flags
class CellFlags(IntFlag):
    ZERO_SUM = auto()
    INERT    = auto()
    SOURCE   = auto()
    SINK     = auto()
    ADHESION = auto()

# System‐wide policies
class SystemFlags(IntFlag):
    AUTO_EXPAND    = auto()
    AUTO_SHRINK    = auto()
    PRESERVE_LOCKS = auto()
    LOG_EVENTS     = auto()
    TRACK_FRAG     = auto()

# New Cell class definition for testing
class Cell:
    def __init__(self, stride, left, right, len, profile='default', leftmost=None, rightmost=None):
        self.len = len
        self.label = f"cell_{id(self)}"
        self.salinity = 0
        self.temperature = 0
        self.leftmost = leftmost
        self.rightmost = rightmost
        #buf = ctypes.create_string_buffer((self.len + 7)// 8)
        #self.obj_map = ctypes.addressof(buf)
        self.left = left
        self.right = right
        self.compressible = 1
        flags = DEFAULT_FLAG_PROFILES.get(profile, DEFAULT_FLAG_PROFILES['default'])
        self.l_wall_flags = flags['left_wall']
        self.r_wall_flags = flags['right_wall']
        self.c_flags      = flags['cell']
        self.system_flags = flags['system']
        self.l_solvent_permiability = 1
        self.r_solvent_permiability = 1
        self.injection_queue = 0
        self.resize_queue = 0
        self.stride = stride


        
        self.pressure = 0
        # Retain reference to avoid garbage collection
        self._buf = None#buf



# ─── Default flag‑profiles ─────────────────────────────────────────────────────
DEFAULT_FLAG_PROFILES = {
    'default': {
        'left_wall':   LeftWallFlags.ELASTIC,
        'right_wall':  RightWallFlags.ELASTIC,
        'cell':        CellFlags.ZERO_SUM,
        'system':      SystemFlags.AUTO_EXPAND | SystemFlags.PRESERVE_LOCKS,
    },
    'rigid_partition': {
        'left_wall':   LeftWallFlags.LOCK,
        'right_wall':  RightWallFlags.LOCK,
        'cell':        CellFlags.ZERO_SUM | CellFlags.INERT,
        'system':      SystemFlags.PRESERVE_LOCKS,
    },
    'open_pipe': {
        'left_wall':   LeftWallFlags.PERMEABLE,
        'right_wall':  RightWallFlags.PERMEABLE,
        'cell':        CellFlags.ZERO_SUM,
        'system':      SystemFlags.AUTO_SHRINK,
    },
    'source_driven': {
        'left_wall':   LeftWallFlags.ELASTIC,
        'right_wall':  RightWallFlags.ELASTIC,
        'cell':        CellFlags.SOURCE,
        'system':      SystemFlags.AUTO_EXPAND,
    },
    'sink_driven': {
        'left_wall':   LeftWallFlags.ELASTIC,
        'right_wall':  RightWallFlags.ELASTIC,
        'cell':        CellFlags.SINK,
        'system':      SystemFlags.AUTO_SHRINK,
    },
    'adhesive_net': {
        'left_wall':   LeftWallFlags.ELASTIC,
        'right_wall':  RightWallFlags.ELASTIC,
        'cell':        CellFlags.ZERO_SUM | CellFlags.ADHESION,
        'system':      SystemFlags.TRACK_FRAG,
    },
}

LEFT_WALL = Cell(
    stride=1, left=0, right=0, len=0, profile='rigid_partition', rightmost=0, leftmost=0
)
RIGHT_WALL = Cell(
    stride=1, left=0, right=0, len=0, profile='rigid_partition'
)

def build_fill_buffer(fill_value: int, length_bits: int) -> bytearray:
    """
    Create a buffer of exactly length_bits, all set to fill_value (0 or 1).
    """
    buf = bytearray((length_bits + 7) // 8)
    if fill_value:
        # set every bit to 1
        for i in range(length_bits):
            byte_i = i // 8
            bit_off = 7 - (i % 8)
            buf[byte_i] |= (1 << bit_off)
    # clear any pad bits beyond length_bits
    return mask_padding_bits(buf, length_bits)
from typing import Union, Iterable, Tuple

def stamp(
    raw: bytearray,
    indices: Iterable[Union[int, Tuple[int, int], Tuple[int, int, int]]],
    default_stride: int,
    default_value: int = 1
) -> bytearray:
    """
    In-place stamp bits at each specified index. 
    Supports:
      - int: uses default_stride and default_value
      - (index, stride): uses given stride and default_value
      - (index, stride, value): uses given stride and value
    
    Args:
      raw: The target bytearray to mutate.
      indices: Iterable of int or 2-/3-tuples.
      default_stride: Fallback stride for plain int indices.
      default_value: Fallback fill (0 or 1) for plain int or 2-tuples.
    Returns:
      The modified `raw` bytearray.
    """
    for entry in indices:
        # Unpack index, stride, and fill_value based on entry shape
        if isinstance(entry, (tuple, list)):
            if len(entry) == 2:
                gap, stride = entry
                value = default_value
            elif len(entry) == 3:
                gap, stride, value = entry
            else:
                raise ValueError(f"stamp entry {entry} must be int, (idx,stride), or (idx,stride,value)")
        else:
            gap = entry
            stride = default_stride
            value = default_value
        
        # Build a fill buffer once for this gap
        fill_buf = build_fill_buffer(value, stride)
        write_bit_region(raw, gap, fill_buf, stride)
    
    return raw


def extract_bit_region(data: bytes, start_bit: int, length: int) -> bytearray:
    # PAD_CLIP: Extracting a region of bits with clipping for odd bit lengths, ensuring proper buffer size.
    #print(data)
    if not isinstance(data, (bytes, bytearray)):
        if hasattr(data, 'raw'):
            data = data.raw
        elif isinstance(data, int):
            # This is a c pointer
            data = ctypes.string_at(data, intceil(length))
        else:
            data = bytes(data)
    out = bytearray((length + 7) // 8)
    for i in range(length):
        src_bit = (data[(start_bit + i) // 8] >> (7 - ((start_bit + i) % 8))) & 1
        #print(f"Extracting bit {i}: {src_bit} from data at bit offset {start_bit + i}")
        #print(f"putting in i//8: {i // 8}")
        out[i // 8] |= src_bit << (7 - (i % 8))
    return out

def mask_padding_bits(buf: bytearray, length_bits: int) -> bytearray:
    """
    Clear any bits in `buf` beyond the first `length_bits` bits.
    """
    total_bits = len(buf) * 8
    pad_bits = total_bits - length_bits
    if pad_bits <= 0:
        return buf
    # Clear the last 'pad_bits' bits
    for bit_index in range(length_bits, total_bits):
        byte_i = bit_index // 8
        bit_offset = 7 - (bit_index % 8)
        buf[byte_i] &= ~(1 << bit_offset)
    return buf

def write_bit_region(target: bytearray,
                     start_bit: int,
                     buf: Union[bytes, bytearray],
                     length_bits: int) -> None:
    """
    Write exactly `length_bits` bits from `buf` into `target` starting at bit offset `start_bit`.
    Bits in `buf` are read MSB-first within each byte.
    """
    for i in range(length_bits):
        # Determine source bit value
        src_byte = buf[i // 8]
        src_bit_val = (src_byte >> (7 - (i % 8))) & 1

        # Determine target location
        dest_bit = start_bit + i
        dest_byte_i = dest_bit // 8
        dest_bit_offset = 7 - (dest_bit % 8)

        if src_bit_val:
            target[dest_byte_i] |= (1 << dest_bit_offset)
        else:
            target[dest_byte_i] &= ~(1 << dest_bit_offset)

STRIDE = 12
CELL_COUNT = 1
MASK_BITS_TO_DATA_BITS = 16
TEST_SIZE_STRIDE_TIMES_UNITS = (STRIDE ** 2 * 8 * STRIDE * CELL_COUNT) // ( 8 * STRIDE * CELL_COUNT)
assert TEST_SIZE_STRIDE_TIMES_UNITS % STRIDE == 0
# Simulator class to coordinate simulation steps
def intceil(value: int, multiple: int = 8) -> int:
    """
    Round `value` up to the nearest multiple of `multiple`.
    """
    if multiple <= 0:
        raise ValueError("Multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple

from salinepressure import SalineHydraulicSystem



class Simulator:
    FORCE_THRESH = .5
    LOCK = 0x1
    ELASTIC = 0x2

    def __init__(self, cells):
        self.assignable_gaps = {}
        self.cells = cells
        self.input_queues = {}
        self.system_pressure = 0
        self.elastic_coeff = 0.1
        self.mask_bit_length = TEST_SIZE_STRIDE_TIMES_UNITS * CELL_COUNT
        self.mask_byte_length = lambda: intceil(self.mask_bit_length) // 8
        self.data_bit_length = self.mask_bit_length * MASK_BITS_TO_DATA_BITS
        self.data_byte_length = lambda: intceil(self.data_bit_length) // 8
        self.data = bytearray(self.data_byte_length())
        self.bitmask = bytearray(self.mask_byte_length())
        self.locked_data_regions = []
        # create one C-array view on that same bytearray:
        self._data_buf  = (ctypes.c_ubyte * self.data_byte_length()).from_buffer(self.data)
        self._data_base = ctypes.addressof(self._data_buf)
        self._bitmask_buf = (ctypes.c_ubyte * len(self.bitmask)).from_buffer(self.bitmask)
        self._bitmask_base = ctypes.addressof(self._bitmask_buf)

        self.s_exprs = [Integer(0) for _ in range(CELL_COUNT)]
        self.p_exprs = [Integer(1) for _ in range(CELL_COUNT)]

        self.engine = None
        self.fractions = None

        self.run_saline_sim()
        # 3) Print/use them
        print(self.fractions)  # → a list of N floats summing to 1.0
    def run_saline_sim(self):
        # 1) Instantiate engine with your per‐cell salinity & pressure expressions (or plain numbers)
        self.engine = SalineHydraulicSystem(
            self.s_exprs,           # e.g. [Integer(s0), Integer(s1), …]
            self.p_exprs,           # e.g. [Integer(p0), Integer(p1), …]
            width=self.mask_bit_length, # the total bit‐space you’re dividing
            chars=[chr(97+i) for i in range(CELL_COUNT)],
            tau=5, math_type='int',
            int_method='adams',
            protect_under_one=True,
            bump_under_one=True
        )

        # 2) Ask for the equilibrium fractions at t=0
        self.fractions = self.engine.equilibrium_fracs(0.0)
        for cell in cells:
            if cell.salinity == 0:
                cell.salinity = 1
        
        necessary_size = sum(cell.salinity for cell in cells if hasattr(cell, 'salinity') and cell.salinity > 0)
        print(f"Necessary size: {necessary_size} bits")
        if self.mask_bit_length < necessary_size:
            offsets = [(cell.rightmost - cell.leftmost)//2+cell.leftmost for cell in cells if hasattr(cell, 'leftmost')]
            sizes = [(cell.salinity) for cell in cells if hasattr(cell, 'salinity') and cell.salinity > 0]
            print(f"Expanding system to fit {necessary_size} bits:")
            print(offsets)
            print(sizes)
            self.expand(offsets, sizes, cells)

        self.snap_cell_walls(self.cells)

    def get_cell_mask(self, cell: Cell) -> bytearray:
        print(f"Extracting mask for cell {cell.label} from {cell.left} to {cell.right}")
        return extract_bit_region(self.bitmask, cell.left, cell.right - cell.left)
    def set_cell_mask(self, cell: Cell, mask: bytearray) -> None:
        write_bit_region(self.bitmask, cell.left, mask, cell.right - cell.left)
    def pull_cell_mask(self, cell):
        cell._buf = self.get_cell_mask(cell)
    def push_cell_mask(self, cell):
        self.set_cell_mask(cell, cell._buf)
    

    def evolution_tick(self, cells):
        # Use the saline pressure system to set cell proportions
        # inside minimize(…) or evolution_tick(…), once every cell.pressure & .salinity are up‑to‑date:

        # rebuild the engine’s callables so they always return the current attributes
        self.engine.s_funcs = [
            (lambda _t, s=cell.salinity: s)
            for cell in cells
        ]
        self.engine.p_funcs = [
            (lambda _t, p=cell.pressure: p)
            for cell in cells
        ]


        fractions = self.engine.equilibrium_fracs(0.0)
        total_space = self.mask_bit_length
        current_left = 0
        for cell, frac in zip(cells, fractions):
            new_width = max(cell.salinity, int(total_space * frac))
            cell.left = current_left
            cell.leftmost = cell.left
            cell.right = current_left + new_width
            cell.rightmost = cell.right
            #cell.pressure = 0  # reset pressure after reallocation
            current_left = cell.right
            print(f"Cell {cell.label} resized to {cell.left} - {cell.right} ({new_width} bits)")
        self.snap_cell_walls(cells)
        self.print_system(cells, data_buffer=self.data)

    def print_system(self, cells, data_buffer=None, width=80):
        """
        Draw the entire address space scaled to `width` characters,
        and report total size and fragmentation percentage.
        """
        # total bits across all cells
        total_bits = self.data_bit_length
        if total_bits == 0:
            print("<empty>")
            return

        labels = string.ascii_lowercase

        # build per-bit info using only extract_bit_region
        bit_info = []
        for bit in range(total_bits):
            info = (None, False, False)
            for idx, cell in enumerate(cells):
                if cell.left <= bit < cell.right:
                    # mask bit:
                    mset = bool(extract_bit_region(self.bitmask, bit//MASK_BITS_TO_DATA_BITS, 1)[0])
                    # data bit:
                    dset = False
                    if data_buffer is not None:
                        dset = bool(extract_bit_region(self.data, bit, 1)[0])
                    info = (idx, mset, dset)
                    break
            bit_info.append(info)

        # fragmentation: only bits inside cells where mask==0
        free_bits = sum(1 for idx, m, _ in bit_info if idx is not None and not m)
        free_regions = []
        run = 0
        for idx, m, _ in bit_info:
            if idx is not None and not m:
                run += 1
            else:
                if run:
                    free_regions.append(run)
                    run = 0
        if run:
            free_regions.append(run)
        frag_pct = (1 - max(free_regions)/free_bits) * 100 if free_bits else 0.0

        # reporting
        size_string = f"Total size: {total_bits} bits ({total_bits/8:.2f} bytes, mask bits: {self.mask_bit_length})"
        free_string = f"Free: {free_bits} bits; fragmentation: {frag_pct:.2f}%"

        # render map
        out = []
        for col in range(width):
            mid = min(int(((col + 0.5) * total_bits) / width), total_bits - 1)
            idx, m, d = bit_info[mid]
            if idx is None:
                out.append('.')
            else:
                c = labels[idx % len(labels)]
                if m or d:
                    c = c.upper()
                out.append(c)

        print(''.join(out), size_string, free_string)


    def count_from(self, bytes, alignment_offset, direction, n=64, m=64, k=1, j=8, padding=0):
        """Count bits from a side n deep and report pattern in tuples, none for missing instances."""
        return_pattern = []
        def verify_pattern_tuple_one_away(pattern_tuple):
            nonlocal direction, j
            return pattern_tuple[1] % j == (j-1 if direction == 'right' else 0)
        
        whole_spaces = set()  # track whole spaces
        h=0        
        proper_len = len(bytes) * 8 - padding
        initial_h = alignment_offset + (proper_len)// j * j
        if direction == 'left':
            h = initial_h
        else:
            h = alignment_offset
        #print(f"bytes size: {len(bytes)}, alignment_offset: {alignment_offset}, direction: {direction}, n: {n}, m: {m}, k: {k}, j: {j}")
        n = max(n, proper_len // 2)
        n = (n + j - 1)// j * j
        n = min(n, j)
        
        #print(f"Counting from {direction} with n={n}, m={m}, k={k}, j={j}")
        most_recent_value = None
        ##print(f"Starting count_from with bytes: {bytes.hex()}, alignment_offset: {alignment_offset}, direction: {direction}, n: {n}, m: {m}, k: {k}, j: {j}")
        for i in range(n):
            
            if direction == 'left':
                idx = proper_len - i - 1
                #print(f"i={i}, proper_len={proper_len}, idx={idx}")
                bit = (bytes[(idx) // 8] >> (7 - ((idx) % 8))) & 1   
                #print(f"from len(bytes) * 8 - 0 - 1 = {len(bytes) * 8 - 0 - 1}, idx={idx}, bit={bit}")
                ###print(f"to len(bytes) * 8 - n - 1 = {len(bytes) * 8 - n - 1}, idx={idx}, bit={bit}")
                # THIS ABOVE WAS AN OFF BY ONE HACK FIX,
                # UNTIL IT CAN BE ADEQUATELY EXPLAINED WHY IT'S NEEDED
                # DO NOT REMOVE THIS COMMENT

                #print(f"We're starting at len(bytes) * 8 - i - 1 = {len(bytes) * 8 - i - 1}, idx={idx}, bit={bit}")



            elif direction == 'right':
                idx = i
                bit = (bytes[idx // 8] >> (7 - (idx % 8))) & 1

                #print(f"We're starting at i = {i}, idx={idx}, bit={bit}")
            else:
                raise ValueError("Direction must be 'left' or 'right'")
            
            #print(f"Index: {idx}, Bit: {bit}, h: {h}, m: {m}, k: {k}")
# remove check for idx out of bounds, it should not happen
#            if idx < 0 or idx >= len(bytes) * 8:
#                return return_pattern, whole_spaces
            if not bit:
                k -= 1
                #print(f"Bit is 0, k decremented to {k}")
                if return_pattern and return_pattern[-1][0] == 0:
                    if direction == 'left':
                        h -= 1
                    else:
                        h += 1
                    if direction == 'right' and h % j == j-1:
                        #print(f"Found whole space at index {idx}")
                        if verify_pattern_tuple_one_away(return_pattern[-1]):
                            whole_spaces.add(idx-j+1)
                    elif direction == 'left' and h % j == 0:
                        #print(f"Found whole space at index {idx}")
                        if verify_pattern_tuple_one_away(return_pattern[-1]):
                            whole_spaces.add(idx+1)
                elif not return_pattern:
                    #print(f"Starting new pattern with bit {bit} at index {idx}, h={h}, m={m}, k={k}")
                    if direction == 'left':
                        h = initial_h
                    else:
                        h = alignment_offset

            else:
                if return_pattern and return_pattern[-1][0] == 0 and direction == "left":
                    if return_pattern[-1][1] % j == 0:
                        #print(f"Found whole space at index {idx}")
                        if verify_pattern_tuple_one_away(return_pattern[-1]):
                            whole_spaces.add(idx+1)
                elif return_pattern and return_pattern[-1][0] == 0 and direction == "right":
                    if return_pattern[-1][1] % j == j - 1:
                        #print(f"right, bit is 1, count is {return_pattern[-1][1]}, j is {j}, direction is {direction}")
                
                        # you can remove this exit if you encounter it
                        # but it should not happen in the current algorithm
                        if verify_pattern_tuple_one_away(return_pattern[-1]):
                            whole_spaces.add(idx - j + 1)

                        #I don't conceptually understand why this is necessary
                        # but I do rationally understand we were setting our
                        # marker off idx, retrospectively, so we need to look back
                        # even if we just changed to 1s

                        
                #assert h % j == 0, f"Left alignment offset {alignment_offset} + bytes size {len(bytes) * 8} is not aligned with stride {j}"
            
                if direction == 'left':
                    h = initial_h
                else:
                    h = alignment_offset
            if return_pattern and bit == most_recent_value:
                #print(f"Continuing pattern with bit {bit} at index {idx}, h={h}, m={m}, k={k}")
                rt_bit, count = return_pattern[-1]
                return_pattern[-1] = (rt_bit, count + 1)
                if bit == 0 and count % j == j - 1 and direction == 'right':
                    #print(f"bit is 0, count is {count}, j is {j}, direction is {direction}")
                    if verify_pattern_tuple_one_away(return_pattern[-1]):
                        whole_spaces.add(idx - j + 1)
                ##    assert False, "This should have been captured as a whole space"
                if i > 0 and bit == 0 and count % j == 0 and direction == 'left':
                    #print(f"bit is 0, count is {count}, j is {j}, direction is {direction}")
                    if verify_pattern_tuple_one_away(return_pattern[-1]):
                        whole_spaces.add(idx + 1)
                    #assert False, "This should have been captured as a whole space"
            else:
                #print(f"Adding new pattern with bit {bit} at index {idx}, h={h}, m={m}, k={k}")
                return_pattern.append((bit, 1))
                m -= 1
                most_recent_value = bit
            
            # k tracking not aligned with stride, not sure what to do here
            if m <= 0:
                #print(f"m is 0, breaking out of loop at index {idx}")
                break
        if return_pattern and return_pattern[-1][0] == 0 and direction == "left":
            if return_pattern[-1][1] % j == 0:
                #print(f"last pattern is 0 and aligned with stride at index {idx}")
                if verify_pattern_tuple_one_away(return_pattern[-1]):
                    whole_spaces.add(idx) #this is preemptive, we will not have a 1 at the end
        elif return_pattern and return_pattern[-1][0] == 0 and direction == "right":
            if return_pattern[-1][1] % j == j-1:
                pass #do we need this one too? maybe?
                #print(f"last pattern is 1 and aligned with stride at index {idx}")
        return return_pattern, whole_spaces, n

    def center_search(self, bytes, alignment_stride, alignment_offset, n=64, depth=-1, max_depth=None, bit_base_offset=0, padding=0):
        """
        Recursively search for stride-aligned zero blocks from the center outward.
        All returned bit indices are global, not local to the bytes slice.
        """
#        if max_depth is None:
#            max_depth = (len(bytes) * 8) // (2 * alignment_stride)
#        if depth > max_depth:
#            return set(), set()

        if depth == 0 or n < alignment_stride:
            return set()
        
        if max_depth is None:
            max_depth = (len(bytes) * 8) // (2 * alignment_stride)

        if depth == -1:
            bit_base_offset = alignment_offset
            depth = max_depth

        # PAD_CLIP: Adjust region_bits by subtracting 'padding' to remove extra clipped bits.
        region_bits = len(bytes) * 8 - padding
        assert region_bits % alignment_stride == 0, f"Region bits {region_bits} is not aligned with stride {alignment_stride}: %: {region_bits % alignment_stride}"
        center = (((region_bits // 2) // alignment_stride) * alignment_stride)
        #center = (((region_bits // 2 + alignment_stride - 1) // alignment_stride) * alignment_stride)
        left_span_start = max(0, center - depth * alignment_stride)
        #print(f"Center: {center}, left_span_start: {left_span_start}, depth: {depth}, alignment_stride: {alignment_stride}, region_bits: {region_bits}")
        #assert left_span_start >= 0, f"Left span start {left_span_start} is negative, center: {center}, depth: {depth}, alignment_stride: {alignment_stride}"
        assert left_span_start % alignment_stride == 0, f"Left span start {left_span_start} is not aligned with stride {alignment_stride}"
        right_span_start = min((center + depth * alignment_stride), region_bits)

        ##print(f"Center: {center}, left_span_start: {left_span_start}, right_span_start: {right_span_start}, alignment_stride: {alignment_stride}, region_bits: {region_bits}")
        

        slots = set()
        left_gaps = set()
        right_gaps = set()
        short_bytes = extract_bit_region(
            bytes, left_span_start, right_span_start - left_span_start
        )
        #print(f"Short bytes extracted: {short_bytes.hex()}, left_span_start: {left_span_start}, right_span_start: {right_span_start}, region_bits: {region_bits}")
        #short_bytes = bytes[left_span_start // 8:(right_span_start+7)// 8]
        # PAD_CLIP: Compute short_padding as the difference between the total bits in short_bytes and the exact bit span.
        short_padding = len(short_bytes) * 8 - (right_span_start - left_span_start)
        #print(f"Short bytes: {short_bytes.hex()}, left_span_start: {left_span_start}, right_span_start: {right_span_start}, short_padding: {short_padding}")

        assert (len(short_bytes) * 8 - short_padding)%alignment_stride == 0


        if len(short_bytes) == 0:
            ##print(f"Short bytes is empty, returning empty sets for depth {depth}")
            return set()

        # Left span
        if left_span_start >= 0 and left_span_start + alignment_stride <= region_bits:
            left_pattern, left_gaps, left_n = self.count_from(short_bytes, left_span_start, 'right', n=alignment_stride, j=alignment_stride, padding=short_padding)
            # Convert relative to absolute (from this region)
            ##print(f"Left span start: {left_span_start}, alignment_stride: {alignment_stride}, region_bits: {region_bits}")
            #print(f"Left gaps found: {left_gaps}")
            left_gaps_abs = {left_span_start + gap for gap in left_gaps}
            assert(left_gap % alignment_stride == 0 for left_gap in left_gaps_abs), f"Left gaps {left_gaps_abs} are not aligned with stride {alignment_stride}"
            slots |= left_gaps_abs
            ##print(f"Left pattern: {left_pattern}, left_n: {left_n}, left_gaps_abs: {left_gaps_abs}")

        # Right span
        if right_span_start + alignment_stride >= 0 and right_span_start + alignment_stride <= region_bits + alignment_offset:# and right_span_start != left_span_start:
            right_pattern, right_gaps, right_n = self.count_from(short_bytes, left_span_start, 'left', n=alignment_stride, j=alignment_stride, padding=short_padding)
            ##print(f"Right span start: {right_span_start}, alignment_stride: {alignment_stride}, region_bits: {region_bits}")
            ##print(f"Right gaps found: {right_gaps}")
            right_gaps_abs = {left_span_start + gap for gap in right_gaps}
            slots |= right_gaps_abs
            ##print(f"Right pattern: {right_pattern}, right_n: {right_n}, right_gaps_abs: {right_gaps_abs}")
            assert(right_gap % alignment_stride == 0 for right_gap in right_gaps_abs), f"Right gaps {right_gaps_abs} are not aligned with stride {alignment_stride}"
            
        #else:
            ##print(f"Skipping right span, either out of bounds or same as left span: {right_span_start} == {left_span_start}")
        # Recurse outward
        if depth > 0:
            slots |= self.center_search(
                bytes, alignment_stride, alignment_offset,
                n=n, depth=depth - 1, max_depth=max_depth, bit_base_offset=bit_base_offset-alignment_stride, padding=padding
            )
        
        return slots

    def contiguate(self, raw, pattern, fragmented_slice, stride):
        # fragmented slice contains mixed data, indices not already spoken for
        # as recipients for new data

        # pattern contains tuples of (bit, count) where bit is 0 or 1
        # and count is the number of consecutive bits of that type
        
        

        
        contiguous_strides = [pattern[i][1] for i in range(len(pattern)) if pattern[i][0] == 1]
        contiguous_strides = sorted(contiguous_strides, reverse=True)
        output = []
        junk = []

        i = 0
        print(f"Fragmented slice: {fragmented_slice}")
        print(f"Pattern: {pattern}, stride: {stride}, raw length: {len(raw)}")
        for cluster in pattern:
            if cluster[0] == 1:
                is_junk = cluster[1] % stride
                if is_junk == 0:
                    print(f"Contiguate: cluster={cluster}, stride={stride}, raw length={len(raw)}")
                    for j in range(cluster[1] // stride):
                        pointer_offset = i + j * stride

                        ##print(f"Contiguate: pointer_offset={pointer_offset}, stride={stride}, raw length={len(raw)}")
                        ##print(f"Pointer Offset to stride alignment: {pointer_offset % stride} == 0")
                        
                        mask_data    = extract_bit_region(raw,            pointer_offset, stride)
                        backing_data = extract_bit_region(self.data,      pointer_offset * MASK_BITS_TO_DATA_BITS, stride * MASK_BITS_TO_DATA_BITS)

                        ##print(f"Mask data: {mask_data.hex()}, \nBacking data: {backing_data.hex()}")
                        
                        # Convert to immutable bytes to avoid TypeError
                        pointer = ctypes.addressof(ctypes.create_string_buffer(bytes(backing_data)))
                        #output.append((pointer, stride))
                        i += stride
                        
                else:
                    print(f"Contiguate: cluster={cluster}, stride={stride}, raw length={len(raw)}")
                    junk.append(raw[i:i + cluster[1]])
                    i += cluster[1]
                    assert False, "Junk data found in contiguate function, this should not happen with the current algorithm"
            else:
                # This is a gap, we don't care about it
                i += cluster[1]

        # output is by definition oddball data, because our stride is the length
        # of the objects we're dealing with, so if we are here, with a pattern
        # of 1s, if it's not a perfect integer multiple of the stride,
        # something is wrong with it.
        return output, junk
                
        
    def minimize(self, cells):
        
        system_pressure = 0
        raws = {}
        for i, cell in enumerate(cells):
            # Instead of reinitializing raw from ctypes.string_at(cell.obj_map, cell.len)
            # use the persistent buffer content that was updated last round:
            self.pull_cell_mask(cell)  # Ensure the cell's mask is up-to-date
            raw = cell._buf  # Use the updated persistent buffer

            #calculate forces into pressures
            #add to volumetric pressure
            #keep the metaphor loose because this is actually a simple swap algorithm
            known_gaps = set()
            #if cell.obj_map is None:
                ###print(f"Cell {cell.label} has no object map, skipping.")
                #continue
            #raw = bytearray(ctypes.string_at(cell.obj_map, cell.len))
            left_resistive_force = 0
            right_resistive_force = 0
            center_chances = 0
            pressure = 0
            ##print(f"cell injection queue: {cell.injection_queue}")
            ###print(f"Processing cell {cell.label} with raw data: {raw.hex()}")
            assert cell.left % cell.stride == 0, f"Cell {cell.label} left {cell.left} is not aligned with stride {cell.stride}"
            assert cell.right % cell.stride == 0, f"Cell {cell.label}   right {cell.right} is not aligned with stride {cell.stride}"

            self.padding = intceil(cell.right - cell.left) - (cell.right - cell.left)
            assert self.padding == len(raw) * 8 - (cell.right - cell.left), f"Padding {self.padding} does not match raw length {len(raw) * 8} and cell right-left {cell.right - cell.left}"
            if cell.left != cell.right:

                print(f"Cell {cell.label} left: {cell.left}, right: {cell.right}, stride: {cell.stride}, padding: {self.padding}")
                print(f"Cell {cell.label} raw length: {len(raw) * 8}")
                print(f"self.mask_bit_length: {self.mask_bit_length}, self.data_bit_length: {self.data_bit_length}")
                left_pattern, left_gaps, left_flat_length = self.count_from(raw, cell.left, 'right', j=cell.stride, padding = self.padding)
                right_pattern, right_gaps, right_flat_length = self.count_from(raw, cell.left, 'left', j=cell.stride, padding = self.padding)
                cell.leftmost = cell.left
                cell.rightmost = cell.right
                for pattern in left_pattern:
                    if pattern[0] == 1:
                        cell.leftmost += cell.left + pattern[1] - 1
                        break
                for pattern in right_pattern:
                    if pattern[0] == 1:
                        cell.rightmost -= cell.right - pattern[1] + 1
                        break
                #print(f"Cell {cell.label} left flat length: {left_flat_length}, right flat length: {right_flat_length}")
                center_gap = (cell.right - cell.left) - left_flat_length - right_flat_length
                ##print(f"Cell {cell.label} center gap: {center_gap}")
                center_chances = max(0, center_gap // cell.stride)
                assert center_chances >= 0, f"Cell {cell.label} center chances {center_chances} is negative, check stride and gap calculation"
                #print(f"Cell {cell.label} left pattern: {left_pattern}, right pattern: {right_pattern}, center chances: {center_chances}")
                if len(left_pattern) == 1:
                    cell.compressible = raw[0] == 0
                    if False and cell.compressible:
                        ##print(f"Cell {cell.label} is compressible, setting left/right flags.")
                        
                        known_gaps = list(range((cell.left + cell.stride - 1)//cell.stride * cell.stride, cell.left+left_flat_length, cell.stride))+list(range(((cell.right - cell.left - right_flat_length)+cell.stride-1)//cell.stride * cell.stride, cell.right // cell.stride * cell.stride, cell.stride))
                if False and cell.compressible == 0:
                    pressure = 0
                    cell.l_flags = cell.l_flags | self.LOCK
                    cell.r_flags = cell.r_flags | self.LOCK
                else:
                    left_resistive_force = len(left_pattern) * cell.l_solvent_permiability
                    right_resistive_force = len(right_pattern) * cell.r_solvent_permiability
                    pressure += left_resistive_force + right_resistive_force
                    left_neighbor_stride_equiv = (cells[i-1].stride + cell.stride - 1) // cell.stride if i > 0 else 0
                    right_neighbor_stride_equiv = (cells[i+1].stride + cell.stride - 1) // cell.stride if i < len(cells) - 1 else 0
                    if right_neighbor_stride_equiv < len(right_gaps) and right_neighbor_stride_equiv > 0:
                        cell.r_wall_flags = cell.r_wall_flags | self.ELASTIC
                        pressure -= len(right_gaps) / right_neighbor_stride_equiv
                    if left_neighbor_stride_equiv < len(left_gaps) and left_neighbor_stride_equiv > 0:
                        cell.l_wall_flags = cell.l_wall_flags | self.ELASTIC
                        pressure -= len(left_gaps) / left_neighbor_stride_equiv
                known_gaps = set(left_gaps) | set(right_gaps) | set(known_gaps)
                ##print(f"known gaps for cell {cell.label}: {known_gaps}")
                indices_to_zero = set()
                #if len(known_gaps) == 0:
                    ##print(f"Cell {cell.label} has no known gaps, skipping.")

                #for i, cluster in enumerate(left_pattern):
                #    cell.leftmost = i + cell.left
                #    if cluster[0] == 1:
                #        break

                #for i, cluster in enumerate(right_pattern):
                #    cell.rightmost = cell.right - i
                #    if cluster[0] == 1:
                #        break


                if False and left_resistive_force > self.FORCE_THRESH:
                    spoken_for_slice = { bit
                         for gap in left_gaps
                         for bit in range(gap, gap + cell.stride) }
                    fragmented_slice = set(range(left_flat_length)) - spoken_for_slice
                    window = extract_bit_region(raw, 0, left_flat_length)

                    compacted_strides, junk = self.contiguate(window, left_pattern, fragmented_slice, cell.stride)
                    print(f"Compacted strides for cell {cell.label}: {compacted_strides}, junk: {junk}")

                    #if junk:
                        ##print(f"Junk data found in cell {cell.label}: {junk}")
                    if cell.label not in self.input_queues:
                        self.input_queues[cell.label] = set()
                    print(f"Compacted strides for cell {cell.label}: {compacted_strides}")
                    
                    self.input_queues[cell.label].update(compacted_strides)
                    cell.injection_queue += len(compacted_strides)
                    indices_to_zero.update(fragmented_slice)

                if False and right_resistive_force > self.FORCE_THRESH:
                    
                    spoken_for_slice = { bit 
                         for gap in right_gaps
                         for bit in range(gap, gap + cell.stride) }
                    fragmented_slice = set(range(right_flat_length)) - spoken_for_slice
                    right_reverse = right_pattern[::-1]
                    compacted_strides, junk = self.contiguate(raw[-right_flat_length:], right_reverse, fragmented_slice, cell.stride)
                    
                    #if junk:
                        ##print(f"Junk data found in cell {cell.label}: {junk}")
                    if cell.label not in self.input_queues:
                        self.input_queues[cell.label] = set()
                    self.input_queues[cell.label].update(compacted_strides)
                    cell.injection_queue += len(compacted_strides)
                    indices_to_zero.update(fragmented_slice)

                
                raw = stamp(raw, indices_to_zero, 1, 0)
                print(f"Cell {cell.label} raw data after stamping: {raw.hex()}")
                pressure -= len(indices_to_zero) // cell.stride

                if len(known_gaps) > 0 and cell.injection_queue > 0:
                    if cell.label not in self.assignable_gaps:
                        self.assignable_gaps[cell.label] = set()
                    chosen_few = list(known_gaps)[:cell.injection_queue]
                    self.assignable_gaps[cell.label].update(chosen_few)
                    cell.injection_queue -= len(chosen_few)
                
                center_gaps = set()
                if center_chances > 0 and cell.injection_queue > 0:
                    
                    center_start_bit = left_flat_length
                    center_end_bit = (cell.right - cell.left) - right_flat_length
                    center_bit_length = center_end_bit - center_start_bit

                    center_alignment_offset = cell.left + left_flat_length

                    trimmed_byte_string = extract_bit_region(raw, center_start_bit, center_bit_length)
                    padding = len(trimmed_byte_string) * 8 - center_bit_length
                    #print(f"padding: {padding}")
                    #print(len(trimmed_byte_string)*8, center_bit_length, center_start_bit, cell.left, left_flat_length, cell.right, right_flat_length)
                    #trimmed_byte_string = raw[(left_flat_length + 8 - 1)//8:((cell.right-cell.left)-right_flat_length)//8]
                    #center_alignment_offset = cell.left + left_flat_length
                    ##print(f"byte string: {trimmed_byte_string.hex()}")
                    # since extract_bit_region(...) produced exactly `center_bit_length` bits:
                    #assert len(trimmed_byte_string)*8 == center_bit_length, f"Trimmed byte string length {len(trimmed_byte_string)*8} does not match expected center bit length {center_bit_length}"

                    
                    center_gaps = self.center_search(trimmed_byte_string, cell.stride, center_alignment_offset, padding=padding)
                    
                    new_chosen_few = []
                    assert len(center_gaps) >= 0, f"Cell {cell.label} center gaps {center_gaps} is negative, check stride and gap calculation"
                    if len(center_gaps) > 0:
                        print(f"Center gaps found in cell {cell.label}: {center_gaps}")
                        new_chosen_few = list(center_gaps)[:cell.injection_queue]
                        new_chosen_few_absolute = [gap + center_start_bit for gap in new_chosen_few]
                        ##print(f"New chosen few for cell {cell.label}: {new_chosen_few_absolute}")
                        if len(new_chosen_few) > 0 and cell.label not in self.assignable_gaps:
                            self.assignable_gaps[cell.label] = set()
                        self.assignable_gaps[cell.label].update(new_chosen_few_absolute)
                        cell.injection_queue -= len(new_chosen_few)
                    else:
                        pass
                        #assert False, f"Cell {cell.label} has no center gaps, this should not happen with the current algorithm"
                known_gaps = set(known_gaps) | set(center_gaps)
                pressure += cell.injection_queue
                #pressure *= cell.stride
                cell.pressure = pressure
                cell.salinity = len(self.input_queues[cell.label]) if cell.label in self.input_queues else 0
                system_pressure += pressure
                if cell.label in self.input_queues and len(self.input_queues[cell.label]) > 0 and cell.label in self.assignable_gaps and len(self.assignable_gaps[cell.label]) > 0:
                    print(f"Injecting data into cell {cell.label} with injection queue: {cell.injection_queue} and queue: {self.input_queues[cell.label]}")
                    print(f"self.input_queues: {self.input_queues}")
                    print(f"assignable gaps: {self.assignable_gaps}")
                    print(f"Cell {cell.label} assignable gaps: {self.assignable_gaps[cell.label]}")
                    print(f"data size: len(self.data): {len(self.data)}, self.data_byte_length(): {self.data_byte_length()}, self.data_bit_length:{self.data_bit_length}, self.mask_bit_length:{self.mask_bit_length}")
                    print(f"left pattern: {left_pattern}, right pattern: {right_pattern}")
                    relative_consumed_gaps, consumed_gaps, self.input_queues[cell.label] = self.injection(self.input_queues[cell.label], self.assignable_gaps[cell.label], cell.left)
                    cell.leftmost = min(cell.leftmost, cell.left + min(relative_consumed_gaps)) if relative_consumed_gaps else cell.leftmost
                    cell.rightmost = max(cell.rightmost, cell.left + max(relative_consumed_gaps)) if relative_consumed_gaps else cell.rightmost
                    #this reduction should already occur above
                    #cell.injection_queue -= len(consumed_gaps)
                    ##print(f"Cell {cell.label} processed with raw data: {raw.hex()}")
                    raw = stamp(raw, relative_consumed_gaps, cell.stride, 1)
                    ##print(f"Cell {cell.label} processed with raw data: {raw.hex()}")
                    #for consumed_gap in consumed_gaps:
                        ##print(f"known_gaps: {known_gaps}")
                        ##print(f"Cell {cell.label} consumed gap at {consumed_gap}")
                        #print(f"assignable_gaps: {self.assignable_gaps}")
                        # this was already removed by .pop in a pass by ref
                        #self.assignable_gaps[cell.label].remove(consumed_gap)
                    ##print(f"Cell {cell.label} processed with raw data: {raw.hex()}")
                
                assert cell.injection_queue == 0, f"Cell {cell.label} injection queue is not empty after processing: {cell.injection_queue}"
                raws[cell.label] = raw
                print(f"Cell {cell.label} processed with raw data: {raw.hex()}")

                byte_len = len(cell._buf)                    # same as (cell.len+7)//8
                #print(byte_len)
                
                
                #if cell.injection_queue > 0:
                    #print(f"Cell {cell.label} still has injection queue: {cell.injection_queue}")
                    #if self.assignable_gaps.get(cell.label):
                        #print(f"Cell {cell.label} has assignable gaps: {self.assignable_gaps[cell.label]}")
                        #assert False, "Cell has assignable gaps but injection queue is not empty"
            self.push_cell_mask(cell)
        self.system_pressure = system_pressure
        self.snap_cell_walls(cells)
        self.print_system(cells, data_buffer=self.data if self.data else None)
            #else:
                #print(f"Cell {cell.label} has no left/right distinction, skipping.")
            #print(f"after cell {cell.label}, data: {self.data.hex()}")
        
        return system_pressure, raws
        
    def snap_cell_walls(self, cells):
        """
        Determine the negotiated interstitial spacing between each adjacent pair of cells
        on integer lattice sites without modifying .left/.right. Each boundary snaps
        to its own stride grid; if no common meeting point exists, we select a prev/cur
        pair (a, b) that a ≤ b, minimizing (gap, pressure-shift cost).

        Results are stored as:
        prev.snapped_right = a
        curr.snapped_left  = b
        where gap = b - a is the minimal non-paradoxical spacing.
        """
        import math

        # Initialize fixed extents on first run
        for cell in cells:
            if not hasattr(cell, 'leftmost') or cell.leftmost is None:
                cell.leftmost = cell.left
            if not hasattr(cell, 'rightmost') or cell.rightmost is None:
                cell.rightmost = cell.right
        cell_has_data = {}
        cell_has_data[LEFT_WALL.label] = False
        cell_has_data[RIGHT_WALL.label] = False
        print(f"the left wall id is: {LEFT_WALL.label}, right wall id is: {RIGHT_WALL.label}")
        for cell in cells:
            cell_has_data[cell.label] = True
            # If leftmost == rightmost, it means the cell has no data
            if cell.leftmost == cell.rightmost:
                print(f"Cell {cell.label} has no data, leftmost == rightmost: {cell.leftmost} == {cell.rightmost}")
                print(f"Bitmask: {self.bitmask.hex()}")
                if extract_bit_region(self.bitmask, cell.leftmost, 1) == b'\x00':
                    cell_has_data[cell.label] = False


        #salinity_ratio = (cell.salinity for cell in cells if cell.salinity > 0)
        #salinity_ratio = salinity_ratio / sum(salinity_ratio)


        # Iterate over adjacent pairs
        for i in range(0, len(cells)+1):
            prev = cells[i-1] if i > 0 else LEFT_WALL
            curr = cells[i] if i < len(cells) else RIGHT_WALL
            
            if i == len(cells):
                RIGHT_WALL.leftmost = self.mask_bit_length
                RIGHT_WALL.rightmost = self.mask_bit_length
                RIGHT_WALL.left = self.mask_bit_length
                RIGHT_WALL.right = self.mask_bit_length
            #print(f"Snapping cell walls between {prev.label} and {curr.label}, starting with previous left:{prev.left}, right:{prev.right}, current left:{curr.left}, and right:{curr.right}")

            # Snap raw boundaries to their own grids
            prev_snap = (prev.right // prev.stride) * prev.stride
            curr_snap = ((curr.left + curr.stride - 1) // curr.stride) * curr.stride

            # Define non-paradoxical envelope [low, high]
            low = min(prev.rightmost, curr.leftmost)
            high = max(prev.rightmost, curr.leftmost)
            print(low, high, prev.rightmost, curr.leftmost, prev.label, curr.label)
            print(f"Cell has data: {cell_has_data[cell.label]}")
            
            #print(f"Envelope for snapping: [{low}, {high}]")
            left_back_wall = prev.left
            right_back_wall = curr.right

            
            # Generate all grid-aligned candidates within envelope
            # for prev boundary (multiples of prev.stride)
            a_candidates = []
            start = math.ceil(low / prev.stride) * prev.stride
            for a in range(start, high+1, prev.stride):
                a_candidates.append(a)

            # for curr boundary (multiples of curr.stride)
            b_candidates = []
            start = math.ceil(low / curr.stride) * curr.stride
            for b in range(start, high+1, curr.stride):
                b_candidates.append(b)

            #print(f"Candidates for prev boundary: {a_candidates}")
            #print(f"Candidates for curr boundary: {b_candidates}")

            # Fallback: if one list is empty, include nearest snaps clamped into envelope
            if not a_candidates:
                if i > 0:
                    #print(f"No candidates for prev boundary, clamping to envelope [{low}, {high}]")
                    a = min(max(prev_snap, low), high)
                    a = (a // prev.stride) * prev.stride
                    a_candidates = [a]
                else:
                    a_candidates = [LEFT_WALL.right]
            if not b_candidates:
                if i < len(cells):
                    #print(f"No candidates for curr boundary, clamping to envelope [{low}, {high}]")
                    b = min(max(curr_snap, low), high)
                    b = ((b + curr.stride - 1) // curr.stride) * curr.stride
                    b_candidates = [b]
                else:
                    b_candidates = [self.mask_bit_length]

            # Find best (a, b) with a ≤ b minimizing (gap, cost)
            best = None
            for a in a_candidates:
                for b in b_candidates:
                    if cell_has_data[curr.label] and (a > b or a < left_back_wall or b > right_back_wall or a > right_back_wall or b < left_back_wall):
                        #print(f"Invalid candidate: a={a}, b={b} left_back_wall={left_back_wall}, right_back_wall={right_back_wall}")
                        #print(self.data.hex())
                        #print(f"prev: {prev}, curr: {curr}")
                        continue
                    
                    gap = (b - a)
                    volume = curr.right - curr.left
                    assert not cell_has_data[curr.label] or ( cell_has_data[curr.label] and volume > 0 and curr.salinity > 0), f"Invalid volume for cell {curr.label}: {volume}, left: {curr.left}, right: {curr.right}, salinity: {curr.salinity}"
                    previous_pressure_per_volume = prev.pressure / (prev.right - prev.left) if (prev.right - prev.left) > 0 else 0
                    current_pressure_per_volume = curr.pressure / (curr.right - curr.left) if (curr.right - curr.left) > 0 else 0
                    a_pressure = previous_pressure_per_volume * (a - prev.left)
                    b_pressure = current_pressure_per_volume * (curr.right - b)
                    cost = abs(a - prev.right) * a_pressure - abs(b - curr.left) * b_pressure
                    candidate = (gap, cost, a, b)
                    #print(f"Evaluating candidate: {candidate}")
                    #print(f"Current best: {best}")
                    #print(f"Gap: {gap}, Cost: {cost}, a: {a}, b: {b}")
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                assert False, f"No valid candidates found for snapping between {prev.label} and {curr.label}. This should not happen."
                # no a ≤ b: force minimal overlap pair by allowing a>b but picking minimal |b-a|
                for a in a_candidates:
                    for b in b_candidates:
                        gap = abs(b - a)
                        cost = abs(a - prev.right) * prev.pressure + abs(b - curr.left) * curr.pressure
                        candidate = (gap, cost, a, b)
                        if best is None or candidate < best:
                            best = candidate
            # Unpack best
            _, _, a_best, b_best = best
            
            
            self.pull_cell_mask(prev)  # Ensure the cell's mask is up-to-date
            self.pull_cell_mask(curr)  # Ensure the cell's mask is up-to-date
            prev.right = a_best
            curr.left  = b_best
            prev.rightmost = a_best
            curr.leftmost  = b_best
            # Recompute proportional pressures based on new sub-lengths
            orig_a = prev.rightmost - prev.leftmost
            orig_b = curr.rightmost - curr.leftmost
            new_a  = a_best - prev.leftmost
            new_b  = curr.rightmost - b_best

            new_p_a = (prev.pressure * new_a) // orig_a if orig_a > 0 else 0
            new_p_b = (curr.pressure * new_b) // orig_b if orig_b > 0 else 0

            # Adjust system pressure and update cell pressures
            self.system_pressure += (new_p_a + new_p_b) - (prev.pressure + curr.pressure)
            prev.pressure = new_p_a
            curr.pressure = new_p_b


        # this is wrong, this is confusing cell index and data index
        # I'm so angry at this mental block it's like parkinsons its like a plaque it's fucking stupid
        # and it makes me fucking furious why I can't just change this like some asshole motherfuckcer
        # it holding a cotter pin refusing to let slip the reality that this just needs to be changed o the
        # right fucking equation which is simply and plainly fucking easy to evaluate it's just a ratio
        # the mask has bits to bytes bits per bit in the data AND IT'S NOT FUCKING EIGHT IT WILL NEVER BE EIGHT STOP ASSUMING IT'S EIGHT

        if cells[-1].right * MASK_BITS_TO_DATA_BITS > self.data_bit_length:
            #print(f"Expanding data buffer to accommodate last cell's right boundary: {cells[-1].right * MASK_BITS_TO_DATA_BITS} bits")
            self.expand(self.data_bit_length, (MASK_BITS_TO_DATA_BITS*cells[-1].right - self.data_bit_length), cells, warp=False)


        if self.system_pressure > 0:

            #print(f"System pressure after snapping cell walls: {self.system_pressure}")
            self.expand(self.data_bit_length, int(self.system_pressure * 8), cells, warp=False)
    def build_metadata(self, offset_bits, size_bits, cells):
        events = []
        # make sure these are lists
        offs = offset_bits if isinstance(offset_bits, (list,tuple)) else [offset_bits]
        szs  = size_bits   if isinstance(size_bits,   (list,tuple)) else [size_bits]

        for off, sz in zip(offs, szs):
            # 1) try to find a cell that contains `off`
            for cell in cells:
                if cell.left <= off < cell.right:
                    center = (cell.left + cell.right) // 2
                    events.append((center, sz))
                    break
            else:
                # 2) fallback ‑ split `sz` evenly (or proportionally) among all cells
                n = len(cells)
                base = sz // n
                rem  = sz % n
                for idx, cell in enumerate(cells):
                    share = base + (1 if idx < rem else 0)
                    center = (cell.left + cell.right) // 2
                    events.append((center, share))

        
        final = [(pos, share) for pos, share in events]
        return sorted(final, key=lambda e: e[0])
    
    
    def expand(self, offset_bits, size_bits, cells, warp=True):
        """
        Expand the data buffer by a given size at a specified offset.
        This is a placeholder for actual expansion logic.
        """
        #print(f"Expanding data buffer at offset {offset_bits} by {size_bits} bits, warp={warp}")
        # lets determine early if we are expanding inside a cell, where we know who gains allocation
        # and a system expansion where there is no specific target cell
        # because currently in that scenario we are dumping memory to nowhere
        # and hoping the simulation will naturally pull it together
        # but we have not demonstrated that to be the case
        events = self.build_metadata(offset_bits, size_bits, cells)

        # 3) now do the single‐loop copy+insert we sketched earlier
        old_bits   = self.data_bit_length
        new_bits   = old_bits + sum(sz for _, sz in events)
        new_bytes  = intceil(new_bits) // 8
        new_data   = bytearray(new_bytes)

        src_cursor = 0   # bit offset in old data
        dst_cursor = 0   # bit offset in new_data

        for insert_off, insert_sz in events:
            # copy [src_cursor, insert_off) from old -> new
            length = insert_off - src_cursor
            if length > 0:
                chunk = extract_bit_region(self.data, src_cursor, length)
                chunk = mask_padding_bits(chunk, length)
                write_bit_region(new_data, dst_cursor, chunk, length)
                src_cursor += length
                dst_cursor += length

            # now “insert” insert_sz zero‐bits by just advancing dst_cursor
            dst_cursor += insert_sz

        # 4) copy any remaining tail
        tail = old_bits - src_cursor
        if tail > 0:
            chunk = extract_bit_region(self.data, src_cursor, tail)
            chunk = mask_padding_bits(chunk, tail)
            write_bit_region(new_data, dst_cursor, chunk, tail)

        # 5) swap buffers & update counters
        self.data = new_data
        self._data_buf = (ctypes.c_ubyte * self.data_byte_length()).from_buffer(self.data)
        self.data_bit_length = new_bits
        new_mask_bits = new_bits // MASK_BITS_TO_DATA_BITS
        new_mask_bytes = intceil(new_mask_bits)

        old_mask = self.bitmask
        old_mask_bits = len(old_mask) * 8

        # build a fresh, zeroed mask
        new_mask = bytearray(new_mask_bytes)

        # copy old mask bits into the new one
        write_bit_region(new_mask, 0, old_mask, min(old_mask_bits, new_mask_bits))

        # swap in
        self.bitmask = new_mask
        self._bitmask_buf = (ctypes.c_ubyte * len(new_mask)).from_buffer(self.bitmask)
        self._bitmask_base = ctypes.addressof(self._bitmask_buf)
        self.mask_bit_length = new_mask_bits
        # 6) finally bump every cell’s left/right by the total bits
        #    that fell before its original left edge
        for cell in cells:
            shift = sum(sz for off, sz in events if off <= cell.left)
            cell.left  += shift
            cell.right += shift

    def actual_data_hook(self, src, dst_bits, length_bits, data_bits_to_map_bits=8):
        # 1) how many bits we’re writing
        total_bits    = length_bits * data_bits_to_map_bits

        # 2) grab the raw chunk
        byte_len      = (total_bits + 7) // 8
        assert byte_len <= self.data_byte_length(), f"Byte length {byte_len} exceeds data buffer length {self.data_byte_length()}"
        assert byte_len > 0, f"Byte length {byte_len} must be greater than 0"
        
        chunk         = ctypes.string_at(src, byte_len)
        chunk_int     = int.from_bytes(chunk, 'little') & ((1 << total_bits) - 1)

        # 3) compute bit‑offset into self.data
        start_bit     = dst_bits * data_bits_to_map_bits
        bit_offset    = start_bit % 8
        start_byte    = start_bit // 8

        # 4) figure out how many bytes we really need:
        #    we must cover (bit_offset + total_bits) bits
        # PAD_CLIP: Calculate region_len to cover (bit_offset + total_bits), including any extra bits due to clipping.
        region_len    = (bit_offset + total_bits + 7) // 8
        end_byte      = start_byte + region_len
        assert region_len == end_byte - start_byte, f"Region length {region_len} does not match end byte {end_byte} - start byte {start_byte}"

        #print(f"Writing {total_bits} bits to self.data at bit offset {start_bit}")
        #print(f"Chunk: {chunk.hex()}, Chunk Int: {chunk_int}, Start Bit: {start_bit}, Bit Offset: {bit_offset}, Start Byte: {start_byte}, End Byte: {end_byte}, Region Length: {region_len}")
        #print(f"Data before write: {self.data.hex()}")
        #print(f"Data length: {len(self.data)} bytes, Bit length: {self.data_bit_length} bits")
        #print(f"Src: {src}, Dst Bits: {dst_bits}, Length Bits: {length_bits}, Data Bits to Map Bits: {data_bits_to_map_bits}")

        write_bit_region(self.data, start_bit, extract_bit_region(src, 0, total_bits), total_bits)
        # 5) extract, merge, and write back
        #region_bytes  = self.data[start_byte:end_byte]
        #region_int    = int.from_bytes(region_bytes, 'little')

        #mask          = ((1 << total_bits) - 1) << bit_offset
        #region_int    = (region_int & ~mask) | (chunk_int << bit_offset)

        # 6) pack back into exactly region_len bytes
        #new_bytes     = region_int.to_bytes(region_len, 'little')

        #assert len(new_bytes) == region_len, f"New bytes length {len(new_bytes)} does not match region length {region_len}"

        #self.data[start_byte:end_byte] = new_bytes
        #ctypes.memmove(
        #    self._data_base + start_byte,
        #    new_bytes,
        #    len(new_bytes)
        #)


    # Dummy injection function placeholder
    def injection(self, data, known_gaps, left_offset=0):
        consumed_gaps = set()
        relative_consumed_gaps = set()
        data_copy = data.copy()
        for i, datum in enumerate(data_copy):
            
            if len(known_gaps) > 0:
                
                gap = known_gaps.pop()
                if gap >= self.data_bit_length:
                    print(f"Gap {gap} exceeds data bit length {self.data_bit_length}, skipping")
                    exit()
                relative_consumed_gaps.add(gap)
                gap = gap + left_offset
                consumed_gaps.add(gap)

                data.remove(datum)
                # Pass datum[1] (stride) directly as length to avoid negative subtraction
                #print(f"datum: {datum}, gap: {gap}, stride: {datum[1]}")
                print(f"Injecting data at gap {gap} with stride {datum[1]}")
                print(f"data size: self.data_bit_length: {self.data_bit_length} len(self.data) * 8: { len(self.data) * 8}, gap: {gap}, stride: {datum[1]}")
                #assert self.data_bit_length == len(self.data) * 8, f"Data bit length {self.data_bit_length} does not match data len length {len(self.data) * 8}"
                
                print(f"data in hex: {self.data.hex()}")
                print(gap*MASK_BITS_TO_DATA_BITS)
                print(datum)
                print(f" I think you can do this. It will be okay and you'll be happy with the result.")
                print(f"Injecting data at gap {gap} data length: {self.data_bit_length} gap * convertion: {gap * MASK_BITS_TO_DATA_BITS} stride: {datum[1]}")
                gap_data = extract_bit_region(self.data, gap * MASK_BITS_TO_DATA_BITS, datum[1])
                assert gap_data == b'\x00', f"Data {gap_data} at gap {gap} is not zero before injection"
                self.actual_data_hook(datum[0], gap, datum[1], MASK_BITS_TO_DATA_BITS)
            else:
                break
        return relative_consumed_gaps, consumed_gaps, data

    def step(self, cells):
        # Coordinate one simulation step
        sp, mask = self.minimize(cells)
        sim.evolution_tick(cells)
        return sp, mask

if __name__ == '__main__':
    STRIDE = 0
    previous_string = ''
    while STRIDE < 30:
        try:
            STRIDE += 1    
            import random
            CELL_COUNT = random.choice(range(1, 10))  # Randomly choose a cell count between 1 and 10
            TEST_SIZE_STRIDE_TIMES_UNITS = (STRIDE ** 2 * 8 * STRIDE * CELL_COUNT) // ( 8 * STRIDE * CELL_COUNT)
            assert TEST_SIZE_STRIDE_TIMES_UNITS % STRIDE == 0
            
            # Create a list of test cells with a chosen stride value (e.g., stride=2)
            cells = [Cell(stride=STRIDE, left=i * TEST_SIZE_STRIDE_TIMES_UNITS, len=TEST_SIZE_STRIDE_TIMES_UNITS, right=i*TEST_SIZE_STRIDE_TIMES_UNITS+TEST_SIZE_STRIDE_TIMES_UNITS) for i in range(CELL_COUNT)]
            sim = Simulator(cells)
            # keep all tmp buffers alive so pointers remain valid
            sim._tmp_bufs = []
            step = 0
            print("Testing simulation steps with increasing injection_queue values")
            import random
            #sim.snap_cell_walls(cells)
            while step < (2 ** STRIDE):
                for idx, cell in enumerate(cells):
                    cell.injection_queue += 1
                    sim.input_queues[cell.label] = set() if cell.label not in sim.input_queues else sim.input_queues[cell.label]
                    # 1) convert to immutable bytes so create_string_buffer will accept it
                    import os
                    byte_len = (cell.stride + 7) * MASK_BITS_TO_DATA_BITS          # == 2 for a 12-bit stride
                    random_data = os.urandom(byte_len)              # truly random bits, never all-zero
                    
                    # 2) create buffer and keep a reference
                    buf = ctypes.create_string_buffer(random_data)
                    sim._tmp_bufs.append(buf)
                    # 3) grab pointer + length tuple
                    ptr = ctypes.addressof(buf)
                    sim.input_queues[cell.label].add((ptr, cell.stride))  # Add some dummy data for injection
                sp, masks = sim.step(cells)
                
                mask_demo = list(masks.values())[0] if len(masks) > 0 else None
                # Print every byte in binary (8 bits, zero-padded)
                #check_string = previous_string
                #previous_string = ''.join(f'{b:08b}' for b in mask_demo)
                #if check_string != previous_string:
                    #print(previous_string)
                #print(sim.data.hex())

                ##print(f"Step {step+1}: system_pressure = {sp}")
                if all(all(b == 0xFF for b in mask) for label, mask in masks.items()):
                    # All bits set
                    break

                step += 1
                #for label, mask in masks.items():
                    #print(f"{label} Mask: {mask.hex()}")
                    
#                for i, cell in enumerate(cells):
#                    print(f"  Cell {i}: resize_queue = {cell.resize_queue}")
#                print("---")

        except AssertionError as e:
            print(f"AssertionError: {e}")
            ##print(f"Failed at STRIDE={STRIDE}, CELL_COUNT={CELL_COUNT}, TEST_SIZE_STRIDE_TIMES_UNITS={TEST_SIZE_STRIDE_TIMES_UNITS}")
            break