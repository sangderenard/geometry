import ctypes
import string
from typing import Union
from sympy import Integer
from cell_consts import Cell, CellFlags, LeftWallFlags, RightWallFlags, SystemFlags, MASK_BITS_TO_DATA_BITS, TEST_SIZE_STRIDE_TIMES_UNITS, CELL_COUNT, STRIDE, RIGHT_WALL, LEFT_WALL
from salinepressure import SalineHydraulicSystem
from bitbitbuffer import BitBitBuffer
from bitstream_search import BitStreamSearch


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
        #for cell in self.cells:
        #    print(f"Simulator: Initializing cell {cell.label} with left={cell.left}, right={cell.right}, stride={cell.stride}")
        self.system_lcm   = self.lcm(cells)                       # ← uses your helper
        required_end = max(c.right for c in cells)           # highest bit any cell uses
        #print(f"Simulator: required end is {required_end} bits")
        mask_size    = BitBitBuffer._intceil(required_end, self.system_lcm)
        #print(f"Simulator: mask size is {mask_size} bits, system LCM is {self.system_lcm}")
        self.bitbuffer = BitBitBuffer(mask_size=mask_size,
                                    bitsforbits=MASK_BITS_TO_DATA_BITS)

        self.locked_data_regions = []

        self.search = BitStreamSearch()

        self.s_exprs = [Integer(0) for _ in range(CELL_COUNT)]
        self.p_exprs = [Integer(1) for _ in range(CELL_COUNT)]

        self.engine = None
        self.fractions = None

        self.run_saline_sim()

    def run_saline_sim(self):
        # 1) Instantiate engine with your per‐cell salinity & pressure expressions (or plain numbers)
        self.engine = SalineHydraulicSystem(
            self.s_exprs,           # e.g. [Integer(s0), Integer(s1), …]
            self.p_exprs,           # e.g. [Integer(p0), Integer(p1), …]
            width=self.bitbuffer.mask_size, # the total bit‐space you’re dividing
            chars=[chr(97+i) for i in range(CELL_COUNT)],
            tau=5, math_type='int',
            int_method='adams',
            protect_under_one=True,
            bump_under_one=True
        )
        for cell in self.cells:
            if cell.leftmost is None:
                cell.leftmost = cell.left
            if cell.rightmost is None:
                cell.rightmost = cell.right
        # 2) Ask for the equilibrium fractions at t=0
        self.fractions = self.engine.equilibrium_fracs(0.0)
        #for cell in self.cells:
            #if cell.salinity == 0:
                #cell.salinity = 1

        necessary_size = self.bitbuffer.intceil(sum(cell.salinity for cell in self.cells if hasattr(cell, 'salinity') and cell.salinity > 0), self.system_lcm)
        
        if self.bitbuffer.mask_size < necessary_size:
            offsets = [(cell.rightmost - cell.leftmost)//2+cell.leftmost for cell in self.cells if hasattr(cell, 'leftmost')]
            sizes = [(cell.salinity) for cell in self.cells if hasattr(cell, 'salinity') and cell.salinity > 0]
            size_and_offsets = sorted(list(zip(sizes, offsets)), reverse=True, key=lambda x: x[1])
            for size, offset in size_and_offsets:
                self.expand([offset], self.bitbuffer.intceil(size, self.lcm(self.cells)), self.cells, self.cells)

        self.snap_cell_walls(self.cells, self.cells)

    def get_cell_mask(self, cell: Cell) -> bytearray:
        return self.bitbuffer[cell.left:cell.right]

    def set_cell_mask(self, cell: Cell, mask: bytearray) -> None:
        self.bitbuffer[cell.left:cell.right] = mask

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

        class CellProposal(Cell):
            def __init__(self, cell):
                
                super().__init__(cell.stride, cell.left, cell.right, cell.len, leftmost=cell.leftmost, rightmost=cell.rightmost)

                self.salinity = cell.salinity
                self.pressure = cell.pressure
                self.leftmost = cell.leftmost
                self.rightmost = cell.rightmost

        proposals = []
        fractions = self.engine.equilibrium_fracs(0.0)
        total_space = self.bitbuffer.mask_size
        current_left = 0
        for cell, frac in zip(cells, fractions):
            new_width = max(self.bitbuffer.intceil(cell.salinity,cell.stride), self.bitbuffer.intceil(int(total_space * frac), cell.stride))
            assert new_width % cell.stride == 0, f"New width {new_width} for cell {cell.label} is not aligned with stride {cell.stride}"
            assert cell.stride > 0, f"Cell {cell.label} has non-positive stride {cell.stride}"
            proposal = CellProposal(cell)
            proposals.append(proposal)
            #cell.pressure = 0  # reset pressure after reallocation
            current_left = cell.right
            #print(f"Cell {cell.label} resized to {cell.left} - {cell.right} ({new_width} bits)")
        self.snap_cell_walls(cells, proposals)
        self.print_system(cells)
        return proposal

    def print_system(self, cells, width=80):
        """
        Draw the entire address space scaled to `width` characters,
        and report total size and fragmentation percentage.
        """
        total_bits = self.bitbuffer.mask_size
        if total_bits == 0:
            print("<empty>")
            return
        labels = string.ascii_lowercase

        bit_info = []
        for bit in range(total_bits):
            info = (None, False)
            for idx, cell in enumerate(cells):
                if cell.left <= bit < cell.right:
                    
                    mset = bool(int(self.bitbuffer[bit]))
                    
                    info = (idx, mset)
                    break
            bit_info.append(info)

        # fragmentation: only bits inside cells where mask==0
        free_bits = sum(1 for idx, m in bit_info if idx is not None and not m)
        free_regions = []
        run = 0
        for idx, m in bit_info:
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
        size_string = f"Total size: {total_bits} bits ({total_bits/8:.2f} bytes, mask bits: {self.bitbuffer.mask_size})"
        free_string = f"Free: {free_bits} bits; fragmentation: {frag_pct:.2f}%"

        # render map
        out = []
        for col in range(width):
            mid = min(int(((col + 0.5) * total_bits) / width), total_bits - 1)
            idx, m = bit_info[mid]
            if idx is None:
                out.append('.')
            else:
                c = labels[idx % len(labels)]
                if m:
                    c = c.upper()
                out.append(c)
        print(''.join(out), size_string, free_string)



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
        #print(f"Fragmented slice: {fragmented_slice}")
        #print(f"Pattern: {pattern}, stride: {stride}, raw length: {len(raw)}")
        for cluster in pattern:
            if cluster[0] == 1:
                is_junk = cluster[1] % stride
                if is_junk == 0:
                    #print(f"Contiguate: cluster={cluster}, stride={stride}, raw length={len(raw)}")
                    for j in range(cluster[1] // stride):
                        pointer_offset = i + j * stride

                        ##print(f"Contiguate: pointer_offset={pointer_offset}, stride={stride}, raw length={len(raw)}")
                        ##print(f"Pointer Offset to stride alignment: {pointer_offset % stride} == 0")
                        
                        mask_data    = self.bitbuffer[pointer_offset: pointer_offset + stride]
                        backing_data = self.bitbuffer._data_access[ pointer_offset: pointer_offset + stride ]

                        ##print(f"Mask data: {mask_data.hex()}, \nBacking data: {backing_data.hex()}")
                        
                        # Convert to immutable bytes to avoid TypeError
                        pointer = ctypes.addressof(ctypes.create_string_buffer(bytes(backing_data)))
                        #output.append((pointer, stride))
                        i += stride
                        
                else:
                    #print(f"Contiguate: cluster={cluster}, stride={stride}, raw length={len(raw)}")
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
            self.pull_cell_mask(cell)  # Ensure the cell's mask is up-to-date
            raw = self.bitbuffer[cell.left:cell.right]

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
            # Trust BitBitSlice’s own bookkeeping instead of recomputing
            # with an implicit 8‑bit alignment.
            self.padding = raw.padding            # bits of alignment filler
            assert self.padding >= 0, "negative padding is impossible"
            if cell.left != cell.right:
                # Replace existing calls to self.count_from with BitStreamSearch runs:
                left_pattern = BitStreamSearch.count_runs(raw)
                if left_pattern and left_pattern[0][0] == 0:
                    left_flat_length = left_pattern[0][1]
                else:
                    left_flat_length = 0
                left_gaps = BitStreamSearch.find_aligned_zero_runs(left_pattern, cell.stride)

                # For the right border, reverse the slice to count from the right:
                right_slice = raw[::-1]
                right_pattern = BitStreamSearch.count_runs(right_slice)
                if right_pattern and right_pattern[0][0] == 0:
                    right_flat_length = right_pattern[0][1]
                else:
                    right_flat_length = 0
                right_gaps = BitStreamSearch.find_aligned_zero_runs(right_pattern, cell.stride)

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
                center_gap = (cell.right - cell.left) - left_flat_length - right_flat_length
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
                    left_resistive_force = (len(left_pattern)-1) * cell.l_solvent_permiability
                    right_resistive_force = (len(right_pattern)-1) * cell.r_solvent_permiability
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
                    #print(f"Compacted strides for cell {cell.label}: {compacted_strides}, junk: {junk}")

                    #if junk:
                        ##print(f"Junk data found in cell {cell.label}: {junk}")
                    if cell.label not in self.input_queues:
                        self.input_queues[cell.label] = set()
                    #print(f"Compacted strides for cell {cell.label}: {compacted_strides}")
                    
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

                
                raw = self.bitbuffer.stamp(raw, indices_to_zero, 1, 0)
                #print(f"Cell {cell.label} raw data after stamping: {raw.hex()}")
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

                    trimmed_byte_string = self.bitbuffer.extract_bit_region(raw, center_start_bit, center_bit_length)
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
                        #print(f"Center gaps found in cell {cell.label}: {center_gaps}")
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
                    #print(f"Injecting data into cell {cell.label} with injection queue: {cell.injection_queue} and queue: {self.input_queues[cell.label]}")
                    #print(f"self.input_queues: {self.input_queues}")
                    #print(f"assignable gaps: {self.assignable_gaps}")
                    #print(f"Cell {cell.label} assignable gaps: {self.assignable_gaps[cell.label]}")
                    #print(f"data size: self.bitbuffer.data_size: {self.bitbuffer.data_size}, self.bitbuffer.bittobyte(self.bitbuffer.data_size): {self.bitbuffer.bittobyte(self.bitbuffer.data_size)}, self.bitbuffer.data_size:{self.bitbuffer.data_size}, self.bitbuffer.mask_size:{self.bitbuffer.mask_size}")
                    #print(f"left pattern: {left_pattern}, right pattern: {right_pattern}")
                    relative_consumed_gaps, consumed_gaps, self.input_queues[cell.label] = self.injection(self.input_queues[cell.label], self.assignable_gaps[cell.label], cell.left)
                    cell.leftmost = min(cell.leftmost, cell.left + min(relative_consumed_gaps)) if relative_consumed_gaps else cell.leftmost
                    cell.rightmost = max(cell.rightmost, cell.left + max(relative_consumed_gaps)) if relative_consumed_gaps else cell.rightmost
                    #this reduction should already occur above
                    #cell.injection_queue -= len(consumed_gaps)
                    ##print(f"Cell {cell.label} processed with raw data: {raw.hex()}")
                    raw = self.bitbuffer.stamp(raw, relative_consumed_gaps, cell.stride, 1)
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
                #print(f"Cell {cell.label} processed with raw data: {raw.hex()}")

                byte_len = len(cell._buf)                    # same as (cell.len+7)//8
                #print(byte_len)
                
                
                #if cell.injection_queue > 0:
                    #print(f"Cell {cell.label} still has injection queue: {cell.injection_queue}")
                    #if self.assignable_gaps.get(cell.label):
                        #print(f"Cell {cell.label} has assignable gaps: {self.assignable_gaps[cell.label]}")
                        #assert False, "Cell has assignable gaps but injection queue is not empty"
            self.push_cell_mask(cell)
        self.system_pressure = system_pressure
        
        self.snap_cell_walls(cells, cells)
        self.print_system(cells)
            #else:
                #print(f"Cell {cell.label} has no left/right distinction, skipping.")
            #print(f"after cell {cell.label}, data: {self.data.hex()}")
        
        return system_pressure, raws
    def lcm(self, cells):
        """
        Calculate the least common multiple of all cell strides.
        This is used to ensure that all cells are aligned correctly.
        """
        from math import gcd
        from functools import reduce

        def lcm(a, b):
            return a * b // gcd(a, b)

        return reduce(lcm, (cell.stride for cell in cells if hasattr(cell, 'stride')), 1)
    def snap_cell_walls(self, cells, proposals):
        """
        Determines and applies new cell boundaries using a stable, two-pass approach.
        1. Calculation Pass: Determines all new boundaries and the total required buffer size.
        2. Execution Pass: Expands the buffer once (triggering the desired global distribution)
           and then applies the new boundaries to the cells.
        """
        import math

        # Initialize fixed extents if they don't exist
        for cell in cells:
            if not hasattr(cell, 'leftmost') or cell.leftmost is None:
                cell.leftmost = cell.left
            if not hasattr(cell, 'rightmost') or cell.rightmost is None:
                cell.rightmost = cell.right

        # Initialize fixed extents if they don't exist
        for proposal in proposals:
            if not hasattr(proposal, 'leftmost') or proposal.leftmost is None:
                proposal.leftmost = proposal.left
            if not hasattr(proposal, 'rightmost') or proposal.rightmost is None:
                proposal.rightmost = proposal.right

        for c in [LEFT_WALL, RIGHT_WALL]:
            if getattr(c, "leftmost", None) is None:
                c.leftmost = c.left
            if getattr(c, "rightmost", None) is None:
                c.rightmost = c.right


        sorted_cells = sorted(cells, key=lambda c: c.leftmost)
        sorted_proposals = sorted(proposals, key=lambda p: p.leftmost)
        cells = [LEFT_WALL] + sorted_cells + [RIGHT_WALL]
        proposals = [LEFT_WALL] + sorted_proposals + [RIGHT_WALL]



        # --- Pass 1: Calculate all desired changes ---
        boundary_updates = []
        max_needed = self.bitbuffer.mask_size
        system_lcm = self.lcm(proposals)
        for i in range(len(proposals) + 1):
            prev = proposals[i - 1] if i > 0 else LEFT_WALL
            curr = proposals[i] if i < len(proposals) else RIGHT_WALL

            if i == len(proposals):
                # Ensure RIGHT_WALL is positioned at the current end
                RIGHT_WALL.leftmost = RIGHT_WALL.right = RIGHT_WALL.left = self.bitbuffer.mask_size
            
            # Define non-paradoxical envelope [low, high]
            low = min(prev.rightmost, curr.leftmost)
            high = max(prev.rightmost, curr.leftmost)

            # Generate all grid-aligned candidates within the envelope
            a_candidates = [a for a in range(math.ceil(low / prev.stride) * prev.stride, high + 1, prev.stride)]
            b_candidates = [b for b in range(math.ceil(low / curr.stride) * curr.stride, high + 1, curr.stride)]
            
            # Ensure candidate lists are not empty
            if not a_candidates:
                a_candidates = [(min(max((prev.right // prev.stride) * prev.stride, low), high) // prev.stride) * prev.stride] if i > 0 else [0]
            if not b_candidates:
                b_candidates = [((min(max(((curr.left + curr.stride - 1) // curr.stride) * curr.stride, low), high) + curr.stride - 1) // curr.stride) * curr.stride] if i < len(cells) else [self.bitbuffer.mask_size]

            # Find the best (a, b) pair with a <= b, minimizing gap and cost
            best = None
            for a in a_candidates:
                for b in b_candidates:
                    if a > b or a < prev.left + prev.stride * prev.salinity or b > curr.right - curr.stride * curr.salinity:
                        continue
                    
                    gap = b - a
                    # Simplified cost function for clarity
                    cost = abs(a - prev.right) + abs(b - curr.left)
                    candidate = (gap, cost, a, b)

                    if best is None or candidate < best:
                        best = candidate
            
            if best:
                _, _, a_best, b_best = best
                boundary_updates.append({'index': i, 'a': a_best, 'b': b_best})
                max_needed = max(max_needed, a_best, b_best)



        # --- Pass 2: Apply all calculated changes ---
        for update in boundary_updates:
            i = update['index']
            prev = proposals[i - 1] if i > 0 else LEFT_WALL
            curr = proposals[i] if i < len(proposals) else RIGHT_WALL

            # Modification 2: safety net for empty/ill-ordered cells
            prev.right = max(update['a'], prev.left)    # ensure width ≥ 0
            curr.left  = min(update['b'], curr.right)     # ensure valid boundary

            # Set the new boundaries from the stored values
            a_best = update['a']
            b_best = update['b']

            # Apply new boundaries and update pressures
            orig_a_len = prev.right - prev.left
            orig_b_len = curr.right - curr.left

            prev.right = a_best
            curr.left = b_best
            
            # Recompute proportional pressures based on new sub-lengths
            new_a_len = prev.right - prev.left
            new_b_len = curr.right - curr.left
            
            # Prevent division by zero
            
            new_p_a = (prev.pressure * new_a_len) // orig_a_len if orig_a_len > 0 else 0
            new_p_b = (curr.pressure * new_b_len) // orig_b_len if orig_b_len > 0 else 0
            
            self.system_pressure += (new_p_a + new_p_b) - (prev.pressure + curr.pressure)
            prev.pressure = new_p_a
            curr.pressure = new_p_b


        cells.pop()
        cells.pop(0)  # Remove LEFT_WALL and RIGHT_WALL from the cells list
        proposals.pop()
        proposals.pop(0)  # Remove LEFT_WALL and RIGHT_WALL from the proposals

        # --- Intermission: Perform a single, system-wide expansion if needed ---
        if max_needed > self.bitbuffer.mask_size:
            print(f"Had to expand bitbuffer mask size from {self.bitbuffer.mask_size} to {max_needed} bits for snapping cell walls")
            # This triggers the desired fallback logic in build_metadata to distribute the new space
            
            self.expand(self.bitbuffer.mask_size, self.bitbuffer.intceil(max_needed - self.bitbuffer.mask_size, system_lcm), cells, proposals)


        most_right = max(cell.rightmost for cell in proposals)
        if most_right > self.bitbuffer.mask_size:
            #print(f"Expanding data buffer to accommodate last cell's right boundary: {cells[-1].right * MASK_BITS_TO_DATA_BITS} bits")
            self.expand(self.bitbuffer.mask_size, self.bitbuffer.intceil(most_right - self.bitbuffer.mask_size, system_lcm), cells, proposals, warp=False)


        if self.system_pressure > 0:

            #print(f"System pressure after snapping cell walls: {self.system_pressure}")
            self.expand(self.bitbuffer.mask_size, self.bitbuffer.intceil(self.system_pressure, system_lcm), cells, proposals, warp=False)
    def build_metadata(self, offset_bits, size_bits, cells):
        
        events = []
        # make sure these are lists
        offs = offset_bits if isinstance(offset_bits, (list,tuple)) else [offset_bits]
        szs  = size_bits   if isinstance(size_bits,   (list,tuple)) else [size_bits]

        for offset in offs:
            assert isinstance(offset, int), f"Offset {offset} is not an integer"
            assert offset >= 0, f"Offset {offset} is negative, must be non-negative"
            assert offset <= self.bitbuffer.mask_size, f"Offset {offset} exceeds mask size {self.bitbuffer.mask_size}"

        for off, sz in zip(offs, szs):
            # 1) try to find a cell that contains `off`
            for cell in cells:
                # Modification 4: auto-heal pathological “left > right”
                if cell.right < cell.left:
                    cell.right = cell.left
                if cell.left <= off < cell.right:
                    # Modification 1: robust centre selection (align to LCM within cell)
                    raw_mid = (cell.left + cell.right) // 2
                    aligned = raw_mid - (raw_mid % self.system_lcm)
                    center  = max(cell.left, min(aligned, cell.right - 1))
                    events.append((cell.label, center, sz))
                    break
            else:
                # 2) fallback – split `sz` among cells
                n = len(cells)
                base = sz // n
                rem  = sz % n
                for idx, cell in enumerate(cells):
                    share  = self.bitbuffer.intceil(base + (1 if idx < rem else 0), self.system_lcm)
                    raw_mid = (cell.left + cell.right) // 2
                    center  = raw_mid - (raw_mid % self.system_lcm)
                    center  = max(cell.left, min(center, max(cell.right - self.system_lcm, cell.left)))
                    events.append((cell.label, center, share))

        
        final = [(label, pos, share) for label, pos, share in events]
        return sorted(final, key=lambda e: e[1])
    
    
    def expand(self, offset_bits, size_bits, cells, proposals, warp=True):
        """
        Build the event list exactly as before, then hand it off to BitBitBuffer.
        """
        if isinstance(offset_bits, int):
            offset_bits = [offset_bits]
        for offset in offset_bits:
            assert isinstance(offset, int), f"Offset {offset} is not an integer"
            assert offset >= 0, f"Offset {offset} is negative, must be non-negative"
            assert offset <= self.bitbuffer.mask_size, f"Offset {offset} exceeds mask size {self.bitbuffer.mask_size}"
        events = self.build_metadata(offset_bits, size_bits, cells)
        self.bitbuffer.expand(events, cells, proposals)


    def actual_data_hook(self, src, dst_bits, length_bits, data_bits_to_map_bits=8):
        # 1) how many bits we’re writing
        total_bits = length_bits * data_bits_to_map_bits
        byte_len = (total_bits + 7) // 8
        assert byte_len <= len(self.bitbuffer.data), f"Byte length {byte_len} exceeds data buffer length {len(self.bitbuffer.data)}"
        # Replace write/extract calls with slice access
        value = bytearray(ctypes.string_at(src, byte_len))
        self.bitbuffer._data_access[dst_bits:dst_bits+length_bits] = value

    # Dummy injection function placeholder
    def injection(self, data, known_gaps, left_offset=0):
        consumed_gaps = set()
        relative_consumed_gaps = set()
        data_copy = data.copy()
        for i, datum in enumerate(data_copy):
            if len(known_gaps) > 0:
                gap = known_gaps.pop()
                if gap >= self.bitbuffer.data_size:
                    #print(f"Gap {gap} exceeds data bit length {self.bitbuffer.data_size}, skipping")
                    exit()
                relative_consumed_gaps.add(gap)
                gap += left_offset
                consumed_gaps.add(gap)
                data.remove(datum)
                #print(f"Injecting data at gap {gap} with stride {datum[1]}")
                #print(f"data size: len(self.bitbuffer.data): {len(self.bitbuffer.data)}, data_bit_length:{self.bitbuffer.data_size}, mask_bit_length:{self.bitbuffer.mask_size}")
                #print(f"data in hex: {self.bitbuffer.data.hex()}")
                gap_data = self.bitbuffer._data_access[gap: gap + datum[1]]
                
                # Replace actual_data_hook call using slice assignment
                self.actual_data_hook(datum[0], gap, datum[1], MASK_BITS_TO_DATA_BITS)
            else:
                break
        return relative_consumed_gaps, consumed_gaps, data

    def step(self, cells):
        # Coordinate one simulation step
        sp, mask = self.minimize(cells)
        self.evolution_tick(cells)
        return sp, mask

# ====== Begin new tests ======
import pytest
import random

# Test simulation with various stride values:
# convenient: 2,4,8, popular: 3,5, prime: 7
@pytest.mark.parametrize("stride", [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 29, 31, 64, 128, 256, 512, 1024])
def test_simulation_with_strides(stride):
    CELL_COUNT = random.choice(range(1, 10))
    # simplified test size calculation: value must be a multiple of stride
    TEST_SIZE = stride * 10
    # create test cells with correct left/right boundaries
    cells = [Cell(stride=stride,
                  left=i * TEST_SIZE,
                  len=TEST_SIZE,
                  right=i * TEST_SIZE + TEST_SIZE)
             for i in range(CELL_COUNT)]
    sim = Simulator(cells)
    # perform a few simulation steps
    for _ in range(3):
        sp, _ = sim.step(cells)
        # basic assertion: system pressure should be a number (could be zero or positive)
        assert isinstance(sp, (int, float))
    # confirm each cell's mask length is as expected
    for cell in cells:
        mask = sim.get_cell_mask(cell)
        
        assert len(mask) == (cell.right - cell.left)

if __name__ == '__main__':
    pytest.main([__file__])
# ====== End new tests ======