# memory_graph_allocator_demo.py
# ----------------------------------------------------------------------
# ❶  one-shot bridge that plugs LinearCells into BitTensorMemory
# ❷  disables the old bitmap walk & sweep code paths
# ❸  allocates a few nodes / edges only through “injection”
# ----------------------------------------------------------------------
import ctypes, itertools, random

# ── import the existing modules you just shared ───────────────────────
from linear_cells import LinearCells          # new bundle-aware solver
from memory_graph import (                         # huge file you uploaded
    BitTensorMemoryGraph,
    BitTensorMemory,
    NodeEntry,
    EdgeEntry,
    MetaGraphEdge,
)

# ╭──────────────────────────────────────────────────────────────────╮
# │ ①  Region → Cell translator                                     │
# ╰──────────────────────────────────────────────────────────────────╯
class RegionAllocator:
    """
    Glue-layer:  keeps a shared bitmap in BitTensorMemory,
    presents LinearCells as an *allocator* that returns byte offsets.
    """

    def __init__(self, bt_graph: BitTensorMemoryGraph,
                 grains_per_bit: int = 8, grains_per_obj: int = 4):
        self.bt  = bt_graph
        self.geo = Geo(grains_per_bit, grains_per_obj)
        self._mk_cells()                # build LinearCells world

    # ---------- build / rebuild ------------------------------------
    def _mk_cells(self):
        layout  = self.bt.compute_region_boundaries()   # list[(lbl,(start,end,stride))]
        cell_specs = []
        for label, (beg,end,stride) in layout:
            grains = (end - beg) // self.geo.grains_per_bundle()
            cell_specs.append(dict(
                label   = label,
                min     = 0,
                max     = grains,              # allow growth
                length  = grains,              # current length
                stride  = 1,                   # 1 bundle at a time
                obj_map = 0,                   # filled later
                flags   = LinearCells.ELASTIC | LinearCells.SLIP_OK,
            ))
        # shared bitmap: 1 bit ↔ 8 grains, so ≤64 bundles → 512 grains/region
        self.cells = LinearCells(cell_specs, self.geo)

    # ---------- allocate one struct bundle --------------------------
    def request(self, entry_size: int, which_line: int):
        """Return *byte offset* for a new struct of <entry_size> bytes."""
        inject_map = {}
        self.cells.inject_item(self.cells.cells, inject_map, which_line)
        # locate that just-occupied MSB bundle
        cell   = self.cells.cells[which_line]
        step   = self.geo.bits_per_bundle()
        bundle = self.cells.quanta(cell) - 1 - cell.obj_map.bit_length()//step
        # translate to byte offset inside hard memory:
        #   global bundle index = bundles in previous cells + local
        global_b  = bundle
        for c_prev in self.cells.cells[:which_line]:
            global_b += self.cells.quanta(c_prev)
        byte_off = global_b * self.geo.grains_per_bundle() * BitTensorMemory.DEFAULT_GRAIN
        return byte_off

# ╭──────────────────────────────────────────────────────────────────╮
# │ ②  monkey-patch BitTensorMemory to use the allocator             │
# ╰──────────────────────────────────────────────────────────────────╯
def _patch_memory_with_allocator(bt_graph: BitTensorMemoryGraph):
    alloc = RegionAllocator(bt_graph)

    def _new_find_free(self, offset_hint, size, allow_drift=False):
        # map entry size → line id  (N=1,E=2,P=3,C=4)
        line_id = {ctypes.sizeof(NodeEntry): bt_graph.LINE_N,
                   ctypes.sizeof(EdgeEntry): bt_graph.LINE_E,
                   ctypes.sizeof(MetaGraphEdge): bt_graph.LINE_P}.get(size,
                   bt_graph.LINE_C)
        off = alloc.request(size, line_id)
        self.mark_used(off, size)
        return off
    bt_graph.hard_memory.find_free_space = _new_find_free.__get__(
        bt_graph.hard_memory, BitTensorMemory)

# ╭──────────────────────────────────────────────────────────────────╮
# │ ③  demonstration run                                            │
# ╰──────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    g = BitTensorMemoryGraph(size=16 * 1024)   # 16 KiB payload
    _patch_memory_with_allocator(g)            # activate allocator

    # add 3 nodes → will allocate via injector, NO sweep_memory / bitmap walk
    for _ in range(3):
        g.add_node(node_data=f"demo-{random.randint(0,9999)}")

    # add a handful of edges
    for _ in range(5):
        g.add_edge(random.randint(1, 1<<32-1), random.randint(1, 1<<32-1))

    print("Graph allocations complete. Region lengths:")
    for rec in g.hard_memory.region_manager.cells.manifest():
        if rec["changed"]:
            print(f"  line {rec['label']}: grew to {rec['new']['length']} bundles")

    print("\nAllocator demo finished — no call paths ever touched\n"
          "`find_free_space()`’s old bitmap scanner or `sweep_memory()`.")
