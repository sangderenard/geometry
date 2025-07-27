# salinepressure_3d.py – 3‑D hierarchical extension of salinepressure.SalineHydraulicSystem
# =====================================================================================
# This module generalises the original 1‑D bar simulation to a 3‑D, octree‑backed cell
# lattice.  The original behaviour is recovered exactly when the grid is flattened to
# (width, 1, 1) – i.e. a single row of unit voxels – so existing tests continue to
# pass without modification.
#
# CHANGELOG
# ---------
# 2025‑07‑27  ✱  Fix init‑order bug: `_leaf_nodes` now initialised **before** we call
#               the parent constructor, and `reset_state()` is made tolerant of the
#               pre‑tree call that originates inside `super().__init__()`.
#
# Key ideas
# ---------
# * **Unit voxels** (1×1×1) are the atomic leaves.  They inherit all salinity/pressure
#   logic from the parent class.
# * **Octree hierarchy** is built automatically from the requested `grid_shape`.
#   Internal nodes cache the sum of their children’s volumes, enabling multi‑scale
#   analysis or future physics couplings (e.g. pressure waves propagating through
#   adjacent regions).
# * All dynamic volume updates still occur on the *leaf list* held by the base
#   `SalineHydraulicSystem`; after every Euler step we propagate those volumes up
#   the tree so that parent nodes stay consistent.
# * The public API mirrors the 1‑D class.  Extra helpers (`slice_xy`, `slice_xz`,
#   etc.) provide quick 2‑D numpy views for visual/debug use, but are not needed by
#   the core engine.
#
# ------------------------------------------------------------------------------
# © 2025 – released under MIT licence like the rest of the project.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterable

import numpy as np
from sympy import Expr

# Re‑use the 1‑D engine (renamed here for clarity; same file as user upload)
from salinepressure import SalineHydraulicSystem

__all__ = [
    "TreeNode3D",
    "SalineHydraulicSystem3D",
]

################################################################################
# Octree data structure
################################################################################

@dataclass
class TreeNode3D:
    """Recursive octree node holding aggregate volume for a voxel block."""

    bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    children: Optional[List["TreeNode3D"]] = field(default=None)
    volume: float | int = 0
    leaf_index: Optional[int] = None

    # ------------------------------------------------------------------
    def is_leaf(self) -> bool:
        return self.children is None

    def propagate_volume_up(self) -> float:
        """Re‑compute and return this node’s aggregate volume."""
        if self.is_leaf():
            return self.volume
        self.volume = sum(child.propagate_volume_up() for child in self.children)
        return self.volume

    def __str__(self):
        (x0, x1), (y0, y1), (z0, z1) = self.bounds
        return (
            f"TreeNode3D[{x0}:{x1}, {y0}:{y1}, {z0}:{z1}] «vol={self.volume}» "
            f"{'leaf' if self.is_leaf() else f'children={len(self.children)}'}"
        )

################################################################################
# 3‑D hydraulic system
################################################################################

class SalineHydraulicSystem3D(SalineHydraulicSystem):
    """3‑D extension of :class:`salinepressure.SalineHydraulicSystem`."""

    # --------------------------- construction ---------------------------
    def __init__(
        self,
        s_exprs: List[Expr],
        p_exprs: List[Expr],
        grid_shape: Tuple[int, int, int],
        *,
        chars: Optional[Iterable[str]] = None,
        tau: float | int = 1.0,
        math_type: str = "float",
        int_method: str = "adams",
        protect_under_one: bool = True,
        bump_under_one: bool = False,
        epsilon: float = 1e-6,
    ):
        # Prepare attributes *needed* by the parent ctor (reset_state is called
        # from inside super().__init__)
        self._leaf_nodes: List[TreeNode3D] = []  # will be filled post‑super
        self.grid_shape: Tuple[int, int, int] = grid_shape
        self._nx, self._ny, self._nz = grid_shape

        # Validate shape ↔ expr count
        nx, ny, nz = grid_shape
        leaf_count = nx * ny * nz
        if leaf_count != len(s_exprs):
            raise ValueError(
                f"len(s_exprs) ({len(s_exprs)}) must equal nx*ny*nz ({leaf_count})"
            )
        if leaf_count != len(p_exprs):
            raise ValueError(
                f"len(p_exprs) ({len(p_exprs)}) must equal nx*ny*nz ({leaf_count})"
            )

        # Call 1‑D engine (will invoke our override of reset_state *once*)
        super().__init__(
            s_exprs,
            p_exprs,
            width=leaf_count,
            chars=list(chars) if chars is not None else None,
            tau=tau,
            math_type=math_type,
            int_method=int_method,
            protect_under_one=protect_under_one,
            bump_under_one=bump_under_one,
            epsilon=epsilon,
        )

        # Build octree & wire leaves to flat indices
        self.root: TreeNode3D = self._build_octree(grid_shape)
        assert len(self._leaf_nodes) == leaf_count

        # Now that the tree exists, refresh volumes so parent & tree agree
        self._sync_tree_from_flat()

    # --------------------------- reset / sync helpers -------------------
    def _sync_tree_from_flat(self):
        """Push current `self.volumes` into leaf nodes and propagate upward."""
        for leaf in self._leaf_nodes:
            leaf.volume = self.volumes[leaf.leaf_index]
        self.root.propagate_volume_up()

    def reset_state(self):  # override tolerant of early call
        super().reset_state()
        # Early in constructor the octree hasn't been built yet – just bail.
        if not getattr(self, "_leaf_nodes", None):
            return
        self._sync_tree_from_flat()

    # --------------------------- tree builder ---------------------------
    def _build_octree(self, shape: Tuple[int, int, int], offset=(0, 0, 0)) -> TreeNode3D:
        nx, ny, nz = shape
        ox, oy, oz = offset
        if nx == ny == nz == 1:  # leaf voxel
            idx = len(self._leaf_nodes)
            node = TreeNode3D(
                bounds=((ox, ox + 1), (oy, oy + 1), (oz, oz + 1)),
                children=None,
                volume=self.volumes[idx],
                leaf_index=idx,
            )
            self._leaf_nodes.append(node)
            return node
        # split dims
        hx, hy, hz = (math.ceil(nx / 2), math.ceil(ny / 2), math.ceil(nz / 2))
        lx, ly, lz = (nx - hx, ny - hy, nz - hz)
        child_specs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
        for dz in (0, 1):
            cz = hz if dz == 0 else lz
            if cz == 0:
                continue
            for dy in (0, 1):
                cy = hy if dy == 0 else ly
                if cy == 0:
                    continue
                for dx in (0, 1):
                    cx = hx if dx == 0 else lx
                    if cx == 0:
                        continue
                    child_specs.append(
                        ((cx, cy, cz), (ox + dx * hx, oy + dy * hy, oz + dz * hz))
                    )
        children = [self._build_octree(s, o) for s, o in child_specs]
        node = TreeNode3D(
            bounds=((ox, ox + nx), (oy, oy + ny), (oz, oz + nz)),
            children=children,
        )
        node.propagate_volume_up()
        return node

    # --------------------------- dynamics -------------------------------
    def step(self, dt: float | int = 1.0):
        bar = super().step(dt)
        self._sync_tree_from_flat()
        return bar

    # --------------------------- 3‑D views ------------------------------
    def _volumes_3d(self) -> np.ndarray:
        return np.array(self.volumes, dtype=float).reshape(self.grid_shape)

    def slice_xy(self, z: int = 0) -> np.ndarray:
        if not (0 <= z < self._nz):
            raise ValueError("z index out of range")
        return self._volumes_3d()[:, :, z]

    def slice_xz(self, y: int = 0) -> np.ndarray:
        if not (0 <= y < self._ny):
            raise ValueError("y index out of range")
        return self._volumes_3d()[:, y, :]

    def slice_yz(self, x: int = 0) -> np.ndarray:
        if not (0 <= x < self._nx):
            raise ValueError("x index out of range")
        return self._volumes_3d()[x, :, :]

################################################################################
# Example usage (will run if executed directly) – demonstrates dimensional
# reduction: the 1‑D bar is recovered by choosing a shape of (width,1,1).
################################################################################

if __name__ == "__main__":
    from sympy import symbols, sin, pi
    import time

    t = symbols("t")
    shape = (4, 3, 2)
    nvox = math.prod(shape)
    s_exprs = [1 + sin(t + 2 * pi * k / nvox) for k in range(nvox)]
    p_exprs = [1 + sin(t + pi / 3 + 2 * pi * k / nvox) for k in range(nvox)]

    sim3d = SalineHydraulicSystem3D(
        s_exprs,
        p_exprs,
        grid_shape=shape,
        tau=5,
        math_type="int",
        int_method="adams",
        protect_under_one=True,
        bump_under_one=True,
        chars=[chr(97 + (k % 26)) for k in range(nvox)],
    )

    print("Initial XY slice (z=0):")
    print(sim3d.slice_xy(0))

    dt = (2 * math.pi) / 20
    for _ in range(500):
        bar = sim3d.step(dt)
        print(bar)
        time.sleep(0.05)

    print("Final XZ slice (y=1):")
    print(sim3d.slice_xz(1))
