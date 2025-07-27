# pressure_planes_viewer.py – Visualise alpha‑blended planes cut through a
# SalineHydraulicSystem3D domain
# ============================================================================
# This utility wraps the SalineHydraulicSystem3D (see salinepressure_3d.py) and
# lets you declare a set of 2‑D planes, each with its own alpha value, then
# composites those slices into a single RGBA image for quick inspection in a
# 2‑D window.  The goal is to give you a lightweight way to *see* how the 3‑D
# volume fractions settle without firing up a full 3‑D renderer.
#
# Usage example (also runnable via `python pressure_planes_viewer.py`):
# ---------------------------------------------------------------------------
# * builds a 20×20×20 lattice with toy osmotic expressions;
# * defines three translucent planes: XY@z=10, XZ@y=15, YZ@x=5;
# * runs the solver for a few timesteps, updating the window every frame.
#
# Requires: matplotlib ≥ 3.8 for `imshow` with RGBA arrays.
# ----------------------------------------------------------------------------
# © 2025 — MIT licence, same as the parent project.

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sympy import Expr, symbols, sin, pi

from salinepressure_3d import SalineHydraulicSystem3D

__all__ = ["Plane", "PressurePlanesViewer"]

###############################################################################
# Plane specification
###############################################################################

@dataclass
class Plane:
    """A 2‑D slice through the 3‑D lattice.

    Parameters
    ----------
    axis : str
        One of "xy", "xz", "yz".
    index : int
        Voxel index along the orthogonal axis (0‑based).
    alpha : float, default 0.5
        Opacity when composited (0   = invisible; 1 = fully opaque).
    cmap : str or matplotlib Colormap, default "viridis"
        Colour map used to convert scalar volume values to RGB.
    """

    axis: str
    index: int
    alpha: float = 0.5
    cmap: str | colors.Colormap = "viridis"

###############################################################################
# Viewer / compositor
###############################################################################

class PressurePlanesViewer:
    """Alpha‑blend arbitrary lattice slices into one RGBA frame."""

    def __init__(self, system: SalineHydraulicSystem3D, planes: List[Plane]):
        self.sys = system
        self.planes = planes
        # Pre‑validate indices
        nx, ny, nz = self.sys.grid_shape
        for p in planes:
            if p.axis == "xy" and not (0 <= p.index < nz):
                raise ValueError(f"z index {p.index} out of range 0…{nz-1}")
            if p.axis == "xz" and not (0 <= p.index < ny):
                raise ValueError(f"y index {p.index} out of range 0…{ny-1}")
            if p.axis == "yz" and not (0 <= p.index < nx):
                raise ValueError(f"x index {p.index} out of range 0…{nx-1}")

    # ------------------------------------------------------------------
    def _slice(self, plane: Plane) -> np.ndarray:
        """Return a *scalar* 2‑D slice for the given plane."""
        if plane.axis == "xy":
            return self.sys.slice_xy(plane.index)
        if plane.axis == "xz":
            return self.sys.slice_xz(plane.index)
        if plane.axis == "yz":
            return self.sys.slice_yz(plane.index)
        raise ValueError("axis must be 'xy', 'xz', or 'yz'")

    # ------------------------------------------------------------------
    def composite_rgba(self) -> np.ndarray:
        """Return an (H, W, 4) RGBA image with values in [0,1]."""
        # Choose base canvas size from the first plane
        first = self.planes[0]
        sl = self._slice(first)
        h, w = sl.shape
        out = np.zeros((h, w, 4), dtype=float)  # RGBA
        # Composite in declaration order (Painter's algorithm)
        for pl in self.planes:
            data = self._slice(pl)
            norm = colors.Normalize(vmin=data.min(), vmax=data.max(), clip=True)
            rgb = plt.cm.get_cmap(pl.cmap)(norm(data))[:, :, :3]  # drop alpha
            a = pl.alpha
            # Alpha‑blend: out = src*a + dst*(1-a)
            out[..., :3] = rgb * a + out[..., :3] * (1 - a)
            out[..., 3] = 1 - (1 - a) * (1 - out[..., 3])  # cumulative alpha
        return out

    # ------------------------------------------------------------------
    def show(self, block: bool = False):
        """Pop up a matplotlib window with the current composite."""
        img = self.composite_rgba()
        plt.imshow(img)
        plt.axis("off")
        plt.pause(0.001)
        if block:
            plt.show()

###############################################################################
# Stand‑alone demo
###############################################################################

def _demo():
    t = symbols("t")
    shape = (20, 20, 20)
    nvox = math.prod(shape)

    # Toy osmotic law: target volume waves around each voxel index
    s_exprs: List[Expr] = [1 for _ in range(nvox)]  # not used in this demo
    p_exprs: List[Expr] = [1 + 0.2 * sin(t + 2 * pi * k / nvox) for k in range(nvox)]

    sim = SalineHydraulicSystem3D(
        s_exprs,
        p_exprs,
        grid_shape=shape,
        tau=3,
        math_type="float",
    )

    planes = [
        Plane("xy", 10, alpha=0.6, cmap="plasma"),
        Plane("xz", 15, alpha=0.4, cmap="viridis"),
        Plane("yz", 5, alpha=0.5, cmap="magma"),
    ]

    viewer = PressurePlanesViewer(sim, planes)

    plt.figure("SalinePressure3D – plane composite")
    dt = (2 * math.pi) / 30
    for _ in range(60):
        sim.step(dt)
        viewer.show(block=False)
        time.sleep(0.05)
    plt.show()

###############################################################################

if __name__ == "__main__":
    _demo()
