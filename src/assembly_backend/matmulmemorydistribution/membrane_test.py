"""membrane_demo.py
==================================
Standalone demo exercising the BindingMembrane class together with
its symbolic helpers via **graph_express2**.  Run:

    python membrane_demo.py

The script performs three independent smoke‑tests:

1. **Collider check** – verifies `check_bounds` on a couple of test
   particles.
2. **Symbolic graphs** – walks every entry in
   `binding_memranes_sympy.SYM_EXPR`, builds a dependency graph with
   *graph_express2*, schedules it (ALAP), and prints colourised band / op
   summaries through `GraphExpresss2Printer`.
3. **Surface evolution toy‑sim** – spawns a handful of mobile agents on
   the membrane surface, lets them wander until they bond to the nearest
   unoccupied mesh vertex, and prints bonding progress per step.

The last part previews the future step‑function you asked for – see the
`surface_step()` helper near the bottom. You can lift that verbatim into
`binding_membrane.py` as a **`BindingMembrane.step_surface_agents`**
method once you’re happy with the behaviour.
"""
from __future__ import annotations

import math
import random
from typing import Tuple

import torch
import sympy as sp

from binding_membrane import BindingMembrane
from binding_memranes_sympy import SYM_EXPR

from graph_express2 import ProcessGraph
from graph_express2printing import GraphExpresss2Printer

# ---------------------------------------------------------------------------
#  Utility: printable graph = ProcessGraph × printer mix‑in
# ---------------------------------------------------------------------------
class PrintableProcessGraph(GraphExpresss2Printer, ProcessGraph):
    """Thin helper so we can call the colourised print helpers directly."""
    def __init__(self, recomb_level: int = 0, *, expand_complex: bool = False):
        ProcessGraph.__init__(self, recomb_level, expand_complex)

# ---------------------------------------------------------------------------
#  1. Collider sanity‑check
# ---------------------------------------------------------------------------

def demo_membrane_instance() -> None:
    mem = BindingMembrane(centre=(0., 0., 0.), radius=50.0)
    print("\n=== Collider sanity‑check ===")

    pos   = torch.tensor([[60., 0., 0.],   # outside – should be clamped
                          [10., 0., 0.]])  # inside  – unchanged
    vel   = torch.tensor([[ 1., 0., 0.],
                          [ 0., 0., 0.]])
    force = torch.zeros_like(pos)

    p2, v2, f2 = mem.check_bounds(pos, vel, force)
    for i in range(len(pos)):
        state = "outside" if pos[i,0] > mem.radius else "inside"
        print(f"• particle {i}: {state} → new_pos = {p2[i].tolist()}")

# ---------------------------------------------------------------------------
#  2. Symbolic dependency‑graph demos
# ---------------------------------------------------------------------------

def demo_symbolic_graphs(recomb: int = 0) -> None:
    print("\n=== Symbolic graph demos ===")
    for name, expr in SYM_EXPR.items():
        print(f"\n--- {name} ---")
        pg = PrintableProcessGraph(recomb)
        # Build graph – fall back gracefully if helper not available.
        if hasattr(pg, "build_from_expression"):
            pg.build_from_expression(expr, 1)
        else:
            pg.build_graph(expr)
            pg.finalize_graph_with_outputs()
        pg.compute_levels(method="alap")
        pg.print_bands_and_ops()

# ---------------------------------------------------------------------------
#  3. Surface‑agent evolution toy‑sim
# ---------------------------------------------------------------------------

def surface_step(mem: BindingMembrane,
                 agents_pos: torch.Tensor,
                 occupancy: torch.Tensor,
                 bond_dist: float = 5.0,
                 drift: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single Euler‑step for agents living on the membrane surface.

    Parameters
    ----------
    mem : BindingMembrane
        The membrane providing `mesh_verts` and `radius`.
    agents_pos : (N,3) float32
        Current cartesian positions on the surface (norm ≈ radius).
    occupancy : (V,) bool
        Per‑vertex availability mask.
    bond_dist : float, optional
        Maximum distance at which an agent bonds permanently.
    drift : float, optional
        Tangential random walk step‑size when closest vertex is occupied.

    Returns
    -------
    new_pos : (N,3) float32
    bonded   : (N,) int64  –  −1 if un‑bonded else vertex index.
    """
    radius = mem.radius
    mesh   = mem.mesh_verts

    N = agents_pos.shape[0]
    bonded = torch.full((N,), -1, dtype=torch.long)

    for i in range(N):
        # Skip once bonded in previous iteration
        if occupancy.any() and (mesh == agents_pos[i]).all(dim=1).any():
            continue
        d = torch.linalg.norm(mesh - agents_pos[i], dim=1)
        v_idx = int(torch.argmin(d))

        if occupancy[v_idx]:
            # Random tangential drift
            tangent = torch.randn(3)
            tangent -= (tangent @ agents_pos[i]) / (radius**2) * agents_pos[i]
            tangent = tangent / torch.linalg.norm(tangent) * drift
            new_pos = agents_pos[i] + tangent
            agents_pos[i] = radius * new_pos / torch.linalg.norm(new_pos)
        else:
            if d[v_idx] < bond_dist:
                occupancy[v_idx] = True
                agents_pos[i]    = mesh[v_idx]  # snap exactly
                bonded[i]        = v_idx
            else:
                # Glide towards candidate vertex along great‑circle arc
                direction = (mesh[v_idx] - agents_pos[i])
                agents_pos[i] += 0.5 * direction / torch.linalg.norm(direction)
                agents_pos[i]  = radius * agents_pos[i] / torch.linalg.norm(agents_pos[i])

    return agents_pos, bonded


def demo_surface_evolution(steps: int = 10) -> None:
    print("\n=== Surface evolution toy‑sim ===")

    # Minimal octahedron mesh so every vertex is ‘authoritative’.
    base = torch.tensor([[ 1, 0, 0], [-1, 0, 0],
                         [ 0, 1, 0], [ 0,-1, 0],
                         [ 0, 0, 1], [ 0, 0,-1]], dtype=torch.float32)

    mem = BindingMembrane(centre=(0., 0., 0.), radius=50.0)
    mem.mesh_verts = base * mem.radius  # overwrite default empty mesh

    V = mem.mesh_verts.shape[0]
    occupancy = torch.zeros(V, dtype=torch.bool)

    # Spawn three agents at random surface positions.
    rng = torch.rand(3, 2)
    phi   = rng[:, 0] * 2 * math.pi
    costh = 2 * rng[:, 1] - 1.0
    sinth = torch.sqrt(1 - costh**2)
    agents_pos = mem.radius * torch.stack([sinth * torch.cos(phi),
                                           sinth * torch.sin(phi),
                                           costh], dim=1)

    for t in range(steps):
        agents_pos, bonded = surface_step(mem, agents_pos, occupancy)
        print(f"step {t:2d}: bonds → {bonded.tolist()}")

    print("\nFinal bonds:")
    for i, b in enumerate(bonded):
        if b >= 0:
            coord = mem.mesh_verts[b].tolist()
            print(f"  agent {i} bonded to vertex {b} at {coord}")
        else:
            print(f"  agent {i} still searching…")
# ---------------------------------------------------------------------------
#  3. Manhattan‑style diffusion on mesh graph
# ---------------------------------------------------------------------------

def build_adjacency(faces: torch.Tensor) -> Dict[int, List[int]]:
    """Return {vertex: [neighbour vertices]} from (F,3) index tensor."""
    adj: Dict[int, List[int]] = {}
    for tri in faces.tolist():
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    # Deduplicate neighbour lists
    for k, lst in adj.items():
        adj[k] = list(set(lst))
    return adj


def edge_prob_step(mem: BindingMembrane,
                   agents_idx: torch.Tensor,  # (N,) int – current vertex id
                   occupancy: torch.Tensor,   # (V,) float – crowd count
                   beta: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single discrete‑time diffusion step along mesh edges.

    *Transition probability*  p(i→j) ∝ exp( −β·occ[j] ).  Staying put is
    always an option with the same Boltzmann weighting.

    Returns
    -------
    new_idx : (N,) int
    bonded  : (N,) int  – vertex id where bonded, else −1
    """
    V = occupancy.numel()
    adj = mem._cached_adj   # set by caller
    N  = agents_idx.numel()
    bonded = torch.full((N,), -1, dtype=torch.long)

    for a in range(N):
        v_curr = int(agents_idx[a])
        neigh  = adj[v_curr] + [v_curr]   # include self‑stay option
        occ    = occupancy[neigh]
        w      = torch.exp(-beta * occ)
        probs  = w / w.sum()
        next_v = int(torch.multinomial(probs, 1).item())
        next_vid = neigh[next_v]

        # If vertex unoccupied→bond + increment crowd count
        if occupancy[next_vid] < 0.5:   # treat <0.5 as free
            occupancy[next_vid] += 1.0
            bonded[a] = next_vid
        agents_idx[a] = next_vid
    return agents_idx, bonded


def demo_diffusion(steps: int = 20) -> None:
    print("\n=== Manhattan diffusion demo ===")

    # Octahedron mesh (6 vertices, 8 faces)
    base = torch.tensor([[ 1, 0, 0], [-1, 0, 0],
                         [ 0, 1, 0], [ 0,-1, 0],
                         [ 0, 0, 1], [ 0, 0,-1]], dtype=torch.float32)
    faces = torch.tensor([[0,4,2],[2,4,1],[1,4,3],[3,4,0],
                          [0,2,5],[2,1,5],[1,3,5],[3,0,5]])

    mem = BindingMembrane(centre=(0.,0.,0.), radius=50.0)
    mem.mesh_verts = base * mem.radius
    mem.mesh_faces = faces
    mem._cached_adj = build_adjacency(mem.mesh_faces)

    V = mem.mesh_verts.shape[0]
    occupancy = torch.zeros(V)

    # Start three agents at random vertices
    agents_idx = torch.randint(0, V, (3,))

    for t in range(steps):
        agents_idx, bonded = edge_prob_step(mem, agents_idx, occupancy, beta=1.2)
        print(f"step {t:02d}: agents @ {agents_idx.tolist()}  bonds → {bonded.tolist()}")
        if (bonded >= 0).all():
            break

    print("\nFinal occupancy:")
    for v_id, occ in enumerate(occupancy.tolist()):
        print(f"  v{v_id}: {int(occ)} agent(s)")
# ---------------------------------------------------------------------------
#  Main entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    demo_membrane_instance()
    demo_symbolic_graphs(recomb=0)
    demo_surface_evolution(steps=12)
