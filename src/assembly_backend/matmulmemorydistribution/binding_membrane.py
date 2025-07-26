import torch
# ---------------------------------------------------------------------------
#  MembraneGraph  –  scene-level container & vectorised dispatch
# ---------------------------------------------------------------------------
import networkx as nx
import torch
from typing import Iterable, Dict, Tuple

class BindingMembrane:
    """
    Multipurpose container that can evolve from a simple hard sphere into
    a full physically-based membrane.

    Public attributes (initialised to sane defaults or empty tensors)
    -----------------------------------------------------------------
    centre            (3,)  – current geometric centre
    radius            float – for the fallback spherical collider

    mesh_verts        (V,3) – surface vertices  (empty if unused)
    mesh_faces        (F,3) – index triples     (empty if unused)

    voxel_pressure    (*N,3) – scalar field inside the membrane
    voxel_species     dict[str, Tensor] – extra scalar fields (e.g. ions)

    face_permeability (F,)  – 0-1 coefficient per mesh face / DEC cell

    Notes
    -----
    • All tensors live on CPU by default; move them to CUDA if required.
    • Geometry helpers are written for *batch* inputs: `pos`, `vel`,
      `force` are (N,3) tensors.
    """

    # ---------------------------------------------------------------- init
    def __init__(self,
                 centre: torch.Tensor | tuple = (0., 0., 0.),
                 radius: float = 50.0):

        self.centre = torch.as_tensor(centre, dtype=torch.float32)    # (3,)
        self.radius = float(radius)                                   # scalar

        # --- future-heavy data, pre-allocated as empty --------------------
        self.mesh_verts        = torch.empty((0, 3))        # (V,3)
        self.mesh_faces        = torch.empty((0, 3), dtype=torch.long)
        self.face_permeability = torch.empty((0,))          # (F,)

        self.voxel_pressure    = torch.empty((0, 0, 0))     # 3-D grid
        self.voxel_species     = {}                         # name → Tensor

    # =============================================================== helpers
    def _distance_from_centre(self, pos: torch.Tensor) -> torch.Tensor:
        """Return (N,1) Euclidean distance to the *current* centre."""
        return torch.norm(pos - self.centre, dim=-1, keepdim=True)

    # ================================================================ public
    # --- full-trajectory gate ---------------------------------------------
    def check_bounds(self,
                     pos:   torch.Tensor,
                     vel:   torch.Tensor,
                     force: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Entry-point called each frame **before** positions are committed.

        You can pipe the data through more steps:

            pos, vel, force = self.apply_slip_boundary(... )
            pos, vel, force = self.respond_transgression(... )
            pos, vel, force = self.apply_osmosis(... )

        For now we delegate only to the conservative collider.
        """
        return self.respond_transgression(pos, vel, force)

    # --- collider (kept from the minimal version) -------------------------
    def respond_transgression(self,
                              pos:   torch.Tensor,
                              vel:   torch.Tensor,
                              force: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hard-sphere “freeze on outward motion”.

        TODO: upgrade to mesh-aware SDF or triangle tests when
        `self.mesh_verts` / `self.mesh_faces` are populated.
        """
        r_vec = pos - self.centre                    # (N,3)
        dist  = torch.norm(r_vec, dim=-1, keepdim=True)

        inside = dist <= self.radius
        if inside.all():
            return pos, vel, force                  # fast path

        # outward unit normal
        n = r_vec / (dist + 1e-9)

        v_out = (vel   * n).sum(-1, keepdim=True).clamp(min=0.)
        f_out = (force * n).sum(-1, keepdim=True).clamp(min=0.)

        vel   = vel   - n * v_out                   # cancel outward comp.
        force = force - n * f_out
        pos   = self.centre + n * self.radius       # clamp to surface

        return pos, vel, force

    # ===================================================== future extensions
    # ---------- geometry ---------------------------------------------------
    def inflate(self, dr: float):
        """Uniformly grow / shrink by *dr* ( >0 inflate, <0 deflate )."""
        self.radius = max(1e-6, self.radius + dr)          # keep >0
        # If you have a mesh: scale vertices *about* the centre
        if self.mesh_verts.numel():
            self.mesh_verts = (self.mesh_verts - self.centre) * \
                               ((self.radius) / (self.radius - dr)) + self.centre

    def build_convex_hull(self, points: torch.Tensor):
        """
        Given an (M,3) cloud, compute and store a convex hull mesh.

        Stub only – call out to `scipy.spatial.ConvexHull` or `trimesh`
        in a real implementation.
        """
        # TODO: fill in
        raise NotImplementedError

    # ---------- voxel fields ----------------------------------------------
    def set_voxel_pressure(self, grid: torch.Tensor):
        """Replace internal pressure field with *grid* (H,W,D)."""
        self.voxel_pressure = grid.clone()

    # ---------- permeability / osmosis ------------------------------------
    def compute_osmotic_flux(self, ext_concentration: torch.Tensor):
        """
        Placeholder: compute per-face flux given an outside concentration
        field.  Store the result or return it for the caller to apply.

        Returns
        -------
        flux : Tensor (F,) – positive = outward, negative = inward
        """
        # TODO: realistic implementation
        return torch.zeros_like(self.face_permeability)

    def apply_osmotic_exchange(self, dt: float,
                               ext_concentration: torch.Tensor | None = None):
        """
        Adjust internal voxel_species / pressure according to fluxes.

        At present this is a stub that does nothing.
        """
        if ext_concentration is None or not self.face_permeability.numel():
            return
        # TODO: integrate flux over dt and update self.voxel_species
        pass

    # ---------- slip-boundary (for lipid membranes) ------------------------
    def apply_slip_boundary(self,
                            pos:   torch.Tensor,
                            vel:   torch.Tensor,
                            mu:    float = 0.1):
        """
        Tangential slip: damp only the *normal* component of velocity by
        factor `mu` (0 = full slip, 1 = stick).

        Currently active only for the spherical fallback.
        """
        r_vec = pos - self.centre
        dist  = torch.norm(r_vec, dim=-1, keepdim=True)
        on_surface = torch.isclose(dist, torch.tensor(self.radius),
                                   atol=1e-3, rtol=0.)
        if not on_surface.any():
            return vel                                 # nothing to do

        n = r_vec / (dist + 1e-9)
        v_n = (vel * n).sum(-1, keepdim=True)
        vel = vel - mu * n * v_n                       # damp normal comp.
        return vel

class MembraneGraph:
    """
    Scene-wide manager holding **all** BindingMembrane instances and
    running their boundary logic in one vectorised sweep.

    Typical use
    -----------
    >>> memG = MembraneGraph()
    >>> mid = memG.add_membrane(BindingMembrane((0,0,0), 40.))
    >>> memG.add_membrane_child(parent=mid,
    ...       membrane=BindingMembrane((10,0,0), 8.0))
    >>> pos, vel, frc = memG.step(pos, vel, frc, dt)

    Internally
    ----------
    • Nodes:    graph node-id  → {"mem": BindingMembrane, "level": int}
    • Edges:    parent → child (`relation="contains"`) or lateral
    • All tensors are *not* copied – only views / in-place writes.
    """

    # ............................................................. lifecycle
    def __init__(self):
        self.G    : nx.DiGraph = nx.DiGraph()
        self._ids : Dict[int, BindingMembrane] = {}   # nid → object

    # ............................................................. adders
    def add_membrane(self,
                     membrane: BindingMembrane,
                     tag: str | None = None) -> int:
        nid = id(membrane)
        if nid in self.G:
            raise ValueError("Membrane already registered")
        self.G.add_node(nid,
                        mem   = membrane,
                        tag   = tag,
                        level = 0)
        self._ids[nid] = membrane
        return nid

    def add_membrane_child(self,
                           parent:   int,
                           membrane: BindingMembrane,
                           relation: str = "contains",
                           tag: str | None = None) -> int:
        child = self.add_membrane(membrane, tag)
        lvl   = self.G.nodes[parent]["level"] + 1
        self.G.nodes[child]["level"] = lvl
        self.G.add_edge(parent, child, relation=relation)
        return child

    # ............................................................. core step
    @torch.no_grad()
    def step(self,
             pos   : torch.Tensor,
             vel   : torch.Tensor,
             force : torch.Tensor,
             dt    : float = 1.0
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorised membrane processing.

        Parameters
        ----------
        pos, vel, force : (N,3) tensors – *shared* arrays for every particle.
        dt              : timestep, forwarded to future physics (osmosis, …).

        Returns the modified tensors (same objects, just for chaining).
        """

        # 1) Topological order – parents first (nesting matters)
        for nid in nx.topological_sort(self.G):
            mem = self.G.nodes[nid]["mem"]

            # slice-mask for particles this membrane owns/responsible for.
            # Today → all particles.  Tomorrow → use spatial hash / tags.
            mask = torch.ones(pos.shape[0], dtype=torch.bool,
                              device=pos.device)

            # batch-apply
            _p, _v, _f = mem.check_bounds(pos[mask], vel[mask], force[mask])

            # in-place write-back (views share storage – cheap)
            pos[mask]   = _p
            vel[mask]   = _v
            force[mask] = _f

            # ---- optional future hooks -----------------------------------
            # mem.apply_osmotic_exchange(dt, ext_conc)   # not yet
            # mem.apply_slip_boundary(...)

        return pos, vel, force

import math
import torch

# -----------------------------------------------------------
#  High-level actuator / façade for BindingMembrane
# -----------------------------------------------------------
class MembraneActions:
    """
    Thin wrapper that offers semantic operations for gameplay /
    simulation scripts.  Everything ultimately delegates back to
    the underlying BindingMembrane instance.

    • grow_surface(dA) / shrink_surface(dA) – change shell area
    • apply_internal_pressure(dp)           – uniform pressure bump
    • leak(species, dmol)                   – stub for ion exchange
    • slip(mu)                              – adjust slip coefficient
    """
    def __init__(self, membrane: BindingMembrane):
        self.m = membrane

    # --------------- geometry ------------------------------------------------
    def grow_surface(self, dA: float):
        """Increase surface area by *dA* (same sign-convention as inflate())."""
        if dA <= 0:
            raise ValueError("dA must be positive; use shrink_surface() instead.")
        self._area_delta(dA)

    def shrink_surface(self, dA: float):
        """Decrease surface area by *dA* until minimum radius is reached."""
        if dA <= 0:
            raise ValueError("dA must be positive.")
        self._area_delta(-dA)

    def _area_delta(self, dA: float):
        r  = self.m.radius
        dr = dA / (8.0 * math.pi * r)            # dA/dR = 8πR  →  dR = dA / 8πR
        self.m.inflate(dr)

    # --------------- pressure -----------------------------------------------
    def apply_internal_pressure(self, dp: float):
        """Uniformly add *dp* to voxel_pressure (creates a 1-cell grid if empty)."""
        if not self.m.voxel_pressure.numel():
            self.m.voxel_pressure = torch.zeros((1, 1, 1))
        self.m.voxel_pressure += dp

    # --------------- species exchange ----------------------------------------
    def leak(self, species: str, dmol: float):
        """Very crude placeholder: subtract / add *dmol* from a scalar store."""
        store = self.m.voxel_species.setdefault(species, torch.tensor(0.0))
        store += dmol
        self.m.voxel_species[species] = store

    # --------------- slip toggle --------------------------------------------
    def slip(self, mu: float):
        """Set global slip factor 0–1 (for now stored on the object)."""
        self.m.slip_mu = float(mu)


# -----------------------------------------------------------
#  Minimal text-based smoke test
# -----------------------------------------------------------
def random_particles(n, spread=60.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos   = (torch.rand((n, 3), generator=g) - 0.5) * spread
    vel   = torch.zeros_like(pos)
    force = torch.zeros_like(pos)
    return pos, vel, force


def main():
    # ------------------------------------------------------------------
    # 1) build a tiny hierarchy (unchanged)
    # ------------------------------------------------------------------
    outer = BindingMembrane((0, 0, 0), 40.0)
    inner = BindingMembrane((10, 0, 0), 8.0)

    world = MembraneGraph()
    outer_id = world.add_membrane(outer, tag="outer")
    world.add_membrane_child(outer_id, inner, tag="inner")

    # optional: seed internal state so fluxes have something to work with
    for m in (outer, inner):
        m.voxel_pressure = torch.full((1, 1, 1), 2.0)     # 2 Pa everywhere
        m.voxel_species["Na⁺"] = torch.full((1, 1, 1), 0.2)  # 0.2 mol L⁻¹

    # ------------------------------------------------------------------
    # 2) external bath conditions   (could be tensors of any shape)
    # ------------------------------------------------------------------
    bath_pressure = torch.tensor(1.0)      # scalar solvent pressure
    bath_conc     = {"Na⁺": torch.tensor(0.05)}   # 0.05 mol L⁻¹ outside

    # coupling coefficients  (tune freely or make tensors)
    k_p = 0.3   # pressure relaxation rate   [s⁻¹]
    k_c = 0.1   # concentration relaxation   [s⁻¹]

    # ------------------------------------------------------------------
    # 3) particles and loop
    # ------------------------------------------------------------------
    pos, vel, frc = random_particles(200, spread=120.0)
    n_steps, dt = 200, 0.05

    for step in range(n_steps):
        # --- membrane geometry interaction (vectorised)
        pos, vel, frc = world.step(pos, vel, frc, dt)

        # --- continuum exchange for every membrane --------------------
        for mem in world._ids.values():
            # ---------- pressure -------------------------------------
            intP = mem.voxel_pressure.mean() if mem.voxel_pressure.numel() else 0.
            dP   = intP - bath_pressure        # +ve ⇒ inside > outside
            mem.voxel_pressure += (-k_p * dP * dt)

            # ---------- every tracked species -------------------------
            for sp, ext_val in bath_conc.items():
                internal = mem.voxel_species.get(sp, torch.zeros(()))
                mean_int = internal.mean() if internal.numel() else internal
                dC       = mean_int - ext_val
                mem.voxel_species[sp] = internal + (-k_c * dC * dt)

        # (optional) diagnostics every few frames
        if step % 40 == 0:
            far = pos.norm(dim=1).max().item()
            print(f"t = {step*dt:6.2f}s | farthest particle r = {far:6.2f}"
                  f" | outer P = {outer.voxel_pressure.mean():5.2f}"
                  f" | outer [Na⁺] = {outer.voxel_species['Na⁺'].mean():5.3f}")

    # ------------------------------------------------------------------
    # 4) high-level actions demo (still valid)
    # ------------------------------------------------------------------
    acts = MembraneActions(outer)
    acts.grow_surface(100.0)
    acts.apply_internal_pressure(+5.0)
    acts.leak("Na⁺", -0.02)

    print("\n--- final outer stats ---")
    print("radius            :", outer.radius)
    print("mean pressure     :", outer.voxel_pressure.mean().item())
    print("mean [Na⁺]        :", outer.voxel_species['Na⁺'].mean().item())


if __name__ == "__main__":
    main()
