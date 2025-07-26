from binding_memranes_sympy import BindingMembrane, BoundarySubgraph, QueueSubgraph, EmbeddingManager
from binding_memranes_sympy import MembraneWalkerKernel
import torch
class MembranePortal:
    """
    Mediator between a free-flight spring graph and the membrane engine.
    • watches nodes that cross the membrane surface
    • keeps tensors *aliased* – no copies, no Python loops
    """
    def __init__(self, membrane: BindingMembrane,
                       verts: torch.Tensor, faces: torch.Tensor,
                       beta=1.2, gamma=0.5, kappa=0.0):
        self.boundary = BoundarySubgraph(verts, faces)
        self.queue    = QueueSubgraph(torch.empty((0,3)), torch.empty(0, dtype=torch.long))
        self.manager  = EmbeddingManager(
            self.boundary, self.queue,
            kernel=MembraneWalkerKernel(beta, gamma, kappa)
        )
        self.mem = membrane                # for hard-sphere fallback

    # ------------ API used by the viewer ------------------------------
    def feed(self, pos, vel, force):
        """
        • clamps ‘illegal’ outward motion via BindingMembrane.check_bounds
        • detects sign-flip (outside→inside or reverse)
          → pushes the node into self.queue if not already there
        """
        p, v, f = self.mem.check_bounds(pos, vel, force)   # conservative

        # sign test (‖x‖ ? radius) to harvest candidates -----
        inside_now  = (torch.norm(p, dim=1) <= self.mem.radius)
        was_inside  = getattr(self, "_inside_mask", torch.zeros_like(inside_now))
        crossed_in  = (~was_inside) & inside_now
        crossed_out = was_inside & (~inside_now)
        self._inside_mask = inside_now                     # persist

        new_ids  = crossed_in | crossed_out
        if new_ids.any():
            self._enqueue_nodes(p[new_ids], crossed_out[new_ids])

        # write back the (possibly clamped) tensors to caller
        pos.copy_(p); vel.copy_(v); force.copy_(f)

    def step(self, dt):
        if self.queue.pos.numel():          # nothing to do if queue empty
            self.manager.step(dt)           # walk / bond / split

    def flush_into(self, pos, vel, force):
        """
        Overwrite viewer tensors for nodes that are currently QUEUED or
        BONDED.  (Indices are cached in self._idx_buffer.)
        """
        if not hasattr(self, "_idx_buffer"):   # no queued nodes yet
            return
        idx = self._idx_buffer
        pos[idx]   = self.queue.pos          # positions frozen / slid
        vel[idx].zero_()                     # freeze dynamics inside slip
        force[idx].zero_()                   # forces handled by membrane

    # ------------ internal helpers ------------------------------------
    def _enqueue_nodes(self, positions, came_from_outside):
        """
        Append new agent rows into queue tensors (no realloc of old data).
        state = +1 if was outside → inside,  –1 for inside → outside
        """
        new_pos   = positions.detach()
        new_state = torch.where(came_from_outside,  # outside→in == +1
                                torch.ones(len(positions), dtype=torch.long),
                               -torch.ones(len(positions), dtype=torch.long))
        # concat once
        self.queue.pos   = torch.cat([self.queue.pos,   new_pos],   dim=0)
        self.queue.state = torch.cat([self.queue.state, new_state], dim=0)

        # remember which viewer rows these were (for flush_into)
        buf = getattr(self, "_idx_buffer", torch.empty(0, dtype=torch.long))
        self._idx_buffer = torch.cat([buf,
                                      torch.nonzero(came_from_outside |
                                                    (~came_from_outside)).flatten()])