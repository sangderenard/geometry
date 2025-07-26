import pygame
import networkx as nx
import graph_express2
from graph_express2 import ProcessGraph
from orbital import Orbit
from sympy import symbols
import math
import random
# PARAMETERS
WIDTH, HEIGHT = 800, 600
NODE_RADIUS = 8
FPS = 60

# Physics constants
SPRING_K = 0.1       # spring stiffness
REPULSION_K = 0.05   # repulsion constant
DAMPING = 0.98       # velocity damping

GLOW_RISE = 0.3    # rise rate
GLOW_DECAY = 0.05  # decay rate
GLOW_PEAK_ALPHA = 255
GLOW_FLOOR_ALPHA = 40
GLOW_PEAK_RADIUS = 1.5
GLOW_FLOOR_RADIUS = 0.7

# Easing/contraction rates
BETA_LEVEL = 0.1     # slow contraction at same level
BETA_TYPE  = 0.3     # medium contraction at same type
BETA_ROLE  = 0.6     # strong contraction at role
ALPHA_IDLE = 0.1     # idle relax back to base

# Contraction targets (fractions of base length)
LEVEL_TARGET_FACTOR = 0.9
TYPE_TARGET_FACTOR  = 0.75
ROLE_TARGET_FACTOR  = 0.5

# Edge styling
EDGE_BASE_COLOR  = (80, 80, 80)
EDGE_BASE_WIDTH  = 2
NODE_BASE_COLOR  = (200, 200, 200)
NODE_BASE_RADIUS = NODE_RADIUS

BACKGROUND_COLOR = (0, 0, 0)

# 3D projection constants and function
CAMERA_DISTANCE = 1000
FOCAL_LENGTH = 500

DEFAULT_EDGE_LENGTH = 500  # default edge length for new edges

def project_point(pos):
    # simple perspective projection
    factor = FOCAL_LENGTH / (FOCAL_LENGTH + pos.z + CAMERA_DISTANCE)
    x = pos.x * factor
    y = pos.y * factor
    return (int(x + WIDTH/2), int(y + HEIGHT/2))

class Node:
    def __init__(self, nid, pos):
        self.id = nid
        self.pos = pygame.Vector3(pos[0] - WIDTH/2, pos[1] - HEIGHT/2, random.uniform(-100, 100))
        self.vel = pygame.Vector3(0, 0, 0)
        self.force = pygame.Vector3(0, 0, 0)
        self.glow_alpha = 50
        self.glow_radius = 0.8

class Edge:
    def __init__(self, u, v):
        self.u        = u
        self.v        = v
        # store static base length
        self.base_len = DEFAULT_EDGE_LENGTH  # default length for new edges
        # dynamic rest length
        self.rest_len = self.base_len


def load_graph(dataG):
    init_pos = nx.spring_layout(dataG, scale=1.0)
    for k, v in init_pos.items():
        init_pos[k] = (
            WIDTH*0.5 + v[0]*WIDTH*0.4,
            HEIGHT*0.5 + v[1]*HEIGHT*0.4
        )
    nodes = {nid: Node(nid, init_pos[nid]) for nid in dataG.nodes}
    edges = [Edge(nodes[u], nodes[v]) for u, v in dataG.edges]
    return nodes, edges


def apply_forces(nodes, edges, lvl_set, typ_set, role_set):
    # reset forces
    for n in nodes.values():
        n.force = pygame.Vector3(0, 0, 0)

    # spring forces with stacked contraction
    for e in edges:
        key = (e.u.id, e.v.id)
        delta = e.v.pos - e.u.pos
        dist = delta.length()
        if dist == 0:
            dist = 1e-4
            dir_v = pygame.Vector3(0, 0, 0)
        else:
            dir_v = delta / dist

        # stacking rest_len update
        rl = e.rest_len
        base = e.base_len
        if key in lvl_set:
            target = LEVEL_TARGET_FACTOR * base
            rl += (target - rl) * BETA_LEVEL
        if key in typ_set:
            target = TYPE_TARGET_FACTOR * base
            rl += (target - rl) * BETA_TYPE
        if key in role_set:
            target = ROLE_TARGET_FACTOR * base
            rl += (target - rl) * BETA_ROLE
        if key not in lvl_set:
            # idle relax back to base
            target = base
            rl += (target - rl) * ALPHA_IDLE
        e.rest_len = rl

        # Hooke’s law
        fs = SPRING_K * (dist - rl)
        f  = dir_v * fs
        e.u.force += f
        e.v.force -= f

    # repulsive forces
    node_list = list(nodes.values())
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            a, b = node_list[i], node_list[j]
            delta = a.pos - b.pos
            d2 = delta.length_squared()
            if d2 == 0:
                continue
            length = math.sqrt(d2)
            dir_v = delta / length
            fr = REPULSION_K / d2
            f = dir_v * fr
            a.force += f
            b.force -= f

    EMERGENCY_RADIUS = 1000
    EMERGENCY_FORCE_K = 0.01  # gentle pull back

    for n in nodes.values():
        n.vel = (n.vel + n.force) * DAMPING
        n.pos += n.vel
        if n.pos.length() > EMERGENCY_RADIUS:
            to_center = -n.pos.normalize() * ((n.pos.length() - EMERGENCY_RADIUS) * EMERGENCY_FORCE_K)
            n.vel += to_center


def main():
    # build process graph
    expr = Orbit.stable_orbit_transfer_solution(
        Orbit.symbolic_orbit('1'), Orbit.symbolic_orbit('2')
    )['equation_of_motion']
    pg = ProcessGraph(recombinatorics_level=5)
    pg.build_from_expression(lambda: expr)
    pg.compute_levels(method='alap')

    dataG   = pg.dataG
    grouped = pg.group_edges_by_dataset(dataG)

    # ordered triples (level, type, role)
    ordered_keys = []
    for lvl in sorted(grouped):
        for typ in sorted(grouped[lvl]):
            for role in sorted(grouped[lvl][typ]):
                ordered_keys.append((lvl, typ, role))

    # helper unions
    def edges_at_level(l):
        s = set()
        for typ in grouped[l]:
            for r in grouped[l][typ]: s |= set(grouped[l][typ][r])
        return s
    def edges_at_type(l, t):
        return set().union(*(grouped[l][t].values()))
    def edges_at_role(r):
        s = set()
        for lvl in grouped:
            for typ in grouped[lvl]:
                s |= set(grouped[lvl][typ].get(r, []))
        return s

    pygame.init()
    screen  = pygame.display.set_mode((WIDTH, HEIGHT))
    clock   = pygame.time.Clock()
    nodes, edges = load_graph(dataG)
    running = True
    frame   = 0

    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

        # current triple
        lvl, typ, role = ordered_keys[frame % len(ordered_keys)]
        lvl_set  = edges_at_level(lvl)
        typ_set  = edges_at_type(lvl, typ)
        role_set = set(grouped[lvl][typ].get(role, []))

        apply_forces(nodes, edges, lvl_set, typ_set, role_set)

        screen.fill(BACKGROUND_COLOR)
        # draw edges
        for e in edges:
            key = (e.u.id, e.v.id)
            r = 255 if key in role_set else 0
            g = 255 if key in typ_set else 0
            b = 255 if key in lvl_set else 0
            color = (r, g, b) if (r, g, b) != (0, 0, 0) else EDGE_BASE_COLOR
            width = max(1, int(EDGE_BASE_WIDTH * (e.rest_len / e.base_len)))
            # project 3D endpoints to 2D
            p1 = project_point(e.u.pos)
            p2 = project_point(e.v.pos)
            # draw edge outline then main line
            pygame.draw.line(screen, (0, 0, 0), p1, p2, width + 2)
            pygame.draw.line(screen, color, p1, p2, width)

        for n in nodes.values():
            active = any(
                (n.id == u or n.id == v)
                for (u, v) in lvl_set | typ_set | role_set
            )
            if active:
                n.glow_alpha += (GLOW_PEAK_ALPHA - n.glow_alpha) * GLOW_RISE
                n.glow_radius += (GLOW_PEAK_RADIUS - n.glow_radius) * GLOW_RISE
            else:
                n.glow_alpha += (GLOW_FLOOR_ALPHA - n.glow_alpha) * GLOW_DECAY
                n.glow_radius += (GLOW_FLOOR_RADIUS - n.glow_radius) * GLOW_DECAY

        # draw nodes
        for n in nodes.values():
            # highlight if incident
            if any((n.id == u and v in nodes) or (n.id == v and u in nodes)
                   for u, v in lvl_set | typ_set | role_set):
                nr = 255 if any(n.id in (u, v) for (u, v) in role_set) else 0
                ng = 255 if any(n.id in (u, v) for (u, v) in typ_set)  else 0
                nb = 255 if any(n.id in (u, v) for (u, v) in lvl_set)  else 0
                color = (nr, ng, nb, n.glow_alpha)
            else:
                color = (*NODE_BASE_COLOR, n.glow_alpha)
            # project 3D position to 2D
            p = project_point(n.pos)
            # draw node outline then fill
            pygame.draw.circle(screen, (0, 0, 0), p, n.glow_radius + 2)
            pygame.draw.circle(screen, color, p, n.glow_radius)

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    pygame.quit()

if __name__ == '__main__':
    main()
