import pygame
import networkx as nx
import graph_express2
from graph_express2 import ProcessGraph
from orbital import Orbit
from sympy import symbols

# PARAMETERS
WIDTH, HEIGHT = 800, 600
NODE_RADIUS = 8
FPS = 60

# Physics constants
SPRING_K = 0.1       # spring stiffness
REPULSION_K = 0.05   # repulsion constant
DAMPING = 0.75       # velocity damping
# easing rates for expansion/contraction
ALPHA_EXPAND = 0.3   # toward 2× base_len
BETA_CONTRACT = 0.5  # toward 0.5× base_len

# Edge styling
EDGE_BASE_COLOR = (80, 80, 80)
EDGE_BASE_WIDTH = 2
NODE_BASE_COLOR = (200,200,200)
NODE_BASE_RADIUS = NODE_RADIUS

BACKGROUND_COLOR = (0,0,0)

class Node:
    def __init__(self, nid, pos):
        self.id = nid
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0,0)
        self.force = pygame.Vector2(0,0)

class Edge:
    def __init__(self, u, v):
        self.u = u
        self.v = v
        # base length
        self.base_len = u.pos.distance_to(v.pos)
        # dynamic rest length
        self.rest_len = self.base_len


def load_graph(dataG):
    init_pos = nx.spring_layout(dataG, scale=1.0)
    # scale to window coords
    for k,v in init_pos.items():
        init_pos[k] = (
            WIDTH*0.5 + v[0]*WIDTH*0.4,
            HEIGHT*0.5 + v[1]*HEIGHT*0.4
        )
    nodes = {nid: Node(nid, init_pos[nid]) for nid in dataG.nodes}
    edges = [Edge(nodes[u], nodes[v]) for u,v in dataG.edges]
    return nodes, edges


def apply_forces(nodes, edges, active_union):
    # reset
    for n in nodes.values(): n.force = pygame.Vector2(0,0)

    # spring forces
    for e in edges:
        delta = e.v.pos - e.u.pos
        dist = delta.length() or 1e-4
        dir_v = delta.normalize()

        # rest_len easing
        if (e.u.id, e.v.id) in active_union:
            # contract
            target = 0.5 * e.base_len
            e.rest_len += (target - e.rest_len) * BETA_CONTRACT
        else:
            # expand
            target = 2.0 * e.base_len
            e.rest_len += (target - e.rest_len) * ALPHA_EXPAND

        fs = SPRING_K * (dist - e.rest_len)
        f = dir_v * fs
        e.u.force += f
        e.v.force -= f

    # repulsion
    node_list = list(nodes.values())
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            a, b = node_list[i], node_list[j]
            delta = a.pos - b.pos
            d2 = delta.length_squared() or 1e-4
            dir_v = delta.normalize()
            fr = REPULSION_K / d2
            f = dir_v * fr
            a.force += f
            b.force -= f

    # integrate
    for n in nodes.values():
        n.vel = (n.vel + n.force) * DAMPING
        n.pos += n.vel
        n.pos.x = max(NODE_BASE_RADIUS, min(WIDTH-NODE_BASE_RADIUS, n.pos.x))
        n.pos.y = max(NODE_BASE_RADIUS, min(HEIGHT-NODE_BASE_RADIUS, n.pos.y))


def main():
    # build process graph
    expr = Orbit.stable_orbit_transfer_solution(
        Orbit.symbolic_orbit('1'), Orbit.symbolic_orbit('2')
    )['equation_of_motion']
    pg = ProcessGraph(recombinatorics_level=5)
    pg.build_from_expression(lambda: expr)
    pg.compute_levels(method='alap')

    dataG = pg.dataG
    grouped = pg.group_edges_by_dataset(dataG)

    # build ordered list of triples (level, type, role)
    ordered_keys = []
    for lvl in sorted(grouped):
        for typ in sorted(grouped[lvl]):
            for role in sorted(grouped[lvl][typ]):
                ordered_keys.append((lvl, typ, role))

    # utility: collect unions
    def edges_at_level(l):
        s = set()
        for typ in grouped[l]:
            for r in grouped[l][typ]: s |= set(grouped[l][typ][r])
        return s
    def edges_at_type(l, t):
        s = set()
        for r in grouped[l][t]: s |= set(grouped[l][t][r])
        return s
    def edges_at_role(r):
        s = set()
        for lvl in grouped:
            for typ in grouped[lvl]:
                s |= set(grouped[lvl][typ].get(r, []))
        return s

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    nodes, edges = load_graph(dataG)
    running = True
    frame = 0

    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

        # determine current triple
        lvl, typ, role = ordered_keys[frame % len(ordered_keys)]
        lvl_set  = edges_at_level(lvl)
        typ_set  = edges_at_type(lvl, typ)
        role_set = edges_at_role(role)
        active_union = lvl_set | typ_set | role_set

        apply_forces(nodes, edges, active_union)

        screen.fill(BACKGROUND_COLOR)
        # draw edges with RGB highlights
        for e in edges:
            key = (e.u.id, e.v.id)
            r = 255 if key in role_set else 0
            g = 255 if key in typ_set else 0
            b = 255 if key in lvl_set else 0
            if (r,g,b) == (0,0,0):
                color = EDGE_BASE_COLOR
            else:
                color = (r, g, b)
            width = max(1, int(EDGE_BASE_WIDTH * (e.rest_len/e.base_len)))
            pygame.draw.line(screen, color, e.u.pos, e.v.pos, width)

        # draw nodes
        for n in nodes.values():
            # highlight node if in any active set
            if any((n.id == u and v in nodes) or (n.id == v and u in nodes)
                   for u, v in active_union):
                nr, ng, nb = 0,0,0
                # combine node color channels
                # red if participates in role
                if any(n.id in (u,v) for (u,v) in role_set): nr = 255
                # green for type
                if any(n.id in (u,v) for (u,v) in typ_set):  ng = 255
                # blue for level
                if any(n.id in (u,v) for (u,v) in lvl_set):  nb = 255
                color = (nr, ng, nb)
            else:
                color = NODE_BASE_COLOR
            pygame.draw.circle(screen, color, (int(n.pos.x), int(n.pos.y)), NODE_BASE_RADIUS)

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    pygame.quit()

if __name__ == '__main__':
    main()

