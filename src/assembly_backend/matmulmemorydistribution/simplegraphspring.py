# graph_viz.py
import pygame
import networkx as nx
import random, math
import graph_express2
from graph_express2 import ProcessGraph, sort_roles
from orbital import Orbit
from sympy import symbols

# parameters
WIDTH, HEIGHT = 800, 600
NODE_RADIUS = 8
SPRING_K = 0.1      # spring stiffness
REPULSION_K = .05  # repulsion constant
DAMPING = 0.75      # velocity damping
FPS = 160

class Node:
    def __init__(self, nid, pos):
        self.id = nid
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0,0)

class Edge:
    def __init__(self, u, v):
        self.u = u
        self.v = v
        # rest length = current distance
        self.rest_len = 100#u.pos.distance_to(v.pos)

def load_graph(dataG):
    # run spring layout once
    init_pos = nx.spring_layout(dataG, scale=1.0)
    # scale to screen
    for k,v in init_pos.items():
        init_pos[k] = (WIDTH*0.5 + v[0]*WIDTH*0.4,
                       HEIGHT*0.5 + v[1]*HEIGHT*0.4)
    nodes = { nid: Node(nid, init_pos[nid]) for nid in dataG.nodes }
    edges = [Edge(nodes[u], nodes[v]) for u,v in dataG.edges]
    return nodes, edges

def apply_forces(nodes, edges, active_set):
    # zero forces
    for n in nodes.values():
        n.force = pygame.Vector2(0,0)
    # spring forces with zero-length handling
    for e in edges:
        delta = e.v.pos - e.u.pos
        dist = delta.length()
        # guard against zero-length
        if dist < 1e-4:
            dist = 1e-4
            # random direction fallback
            direction = pygame.Vector2(random.random()-0.5, random.random()-0.5)
            if direction.length() < 1e-4:
                direction = pygame.Vector2(1, 0)
            else:
                direction = direction.normalize()
        else:
            direction = delta.normalize()
        # Hooke’s law
        rest_length = e.rest_len
        base = e.rest_len  
        alpha_target = 2 * base
        beta_target = 0.5 * base  # target rest length for inactive edges

        # α is your “rate” (0 < α < 1): larger α → faster initial growth
        α = 0.3  
        β = .8 #beta is 0.5 * base
        if (e.u.id, e.v.id) in active_set:
            rest_length -= (beta_target - rest_length) * β
        else:
            rest_length += (alpha_target - rest_length) * α
        fs = SPRING_K * (dist - rest_length)
        force = direction * fs
        # if this edge is active, amplify the force

        e.u.force += force
        e.v.force -= force
    # repulsive forces
    node_list = list(nodes.values())
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            a, b = node_list[i], node_list[j]
            delta = a.pos - b.pos
            d2 = delta.length_squared()
            # guard against zero-radius
            if d2 < 1e-4:
                d2 = 1e-4
                # random direction fallback
                direction = pygame.Vector2(random.random()-0.5, random.random()-0.5)
                if direction.length() < 1e-4:
                    direction = pygame.Vector2(1, 0)
                else:
                    direction = direction.normalize()
            else:
                direction = delta.normalize()
            fr = REPULSION_K / d2
            force = direction * fr
            a.force += force
            b.force -= force
    # integrate
    for n in nodes.values():
        n.vel = (n.vel + n.force)*DAMPING
        n.pos += n.vel
        # optional bounds checking
        n.pos.x = max(NODE_RADIUS, min(WIDTH-NODE_RADIUS, n.pos.x))
        n.pos.y = max(NODE_RADIUS, min(HEIGHT-NODE_RADIUS, n.pos.y))

def main():
    # Build ALAP schedule for orbital transfer EOM
    expr = Orbit.stable_orbit_transfer_solution(
        Orbit.symbolic_orbit('1'), Orbit.symbolic_orbit('2')
    )['equation_of_motion']
    pg = ProcessGraph(recombinatorics_level=5)
    pg.build_from_expression(lambda: expr)
    pg.compute_levels(method='alap')

    # Prepare dataflow DAG and grouping
    dataG = pg.dataG
    grouped = pg.group_edges_by_dataset(dataG)
    ordered_keys = sort_roles(grouped)
    for role, lvl, typ in ordered_keys:
        pg.run_at(lvl, typ, role)

    # highlight edge sets per stage
    highlight_sets = [set(grouped[lvl][typ][role])
                      for (role, lvl, typ) in ordered_keys]

    # Initialize Pygame animation
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    nodes, edges = load_graph(dataG)
    frame = 0
    running = True
    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

        # select active edge set for this frame
        active_set = highlight_sets[frame % len(highlight_sets)]
        apply_forces(nodes, edges, active_set)

        # draw background
        screen.fill((255, 255, 255))
        # draw edges
        for e in edges:
            col, w = (200, 200, 200), 1
            if (e.u.id, e.v.id) in active_set:
                col, w = (255, 0, 0), 3
            pygame.draw.line(screen, col, e.u.pos, e.v.pos, w)
        # draw nodes
        for n in nodes.values():
            color = (100, 100, 255) if any(
                (n.id == u and v in nodes) or (n.id == v and u in nodes)
                for u, v in active_set) else (170, 170, 170)
            pygame.draw.circle(screen, color,
                               (int(n.pos.x), int(n.pos.y)), NODE_RADIUS)

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    pygame.quit()

if __name__ == '__main__':
    main()