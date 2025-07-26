import ast
import inspect
import json
import os
import pygame
import subprocess

AST_JSON = 'ast_reference.json'
SSA_METADATA = 'ssa_helper_metadata.json'
TILE_SIZE = 100
WINDOW_SIZE = (800, 600)

def generate_ast_reference():
    ast_nodes = {}
    for name, cls in inspect.getmembers(ast, inspect.isclass):
        if issubclass(cls, ast.AST) and cls is not ast.AST:
            ast_nodes[name] = {
                'fields': cls._fields,
                'bases': [base.__name__ for base in cls.__bases__ if issubclass(base, ast.AST)],
            }
    with open(AST_JSON, 'w') as f:
        json.dump(ast_nodes, f, indent=2)

# Ensure AST reference exists
if not os.path.exists(AST_JSON):
    generate_ast_reference()

# Load AST and SSA metadata
with open(AST_JSON) as f:
    ast_nodes = json.load(f)

with open(SSA_METADATA) as f:
    ssa_metadata = json.load(f)

# Identify undefined nodes
undefined_nodes = [node for node in ast_nodes if node not in {meta.get('ast_node', '').split('.')[-1] for meta in ssa_metadata.values()}]

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('AST ↔ SSA Node Explorer')
font = pygame.font.SysFont('arial', 16)
clock = pygame.time.Clock()

selected_node = None

running = True
while running:
    screen.fill((30, 30, 30))

    # Display node tiles
    for idx, node_name in enumerate(undefined_nodes):
        x = (idx % 8) * (TILE_SIZE + 10) + 10
        y = (idx // 8) * (TILE_SIZE + 10) + 10
        rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, (70, 70, 200), rect)
        text = font.render(node_name, True, (255, 255, 255))
        screen.blit(text, (x + 5, y + 5))

        if rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(screen, (150, 150, 250), rect, 3)

        if selected_node == node_name:
            pygame.draw.rect(screen, (200, 50, 50), rect, 3)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for idx, node_name in enumerate(undefined_nodes):
                x = (idx % 8) * (TILE_SIZE + 10) + 10
                y = (idx // 8) * (TILE_SIZE + 10) + 10
                rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                if rect.collidepoint(mouse_pos):
                    selected_node = node_name

        elif event.type == pygame.KEYDOWN and selected_node:
            if event.key == pygame.K_RETURN:
                subprocess.Popen(['python', 'ssa.py', '-o', selected_node])

    # Show selected node details
    if selected_node:
        detail_rect = pygame.Rect(50, 400, 700, 180)
        pygame.draw.rect(screen, (50, 50, 50), detail_rect)
        pygame.draw.rect(screen, (200, 200, 200), detail_rect, 2)

        details = ast_nodes[selected_node]
        y_offset = 410
        screen.blit(font.render(f"Node: {selected_node}", True, (255, 255, 255)), (60, y_offset))
        screen.blit(font.render(f"Bases: {', '.join(details['bases'])}", True, (255, 255, 255)), (60, y_offset+20))
        screen.blit(font.render(f"Fields: {', '.join(details['fields'])}", True, (255, 255, 255)), (60, y_offset+40))
        screen.blit(font.render("Press Enter to define handler in ssa.py", True, (100, 255, 100)), (60, y_offset+100))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()