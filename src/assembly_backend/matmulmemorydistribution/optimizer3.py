#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import pygame
import matplotlib.pyplot as plt
import colorsys
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# ======================
# Matrix setup
# ======================
def label_matrix(prefix, shape, syllables):
    labels = np.empty(shape, dtype=object)
    idx = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            syll = syllables[idx % len(syllables)]
            labels[i, j] = f"{prefix}{syll}{i}{j}"
            idx += 1
    return labels

def build_flat_operations(A_labels, B_labels, C_labels, D_labels, M, N, K):
    flat = []
    for i in range(M):
        for j in range(N):
            flat.append(("+=C", (D_labels[i, j], None), [(C_labels[i, j], None)]))
            for k in range(K):
                flat.append(("+=A*B", (D_labels[i, j], None), [(A_labels[i, k], None), (B_labels[k, j], None)]))
    return flat

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TorchParametricSolver(nn.Module):
    def __init__(self, operand_label_matrices):
        super().__init__()
        self.operand_label_matrices = operand_label_matrices
        self.operand_names = [labels.flatten() for labels in operand_label_matrices]
        self.all_labels = np.concatenate(self.operand_names)
        self.num_elements = len(self.all_labels)
        #print("[DEBUG] TorchParametricSolver initialized with {} elements.".format(self.num_elements))
        #print("[DEBUG] All labels:", self.all_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        
        # Build operations (generic, based on last matrix as target)
        self.operations = []
        target_labels = operand_label_matrices[-1]
        for i in range(target_labels.shape[0]):
            for j in range(target_labels.shape[1]):
                self.operations.append(("STORE", (target_labels[i,j], None), 
                                        [(mat[i%mat.shape[0], j%mat.shape[1]], None) for mat in operand_label_matrices[:-1]]))

        # Memory logits & operation scores
        self.mem_logits = nn.Parameter(torch.randn(self.num_elements))
        self.op_params = nn.Parameter(torch.randn(len(self.operations)))

        # === Per-matrix convolutional layers ===
        self.per_matrix_convs = nn.ModuleList([
            nn.Conv2d(1, 8, kernel_size=3, padding=1) for _ in operand_label_matrices
        ])
        # Memory-space convolutional layer
        self.memory_space_conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.memory_space_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.memory_space_proj = nn.Linear(16 * 4 * 4, len(self.operations))
        self.memory_lane_width = 8  # Default, can be set externally

    def build_per_matrix_inputs(self, addr_map, addr_to_rank):
        """
        For each operand matrix, build a (1, H, W) tensor of memory rank order for each element.
        Returns a list of tensors, one per matrix.
        """
        device = self.mem_logits.device
        per_matrix_inputs = []
        for mat in self.operand_label_matrices:
            H, W = mat.shape
            inp = torch.zeros(1, H, W, device=device)
            for i in range(H):
                for j in range(W):
                    label = mat[i, j]
                    if label in addr_map:
                        addr_val = addr_map[label]
                        if hasattr(addr_val, 'item'):
                            inp[0, i, j] = float(addr_val.item())
                        else:
                            inp[0, i, j] = float(addr_val)
                    else:
                        inp[0, i, j] = 0.0
            per_matrix_inputs.append(inp)
        return per_matrix_inputs

    def sinkhorn_sort(self, scores, n_iter=20, temperature=0.1):
        n = scores.shape[0]
        scores = scores / temperature
        P = torch.exp(scores.unsqueeze(0) - scores.unsqueeze(1))
        for _ in range(n_iter):
            P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
            P = P / (P.sum(dim=0, keepdim=True) + 1e-8)
        return P

    def forward(self, use_sinkhorn=False, sinkhorn_temp=0.1):
        # 1. Get memory rank order for each element (from mem_logits)
        logits = self.mem_logits
        if use_sinkhorn:
            # Soft assignment: use Sinkhorn to get a doubly-stochastic matrix
            P_mem = self.sinkhorn_sort(logits, temperature=sinkhorn_temp)
            mem_order = torch.matmul(P_mem, torch.arange(P_mem.shape[0], device=P_mem.device, dtype=torch.float32))
            # Convert soft mem_order to hard ranks (double argsort)
            hard_ranks = torch.argsort(torch.argsort(mem_order))
            addr_map = {self.all_labels[i]: hard_ranks[i] for i in range(self.num_elements)}
            
        else:
            # Hard assignment: use argsort
            sort_idx = torch.argsort(logits)
            addr_map = {self.all_labels[i]: sort_idx[i] for i in range(self.num_elements)}
            # For hard assignment, diversity is maximal (all one-hot), so proximity penalty is zero
            proximity_penalty = torch.tensor(0.0, device=logits.device)
        unique_addrs_sorted = sorted(float(addr_map[self.all_labels[i]].item() if hasattr(addr_map[self.all_labels[i]], 'item') else addr_map[self.all_labels[i]]) for i in range(self.num_elements))
        addr_to_rank = {addr: idx for idx, addr in enumerate(unique_addrs_sorted)}

        # 2. Per-matrix convolutional features (input: memory rank order)
        per_matrix_inputs = self.build_per_matrix_inputs(addr_map, addr_to_rank)
        per_matrix_features = []
        for inp, conv in zip(per_matrix_inputs, self.per_matrix_convs):
            feat = F.relu(conv(inp.unsqueeze(0)))  # (1, 8, H, W)
            per_matrix_features.append(feat)

        # 3. Route per-matrix features into memory-space tensor
        lane_width = getattr(self, 'memory_lane_width', 8)
        n_tiles = self.num_elements
        n_rows = int(np.ceil(n_tiles / lane_width))
        memory_space = torch.zeros(1, 8, n_rows, lane_width, device=logits.device)
        for mat_idx, mat in enumerate(self.operand_label_matrices):
            H, W = mat.shape
            feat = per_matrix_features[mat_idx].squeeze(0)  # (8, H, W)
            for i in range(H):
                for j in range(W):
                    label = mat[i, j]
                    mem_rank = int(float(addr_map[label].item()) if hasattr(addr_map[label], 'item') else addr_map[label])
                    row = mem_rank // lane_width
                    col = mem_rank % lane_width
                    memory_space[0, :, row, col] = feat[:, i, j]

        # 4. Memory-space convolution
        mem_conv = F.relu(self.memory_space_conv(memory_space))  # (1, 16, n_rows, lane_width)
        mem_pooled = self.memory_space_pool(mem_conv)  # (1, 16, 4, 4)
        mem_flat = mem_pooled.view(1, -1)
        op_logits = self.memory_space_proj(mem_flat).squeeze(0)  # (num_operations,)

        # 5. Operation ordering
        if use_sinkhorn:
            P = self.sinkhorn_sort(op_logits, temperature=sinkhorn_temp)
            op_order = torch.matmul(P, torch.arange(P.shape[0], device=P.device, dtype=torch.float32))
            op_sort_idx = torch.argsort(op_order)
        else:
            op_sort_idx = torch.argsort(op_logits)

        # 6. Materialize operations in the computed order, with assigned addresses
        ordered_ops = [self.operations[i] for i in op_sort_idx.detach().cpu().numpy()]
        materialized_ops = [
            (op_type, (dst_label, addr_map[dst_label]),
             [(src_label, addr_map[src_label]) for src_label, _ in sources])
            for op_type, (dst_label, _), sources in ordered_ops
        ]

        # Penalties and regularization

        inter_pen, intra_pen = self.compute_penalties(materialized_ops)


        return inter_pen, intra_pen, materialized_ops, addr_map

    # === (Penalties unchanged from your code) ===
    def compute_penalties(self, operations,
                        intra_forward_exp=1.0, intra_backward_exp=2.0,
                        inter_forward_exp=1.0, inter_backward_exp=2.0,
                        bit_alignment=3):
        byte_stride = 2 ** bit_alignment
        inter_forward_hops, inter_backward_hops = [], []
        intra_forward_hops, intra_backward_hops = [], []
        prev_endpoint = None
        for op_type, (_, dst_addr), sources in operations:
            seq = [s[1] for s in sources] + [dst_addr]
            #print('Operation sequence addresses:', [float(x.item()) if hasattr(x, 'item') else float(x) for x in seq])
            if prev_endpoint is not None:
                hop = seq[0] - prev_endpoint
                # Ensure hop is float tensor
                hop = hop.float() if hasattr(hop, 'float') else torch.tensor(float(hop), device=self.mem_logits.device)
                inter_forward_hops.append(torch.relu(hop))
                inter_backward_hops.append(torch.relu(-hop))
            prev_endpoint = seq[-1]
            for i in range(len(seq)-1):
                a = seq[i]
                b = seq[i+1]
                hop = b - a
                # Ensure hop is float tensor
                hop = hop.float() if hasattr(hop, 'float') else torch.tensor(float(hop), device=self.mem_logits.device)
                intra_forward_hops.append(torch.relu(hop))
                intra_backward_hops.append(torch.relu(-hop))
        device = self.mem_logits.device
        # torch.stack does not accept require_grad; instead, rely on autograd tracking from input tensors
        if inter_forward_hops:
            inter_forward_hops = torch.stack(inter_forward_hops).requires_grad_(True)
        else:
            inter_forward_hops = torch.zeros(1, device=device, requires_grad=True)
        if inter_backward_hops:
            inter_backward_hops = torch.stack(inter_backward_hops).requires_grad_(True)
        else:
            inter_backward_hops = torch.zeros(1, device=device, requires_grad=True)
        if intra_forward_hops:
            intra_forward_hops = torch.stack(intra_forward_hops).requires_grad_(True)
        else:
            intra_forward_hops = torch.zeros(1, device=device, requires_grad=True)
        if intra_backward_hops:
            intra_backward_hops = torch.stack(intra_backward_hops).requires_grad_(True)
        else:
            intra_backward_hops = torch.zeros(1, device=device, requires_grad=True)
        return self._staged_penalty_function(inter_forward_hops, inter_backward_hops, inter_forward_exp, inter_backward_exp), \
               self._staged_penalty_function(intra_forward_hops, intra_backward_hops, intra_forward_exp, intra_backward_exp)

    def _staged_penalty_function(self, fwd_hops, bwd_hops, fwd_exp, bwd_exp):
        cache_line, l1_capacity, l2_capacity, l3_capacity = 2, 4, 8, 16
        penalties = []
        for hops, exp in [(fwd_hops, fwd_exp), (bwd_hops, bwd_exp)]:
            base = hops ** exp
            staged = torch.where(hops <= cache_line, base*1.0,
                      torch.where(hops <= l1_capacity, base*1.5,
                      torch.where(hops <= l2_capacity, base*2.0,
                      torch.where(hops <= l3_capacity, base*3.0, base*5.0))))
            penalties.append(torch.sum(staged))
        return sum(penalties)

# ======================
# HSV mapping helpers
# ======================
def select_value_from_config(key, mem_idx, mem_loc, temporal, source_idx, matrix_pos, memory_distance=None):
    if isinstance(key, float):
        return key
    elif key == 'memory_index':
        return mem_idx
    elif key == 'memory_location':
        return mem_loc
    elif key == 'temporal_order':
        return temporal
    elif key == 'matrix_index':
        return source_idx
    elif key == 'matrix_position':
        return matrix_pos
    elif key == 'memory_distance' and memory_distance is not None:
        return memory_distance
    return 0.5

def chunk_operations(operations, chunk_size):
    return [operations[i:i+chunk_size] for i in range(0, len(operations), chunk_size)]

# ======================
# Generate overlay
# ======================
def generate_overlay_image(
    addr_map, operations, logical_order, D_labels,
    hsv_config,
    border_hsv_config=None,
    shown_groups=('A','B','C','D'),
    separate_matrices=False,
    temporal_usage=None,
    max_time=1
):
    import pygame
    import math
    total_elements = len(logical_order)
    min_S, max_S = 0.2, 1.0
    min_V, max_V = 0.2, 1.0
    unique_addrs_sorted = sorted(set(addr_map[label].item() if hasattr(addr_map[label], 'item') else addr_map[label] for label in logical_order))
    addr_to_rank = {addr: idx for idx, addr in enumerate(unique_addrs_sorted)}
    ranks = np.array([addr_to_rank[addr_map[label].item() if hasattr(addr_map[label], 'item') else addr_map[label]] for label in logical_order])
    norm_addrs = ranks / (len(unique_addrs_sorted) - 1 + 1e-6)
    max_time = max(temporal_usage.values()) if temporal_usage else 1
    tile_xy = {}
    if separate_matrices:
        group_shapes = {'A':(D_labels.shape[0],D_labels.shape[1]), 'B':(D_labels.shape[1],D_labels.shape[0]), 'C':(D_labels.shape[0],D_labels.shape[1]), 'D':(D_labels.shape[0],D_labels.shape[1])}
        width = sum(group_shapes[g][1] for g in shown_groups)
        height = max(group_shapes[g][0] for g in shown_groups)
        group_offsets = {}
        x_cursor = 0
        for g in shown_groups:
            group_offsets[g] = x_cursor
            x_cursor += group_shapes[g][1]
        for i, label in enumerate(logical_order):
            g = label[0]
            if g not in shown_groups:
                continue
            flat_idx = i
            if g == 'A':
                mi, mj = divmod(flat_idx, D_labels.shape[1])
            elif g == 'B':
                mi, mj = divmod(flat_idx - len(D_labels.flatten()), D_labels.shape[0])
            elif g == 'C':
                mi, mj = divmod(flat_idx - len(D_labels.flatten()) - len(D_labels.flatten()), D_labels.shape[1])
            elif g == 'D':
                mi, mj = divmod(flat_idx - len(D_labels.flatten()) - len(D_labels.flatten()) - len(D_labels.flatten()), D_labels.shape[1])
            y = mi
            x = mj + group_offsets[g]
            tile_xy[label] = (x,y)
        scale = 20
        width = width
        height = height
    else:
        grid_size = int(np.ceil(np.sqrt(len(unique_addrs_sorted))))
        def addr_to_xy(addr):
            addr = addr.item() if hasattr(addr, 'item') else addr
            rank = addr_to_rank[addr]
            return divmod(rank, grid_size)
        for i, label in enumerate(logical_order):
            if not any(label.startswith(g) for g in shown_groups):
                continue
            x, y = addr_to_xy(addr_map[label])
            tile_xy[label] = (y, x)
        width = grid_size
        height = grid_size
        scale = 20
    surf = pygame.Surface((width*scale, height*scale))
    surf.fill((128,128,128))
    # --- Draw tiles ---
    for i, label in enumerate(logical_order):
        if label not in tile_xy:
            continue
        mem_idx = i / total_elements
        mem_loc = norm_addrs[i]
        temporal = temporal_usage.get(label, 0) / max_time
        if label.startswith('A'):
            source_idx = 0.0
        elif label.startswith('B'):
            source_idx = 0.25
        elif label.startswith('C'):
            source_idx = 0.5
        elif label.startswith('D'):
            source_idx = 0.75
        else:
            source_idx = 0.0
        matrix_pos = 0.5
        H = select_value_from_config(hsv_config['H'], mem_idx, mem_loc, temporal, source_idx, matrix_pos)
        S = min_S + select_value_from_config(hsv_config['S'], mem_idx, mem_loc, temporal, source_idx, matrix_pos) * (max_S - min_S)
        V = min_V + select_value_from_config(hsv_config['V'], mem_idx, mem_loc, temporal, source_idx, matrix_pos) * (max_V - min_V)
        rgb = tuple(int(255*x) for x in colorsys.hsv_to_rgb(H, S, V))
        x, y = tile_xy[label]
        # Border color
        Hb = select_value_from_config(border_hsv_config['H'], mem_idx, mem_loc, temporal, source_idx, matrix_pos)
        Sb = min_S + select_value_from_config(border_hsv_config['S'], mem_idx, mem_loc, temporal, source_idx, matrix_pos) * (max_S - min_S)
        Vb = min_V + select_value_from_config(border_hsv_config['V'], mem_idx, mem_loc, temporal, source_idx, matrix_pos) * (max_V - min_V)
        border_rgb = tuple(int(255*x) for x in colorsys.hsv_to_rgb(Hb, Sb, Vb))
        border_rect = pygame.Rect(x*scale, y*scale, scale, scale)
        pygame.draw.rect(surf, border_rgb, border_rect)
        border_ratio = 0.15
        inset = int(scale * border_ratio)
        inner_size = scale - 2*inset
        inner_rect = pygame.Rect(x*scale+inset, y*scale+inset, inner_size, inner_size)
        pygame.draw.rect(surf, rgb, inner_rect)
    # --- Draw lines ---
    line_color_config_start = hsv_config.get('line_start', {'H': 0.0, 'S': 1.0, 'V': 1.0})
    line_color_config_end = hsv_config.get('line_end', {'H': 0.6, 'S': 1.0, 'V': 1.0})
    n_segments = 16
    line_offset_radius = scale * 0.3
    def compute_offset(seed_tuple):
        seed = hash(seed_tuple) % 360
        angle = (seed / 360.0) * 2 * math.pi
        return math.cos(angle) * line_offset_radius, math.sin(angle) * line_offset_radius
    # --- Compute max possible memory distance for normalization ---
    max_mem_distance = len(addr_to_rank) - 1 if len(addr_to_rank) > 1 else 1
    #print("operations:", operations)
    for idx, (op_type, (dst_label, dst_addr), sources) in enumerate(operations):
        dst_x, dst_y = tile_xy.get(dst_label, (None,None))
        if dst_x is None: continue
        dst_off_x, dst_off_y = compute_offset((idx, dst_label))
        dst_px = dst_x*scale + scale//2 + dst_off_x
        dst_py = dst_y*scale + scale//2 + dst_off_y
        dst_addr_val = dst_addr.item() if hasattr(dst_addr, 'item') else dst_addr
        # Always use scalar for lookup
        dst_addr_val_scalar = dst_addr_val.item() if hasattr(dst_addr_val, 'item') else dst_addr_val
        dst_rank = addr_to_rank.get(dst_addr_val_scalar, 0)
        for src_label, src_addr in sources:
            src_x, src_y = tile_xy.get(src_label, (None,None))
            if src_x is None: continue
            src_off_x, src_off_y = compute_offset((idx, src_label))
            src_px = src_x*scale + scale//2 + src_off_x
            src_py = src_y*scale + scale//2 + src_off_y
            src_addr_val = src_addr.item() if hasattr(src_addr, 'item') else src_addr
            src_addr_val_scalar = src_addr_val.item() if hasattr(src_addr_val, 'item') else src_addr_val
            src_rank = addr_to_rank[src_addr_val_scalar]
            mem_distance = abs(dst_rank - src_rank)
            #print("Operation:", op_type, "from", src_label, "to", dst_label)
            #print("Memory distance:", mem_distance, "from", src_label, "to", dst_label)
            #print("max_mem_distance:", max_mem_distance)
            #print("Unique addresses:", len(set(addr_map.values())), "out of", len(addr_map))
            norm_mem_distance = mem_distance / max_mem_distance
            mem_idx_src = logical_order.index(src_label) / total_elements if src_label in logical_order else 0.0
            mem_loc_src = norm_addrs[logical_order.index(src_label)] if src_label in logical_order else 0.0
            temporal_src = temporal_usage.get(src_label, 0) / max_time if temporal_usage else 0.0
            source_idx_src = 0.0
            if src_label.startswith('A'):
                source_idx_src = 0.0
            elif src_label.startswith('B'):
                source_idx_src = 0.25
            elif src_label.startswith('C'):
                source_idx_src = 0.5
            elif src_label.startswith('D'):
                source_idx_src = 0.75
            matrix_pos_src = 0.5
            mem_idx_dst = logical_order.index(dst_label) / total_elements if dst_label in logical_order else 0.0
            mem_loc_dst = norm_addrs[logical_order.index(dst_label)] if dst_label in logical_order else 0.0
            temporal_dst = temporal_usage.get(dst_label, 0) / max_time if temporal_usage else 0.0
            source_idx_dst = 0.0
            if dst_label.startswith('A'):
                source_idx_dst = 0.0
            elif dst_label.startswith('B'):
                source_idx_dst = 0.25
            elif dst_label.startswith('C'):
                source_idx_dst = 0.5
            elif dst_label.startswith('D'):
                source_idx_dst = 0.75
            matrix_pos_dst = 0.5
            # --- Use config system for line color, including memory_distance ---
            H0 = select_value_from_config(line_color_config_start.get('H', 0.0), mem_idx_src, mem_loc_src, temporal_src, source_idx_src, matrix_pos_src, norm_mem_distance)
            S0 = select_value_from_config(line_color_config_start.get('S', 1.0), mem_idx_src, mem_loc_src, temporal_src, source_idx_src, matrix_pos_src, norm_mem_distance)
            V0 = select_value_from_config(line_color_config_start.get('V', 1.0), mem_idx_src, mem_loc_src, temporal_src, source_idx_src, matrix_pos_src, norm_mem_distance)
            H1 = select_value_from_config(line_color_config_end.get('H', 0.6), mem_idx_dst, mem_loc_dst, temporal_dst, source_idx_dst, matrix_pos_dst, norm_mem_distance)
            S1 = select_value_from_config(line_color_config_end.get('S', 1.0), mem_idx_dst, mem_loc_dst, temporal_dst, source_idx_dst, matrix_pos_dst, norm_mem_distance)
            V1 = select_value_from_config(line_color_config_end.get('V', 1.0), mem_idx_dst, mem_loc_dst, temporal_dst, source_idx_dst, matrix_pos_dst, norm_mem_distance)
            A0 = select_value_from_config(line_color_config_start.get('A', 1.0), mem_idx_src, mem_loc_src, temporal_src, source_idx_src, matrix_pos_src, norm_mem_distance)
            A1 = select_value_from_config(line_color_config_end.get('A', 1.0), mem_idx_dst, mem_loc_dst, temporal_dst, source_idx_dst, matrix_pos_dst, norm_mem_distance)
            for seg in range(n_segments):
                t0 = seg / n_segments
                t1 = (seg + 1) / n_segments
                x0 = src_px * (1-t0) + dst_px * t0
                y0 = src_py * (1-t0) + dst_py * t0
                x1 = src_px * (1-t1) + dst_px * t1
                y1 = src_py * (1-t1) + dst_py * t1
                dH = H1 - H0
                if abs(dH) > 0.5:
                    if dH > 0:
                        dH -= 1.0
                    else:
                        dH += 1.0
                H = (H0 + dH * (t0 + t1)/2) % 1.0
                S = S0 * (1-t0) + S1 * t0
                V = V0 * (1-t0) + V1 * t0
                rgb = tuple(int(255*x) for x in colorsys.hsv_to_rgb(H, S, V))
                pygame.draw.line(surf, rgb, (x0, y0), (x1, y1), 2)
    arr = pygame.surfarray.array3d(surf)
    arr = np.transpose(arr, (1,0,2))
    return arr

# ======================
# Pygame
# ======================
def run_pygame_gui(image_generator, display_mode='side_by_side'):
    """
    display_mode: 'side_by_side', 'current_only', 'best_only'
    """
    pygame.init()
    screen = None
    for images in image_generator():
        # images can be a tuple (img_current, img_best) or a single image
        if isinstance(images, tuple):
            img_current, img_best = images
        else:
            img_current = images
            img_best = None
        if display_mode == 'side_by_side' and img_best is not None:
            # Make sure heights match
            if img_current.shape[0] != img_best.shape[0]:
                min_h = min(img_current.shape[0], img_best.shape[0])
                img_current = img_current[:min_h,:,:]
                img_best = img_best[:min_h,:,:]
            image = np.concatenate((img_current, img_best), axis=1)
        elif display_mode == 'current_only' or img_best is None:
            image = img_current
        elif display_mode == 'best_only':
            image = img_best
        else:
            image = img_current
        h, w, _ = image.shape
        if screen is None or screen.get_width() != w or screen.get_height() != h:
            screen = pygame.display.set_mode((w,h))
        surf = pygame.surfarray.make_surface(np.transpose(image, (1,0,2)))
        screen.blit(surf, (0,0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


# ======================
# Batched Solver Definition (integrated)
# ======================
import torch.nn.functional as F
import numpy as np

class BatchedTorchParametricSolver(nn.Module):
    def __init__(self, operand_label_matrices):
        super().__init__()
        self.operand_label_matrices = operand_label_matrices
        self.operand_names = [labels.flatten() for labels in operand_label_matrices]
        self.all_labels = np.concatenate(self.operand_names)
        self.num_elements = len(self.all_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        target_labels = operand_label_matrices[-1]
        self.operations = []
        for i in range(target_labels.shape[0]):
            for j in range(target_labels.shape[1]):
                self.operations.append(("STORE", (target_labels[i, j], None),
                                        [(mat[i % mat.shape[0], j % mat.shape[1]], None) for mat in operand_label_matrices[:-1]]))
        self.per_matrix_convs = nn.ModuleList([
            nn.Conv2d(1, 8, kernel_size=3, padding=1) for _ in operand_label_matrices
        ])
        self.memory_space_conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.memory_space_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.memory_space_proj = nn.Linear(16 * 4 * 4, len(self.operations))
        self.memory_lane_width = 8

    def from_logits(self, mem_logits_batch, reinforce=True):
        batch_size = mem_logits_batch.shape[0]
        device = mem_logits_batch.device
        addr_maps_batch = []
        mat_ops_batch = []
        inter_pen_batch = []
        intra_pen_batch = []
        op_logprobs_batch = []
        mem_logprobs_batch = []
        for b in range(batch_size):
            # --- REINFORCE for memory assignment ---
            mem_logits = mem_logits_batch[b]
            gumbel_noise_mem = -torch.empty_like(mem_logits).exponential_().log()
            sampled_mem_order = torch.argsort(mem_logits + gumbel_noise_mem)
            # Log-probability of sampled memory permutation
            mem_logprob = 0.0
            mem_logits_work = mem_logits.clone()
            for i in range(len(sampled_mem_order)):
                idx = sampled_mem_order[i]
                mem_logprob = mem_logprob + torch.log_softmax(mem_logits_work, dim=0)[idx]
                mem_logits_work[idx] = float('-inf')
            addr_map = {self.all_labels[i]: sampled_mem_order[i] for i in range(self.num_elements)}
            unique_addrs_sorted = sorted(addr_map[self.all_labels[i]] for i in range(self.num_elements))
            addr_to_rank = {addr: idx for idx, addr in enumerate(unique_addrs_sorted)}
            per_matrix_inputs = self.build_per_matrix_inputs(addr_map, addr_to_rank, device)
            per_matrix_features = []
            for inp, conv in zip(per_matrix_inputs, self.per_matrix_convs):
                feat = F.relu(conv(inp.unsqueeze(0)))
                per_matrix_features.append(feat)
            lane_width = self.memory_lane_width
            n_tiles = self.num_elements
            n_rows = int(np.ceil(n_tiles / lane_width))
            memory_space = torch.zeros(1, 8, n_rows, lane_width, device=device)
            for mat_idx, mat in enumerate(self.operand_label_matrices):
                H, W = mat.shape
                feat = per_matrix_features[mat_idx].squeeze(0)
                for i in range(H):
                    for j in range(W):
                        label = mat[i, j]
                        mem_rank = addr_map[label]
                        row = mem_rank // lane_width
                        col = mem_rank % lane_width
                        memory_space[0, :, row, col] = feat[:, i, j]
            mem_conv = F.relu(self.memory_space_conv(memory_space.requires_grad_(True)))  # (1, 16, n_rows, lane_width)
            mem_pooled = self.memory_space_pool(mem_conv)
            mem_flat = mem_pooled.view(1, -1)
            op_logits = self.memory_space_proj(mem_flat).squeeze(0)
            # --- REINFORCE for operation ordering ---
            gumbel_noise_op = -torch.empty_like(op_logits).exponential_().log()
            sampled_op_order = torch.argsort(op_logits + gumbel_noise_op)
            op_logprob = 0.0
            op_logits_work = op_logits.clone()
            for i in range(len(sampled_op_order)):
                idx = sampled_op_order[i]
                op_logprob = op_logprob + torch.log_softmax(op_logits_work, dim=0)[idx]
                op_logits_work[idx] = float('-inf')
            op_sort_idx = sampled_op_order
            ordered_ops = [self.operations[i] for i in op_sort_idx.detach().cpu().numpy()]
            materialized_ops = [
                (op_type, (dst_label, addr_map[dst_label]),
                 [(src_label, addr_map[src_label]) for src_label, _ in sources])
                for op_type, (dst_label, _), sources in ordered_ops
            ]
            inter_pen, intra_pen = self.compute_penalties(materialized_ops)
            addr_maps_batch.append(addr_map)
            mat_ops_batch.append(materialized_ops)
            inter_pen_batch.append(inter_pen)
            intra_pen_batch.append(intra_pen)
            op_logprobs_batch.append(op_logprob)
            mem_logprobs_batch.append(mem_logprob)
        return (
            torch.stack(inter_pen_batch).requires_grad_(True),
            torch.stack(intra_pen_batch).requires_grad_(True),
            mat_ops_batch,
            addr_maps_batch,
            torch.stack(op_logprobs_batch),
            torch.stack(mem_logprobs_batch)
        )

    def build_per_matrix_inputs(self, addr_map, addr_to_rank, device):
        per_matrix_inputs = []
        for mat in self.operand_label_matrices:
            H, W = mat.shape
            inp = torch.zeros(1, H, W, device=device, dtype=torch.float32)
            for i in range(H):
                for j in range(W):
                    label = mat[i, j]
                    if label in addr_map:
                        addr_val = addr_map[label]
                        inp[0, i, j] = addr_val
            per_matrix_inputs.append(inp.requires_grad_(True))
        return per_matrix_inputs

    def compute_penalties(self, operations,
                        intra_forward_exp=1.0, intra_backward_exp=2.0,
                        inter_forward_exp=1.0, inter_backward_exp=2.0,
                        bit_alignment=3):
        byte_stride = 2 ** bit_alignment
        device = self.per_matrix_convs[0].weight.device
        # Instead of lists, build 1D tensors directly and concatenate as we go
        inter_forward_hops = None
        inter_backward_hops = None
        intra_forward_hops = None
        intra_backward_hops = None
        prev_endpoint = None
        for op_type, (_, dst_addr), sources in operations:
            seq = [s[1] for s in sources] + [dst_addr]
            if prev_endpoint is not None:
                hop = seq[0] - prev_endpoint
                hop_f = torch.relu(hop).unsqueeze(0)
                hop_b = torch.relu(-hop).unsqueeze(0)
                inter_forward_hops = hop_f if inter_forward_hops is None else torch.cat([inter_forward_hops, hop_f], dim=0)
                inter_backward_hops = hop_b if inter_backward_hops is None else torch.cat([inter_backward_hops, hop_b], dim=0)
            prev_endpoint = seq[-1]
            for i in range(len(seq) - 1):
                a = seq[i]
                b = seq[i + 1]
                hop = b - a
                hop_f = torch.relu(hop).unsqueeze(0)
                hop_b = torch.relu(-hop).unsqueeze(0)
                intra_forward_hops = hop_f if intra_forward_hops is None else torch.cat([intra_forward_hops, hop_f], dim=0)
                intra_backward_hops = hop_b if intra_backward_hops is None else torch.cat([intra_backward_hops, hop_b], dim=0)
        if inter_forward_hops is None:
            inter_forward_hops = torch.zeros(1, device=device, requires_grad=True)
        if inter_backward_hops is None:
            inter_backward_hops = torch.zeros(1, device=device, requires_grad=True)
        if intra_forward_hops is None:
            intra_forward_hops = torch.zeros(1, device=device, requires_grad=True)
        if intra_backward_hops is None:
            intra_backward_hops = torch.zeros(1, device=device, requires_grad=True)
        # Ensure requires_grad is set (should be by default if inputs require grad)

        return self._staged_penalty_function(inter_forward_hops, inter_backward_hops, inter_forward_exp, inter_backward_exp), \
               self._staged_penalty_function(intra_forward_hops, intra_backward_hops, intra_forward_exp, intra_backward_exp)

    def _staged_penalty_function(self, fwd_hops, bwd_hops, fwd_exp, bwd_exp):
        cache_line, l1_capacity, l2_capacity, l3_capacity = 2, 4, 8, 16
        penalties = []
        for hops, exp in [(fwd_hops, fwd_exp), (bwd_hops, bwd_exp)]:
            base = hops ** exp
            staged = torch.where(hops <= cache_line, base * 1.0,
                      torch.where(hops <= l1_capacity, base * 1.5,
                      torch.where(hops <= l2_capacity, base * 2.0,
                      torch.where(hops <= l3_capacity, base * 3.0, base * 5.0))))
            penalties.append(torch.sum(staged))
        return sum(penalties)

# ======================
# Main (Unified: single and batched)
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--size', type=int, default=20, help="matrix size (M=N=K, square matrices)")
    parser.add_argument('--anneal', action='store_true', help="Enable annealing schedule for lr and sorting method")
    parser.add_argument('--anneal_steps', type=int, default=200, help="Number of steps to anneal over")
    parser.add_argument('--sinkhorn_temp_start', type=float, default=0.5, help="Initial temperature for Sinkhorn")
    parser.add_argument('--sinkhorn_temp_end', type=float, default=0.05, help="Final temperature for Sinkhorn")
    parser.add_argument('--display', type=str, default='side_by_side',
        help="Display mode: side_by_side, current_only, best_only (accepts dash or underscore)")
    parser.add_argument('--batch', type=int, default=0, help="Batch size for batched solver (0=disable, >0=enable batched mode)")
    args, unknown = parser.parse_known_args()
    display_mode = args.display.replace('-', '_')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    M,N,K = args.size, args.size, args.size
    A_labels = label_matrix("A", (M,K), ["ka","ki","ku","ke","ko"])
    B_labels = label_matrix("B", (K,N), ["sa","si","su","se","so"])
    C_labels = label_matrix("C", (M,N), ["wa","wi","wu","we","wo"])
    D_labels = label_matrix("D", (M,N), ["da","di","du","de","do"])
    logical_order = list(np.concatenate([A_labels.flatten(), B_labels.flatten(), C_labels.flatten(), D_labels.flatten()]))

    hsv_config = {
        'H': 'temporal_order',
        'S': 'memory_index',
        'V': 'memory_location',
        'line_start': {'H': 0.0, 'S': 'memory_distance', 'V': 1.0, 'A': 0.5},
        'line_end': {'H': 0.6, 'S': 'memory_distance', 'V': 1.0, 'A': 0.5},
    }
    border_hsv_config = {'H': 'matrix_index', 'S': 'matrix_location', 'V': 'matrix_location'} 
    shown_groups = ('A','B','C','D')
    separate_matrices = True
    best_loss = float('inf')
    best_state = None

    # === Unified Batched Solver Mode (batch size 1 = single instance) ===
    batch_size = args.batch if args.batch > 0 else 1
    model = BatchedTorchParametricSolver([A_labels,B_labels,C_labels,D_labels]).to(device)
    # Register mem_logits_batch as a true nn.Parameter inside the model
    model.mem_logits_batch = nn.Parameter(torch.randn(batch_size, model.num_elements, device=device))
    model.register_parameter('mem_logits_batch', model.mem_logits_batch)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)

    best_loss = [float('inf')] * batch_size
    best_state = [{} for _ in range(batch_size)]

    def image_sequence():
        global best_loss, best_state
        for step in range(args.steps):
            # Annealing schedule for learning rate and sorting method
            if args.anneal:
                frac = min(step / max(1, args.anneal_steps), 1.0)
                lr = args.lr * (1.0 - frac) + (args.lr * 0.1) * frac
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                sinkhorn_temp = args.sinkhorn_temp_start * (1.0 - frac) + args.sinkhorn_temp_end * frac
                use_sinkhorn = frac < 1.0
            else:
                use_sinkhorn = False
                sinkhorn_temp = args.sinkhorn_temp_end

            optimizer.zero_grad()
            # Forward pass for all batch elements using model.mem_logits_batch, with REINFORCE for both mem and op logits
            inter_pen_batch, intra_pen_batch, mat_ops_batch, addr_maps_batch, op_logprobs_batch, mem_logprobs_batch = model.from_logits(model.mem_logits_batch)
            loss_batch = inter_pen_batch + intra_pen_batch
            # Compound REINFORCE: multiply loss by both logprobs (score function estimator for both sources of stochasticity)
            reinforce_loss = (- loss_batch * (op_logprobs_batch + mem_logprobs_batch)).mean()
            loss = reinforce_loss

            # Track best for each batch element
            for b in range(batch_size):
                if loss_batch[b].item() < best_loss[b]:
                    best_loss[b] = loss_batch[b].item()
                    # Cache best state for this batch element
                    def to_cpu_addr_map(addr_map):
                        return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k,v in addr_map.items()}
                    def to_cpu_mat_ops(mat_ops):
                        if mat_ops is None:
                            return None
                        out = []
                        for op_type, (dst_label, dst_addr), sources in mat_ops:
                            dst_addr_cpu = dst_addr.detach().cpu() if torch.is_tensor(dst_addr) else dst_addr
                            sources_cpu = [(src_label, src_addr.detach().cpu() if torch.is_tensor(src_addr) else src_addr) for src_label, src_addr in sources]
                            out.append((op_type, (dst_label, dst_addr_cpu), sources_cpu))
                        return out
                    best_state[b]['addr_map'] = to_cpu_addr_map(addr_maps_batch[b])
                    best_state[b]['mat_ops'] = to_cpu_mat_ops(mat_ops_batch[b])
                    # Also cache the best image and the data needed to recreate it
                    temporal_usage_best = {label: idx for idx, (op, (dst_label,_), _) in enumerate(mat_ops_batch[b])
                                          for label in [dst_label] + [src for src,_ in _]}
                    max_time_best = max(1, max(temporal_usage_best.values(), default=0))
                    best_state[b]['temporal_usage'] = temporal_usage_best
                    best_state[b]['max_time'] = max_time_best
                    best_state[b]['img_best'] = generate_overlay_image(
                        best_state[b]['addr_map'],
                        best_state[b]['mat_ops'],
                        logical_order, D_labels,
                        hsv_config=hsv_config,
                        border_hsv_config=border_hsv_config,
                        shown_groups=shown_groups,
                        separate_matrices=False,
                        temporal_usage=temporal_usage_best,
                        max_time=max_time_best
                    )

            loss.backward()
            if False:
                print(f"\n=== Step diagnostics ===")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
                    else:
                        print(f"{name}: grad is None")
            for name, param in model.named_parameters():
                before = param.data.norm().item()
                optimizer.step()
                after = param.data.norm().item()
                #print(f"{name}: weight norm before={before:.6f} after={after:.6f}")

            # Visualization frequency: every 2 steps for batch, every 200 for single
            vis_freq = 2 if batch_size > 1 else 2
            if step % vis_freq == 0:
                # Visualize the first batch element (or only element)
                b = 0
                mat_ops = mat_ops_batch[b]
                addr_map = addr_maps_batch[b]
                temporal_usage = {label: idx for idx, (op, (dst_label,_), _) in enumerate(mat_ops)
                                  for label in [dst_label] + [src for src,_ in _]}
                max_time = max(1, max(temporal_usage.values(), default=0))
                def to_cpu_addr_map(addr_map):
                    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k,v in addr_map.items()}
                def to_cpu_mat_ops(mat_ops):
                    if mat_ops is None:
                        return None
                    out = []
                    for op_type, (dst_label, dst_addr), sources in mat_ops:
                        dst_addr_cpu = dst_addr.detach().cpu() if torch.is_tensor(dst_addr) else dst_addr
                        sources_cpu = [(src_label, src_addr.detach().cpu() if torch.is_tensor(src_addr) else src_addr) for src_label, src_addr in sources]
                        out.append((op_type, (dst_label, dst_addr_cpu), sources_cpu))
                    return out
                cpu_addr_map = to_cpu_addr_map(addr_map)
                cpu_mat_ops = to_cpu_mat_ops(mat_ops)
                # Select image to display based on display_mode
                if display_mode == 'best_only' and best_state[b].get('img_best') is not None:
                    img_current = best_state[b]['img_best']
                else:
                    img_current = generate_overlay_image(
                        cpu_addr_map, cpu_mat_ops, logical_order, D_labels,
                        hsv_config=hsv_config,
                        border_hsv_config=border_hsv_config,
                        shown_groups=shown_groups,
                        separate_matrices=True,
                        temporal_usage=temporal_usage,
                        max_time=max_time
                    )
                tag = '[BATCH]' if batch_size > 1 else ''
                print(f"{tag} Step {step:4d} Loss={loss.item():.2f} Best={min(best_loss):.2f} LR={optimizer.param_groups[0]['lr']:.4f} Sinkhorn={use_sinkhorn}")
                yield (img_current, best_state[b].get('img_best', None))

    run_pygame_gui(image_sequence, display_mode=display_mode)
