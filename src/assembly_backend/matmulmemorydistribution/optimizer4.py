#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sympy
import math as pymath
import os
import pygame
import matplotlib.pyplot as plt
import colorsys
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

MAX_WINDOW_WIDTH = 1000
MAX_WINDOW_HEIGHT = 700

# ======================
# Matrix setup
# ======================

# --- Modernized: label_matrix_ids returns integer IDs and offset
def label_matrix_ids(shape, offset=0):
    total = np.prod(shape)
    matrix_ids = np.arange(offset, offset + total).reshape(shape)
    return matrix_ids, offset + total

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
    addr_map, operations, logical_order, D_ids,
    hsv_config,
    border_hsv_config=None,
    shown_groups=None,  # unused, kept for signature compatibility
    separate_matrices=False,  # unused now
    temporal_usage=None,
    max_time=1
):
    import pygame
    import math

    total_elements = len(logical_order)
    min_S, max_S = 0.2, 1.0
    min_V, max_V = 0.2, 1.0

    # Map addresses to rank order
    unique_addrs_sorted = sorted(set(addr_map[label] for label in logical_order))
    addr_to_rank = {addr: idx for idx, addr in enumerate(unique_addrs_sorted)}
    ranks = np.array([addr_to_rank[addr_map[label]] for label in logical_order])
    norm_addrs = ranks / (len(unique_addrs_sorted) - 1 + 1e-6)
    max_time = max(temporal_usage.values()) if temporal_usage else 1

    # Arrange in a square grid
    grid_size = int(np.ceil(np.sqrt(len(unique_addrs_sorted))))
    def addr_to_xy(addr):
        rank = addr_to_rank[addr]
        return divmod(rank, grid_size)
    tile_xy = {label: addr_to_xy(addr_map[label]) for label in logical_order}

    width, height = grid_size, grid_size
    scale = 20

    surf = pygame.Surface((width * scale, height * scale))
    surf.fill((128,128,128))

    # --- Draw tiles ---
    for i, label in enumerate(logical_order):
        x, y = tile_xy[label]
        mem_idx = i / total_elements
        mem_loc = norm_addrs[i]
        temporal = temporal_usage.get(label, 0) / max_time if temporal_usage else 0.0
        source_idx = matrix_of[label] / 4.0     # 0.0,0.25,0.5,0.75 for A,B,C,D
        matrix_pos = 0.5
        H = select_value_from_config(hsv_config['H'], mem_idx, mem_loc, temporal, source_idx, matrix_pos)
        S = min_S + select_value_from_config(hsv_config['S'], mem_idx, mem_loc, temporal, source_idx, matrix_pos) * (max_S - min_S)
        V = min_V + select_value_from_config(hsv_config['V'], mem_idx, mem_loc, temporal, source_idx, matrix_pos) * (max_V - min_V)
        rgb = tuple(int(255*x) for x in colorsys.hsv_to_rgb(H, S, V))

        Hb = select_value_from_config(border_hsv_config['H'], mem_idx, mem_loc, temporal, source_idx, matrix_pos)
        Sb = min_S + select_value_from_config(border_hsv_config['S'], mem_idx, mem_loc, temporal, source_idx, matrix_pos) * (max_S - min_S)
        Vb = min_V + select_value_from_config(border_hsv_config['V'], mem_idx, mem_loc, temporal, source_idx, matrix_pos) * (max_V - min_V)
        border_rgb = tuple(int(255*x) for x in colorsys.hsv_to_rgb(Hb, Sb, Vb))

        border_rect = pygame.Rect(x*scale, y*scale, scale, scale)
        pygame.draw.rect(surf, border_rgb, border_rect)
        inset = int(scale * 0.15)
        inner_size = scale - 2*inset
        inner_rect = pygame.Rect(x*scale+inset, y*scale+inset, inner_size, inner_size)
        pygame.draw.rect(surf, rgb, inner_rect)

    # --- Draw lines ---
    n_segments = 16
    line_offset_radius = scale * 0.3
    def compute_offset(seed_tuple):
        seed = hash(seed_tuple) % 360
        angle = (seed / 360.0) * 2 * math.pi
        return math.cos(angle) * line_offset_radius, math.sin(angle) * line_offset_radius

    max_mem_distance = len(addr_to_rank) - 1 if len(addr_to_rank) > 1 else 1
    for idx, (op_type, (dst_id, dst_addr), sources) in enumerate(operations):
        dst_x, dst_y = tile_xy.get(dst_id, (None,None))
        if dst_x is None: continue
        dst_off_x, dst_off_y = compute_offset((idx, dst_id))
        dst_px = dst_x*scale + scale//2 + dst_off_x
        dst_py = dst_y*scale + scale//2 + dst_off_y
        dst_rank = addr_to_rank.get(dst_addr, 0)
        for src_id, src_addr in sources:
            src_x, src_y = tile_xy.get(src_id, (None,None))
            if src_x is None: continue
            src_off_x, src_off_y = compute_offset((idx, src_id))
            src_px = src_x*scale + scale//2 + src_off_x
            src_py = src_y*scale + scale//2 + src_off_y
            src_rank = addr_to_rank[src_addr]
            mem_distance = abs(dst_rank - src_rank)
            norm_mem_distance = mem_distance / max_mem_distance

            # Simplified coloring
            H0 = select_value_from_config(hsv_config['line_start']['H'], 0, 0, 0, 0, 0, norm_mem_distance)
            S0 = select_value_from_config(hsv_config['line_start']['S'], 0, 0, 0, 0, 0, norm_mem_distance)
            V0 = select_value_from_config(hsv_config['line_start']['V'], 0, 0, 0, 0, 0, norm_mem_distance)
            H1 = select_value_from_config(hsv_config['line_end']['H'], 0, 0, 0, 0, 0, norm_mem_distance)
            S1 = select_value_from_config(hsv_config['line_end']['S'], 0, 0, 0, 0, 0, norm_mem_distance)
            V1 = select_value_from_config(hsv_config['line_end']['V'], 0, 0, 0, 0, 0, norm_mem_distance)

            for seg in range(n_segments):
                t0 = seg / n_segments
                t1 = (seg + 1) / n_segments
                x0 = src_px * (1-t0) + dst_px * t0
                y0 = src_py * (1-t0) + dst_py * t0
                x1 = src_px * (1-t1) + dst_px * t1
                y1 = src_py * (1-t1) + dst_py * t1
                dH = H1 - H0
                if abs(dH) > 0.5:
                    dH -= round(dH)
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
        # --- Enforce max window size cap ---
        if w > MAX_WINDOW_WIDTH or h > MAX_WINDOW_HEIGHT:
            scale_w = MAX_WINDOW_WIDTH / w
            scale_h = MAX_WINDOW_HEIGHT / h
            scale_factor = min(scale_w, scale_h, 1.0)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = pygame.transform.smoothscale(pygame.surfarray.make_surface(np.transpose(image, (1,0,2))), (new_w, new_h))
            image = pygame.surfarray.array3d(image)
            image = np.transpose(image, (1,0,2))
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
    def __init__(self, operand_id_matrices, max_matrix_shape=None, max_total_elements=None, attn_embed_dim=32, attn_layers=1):
        super().__init__()
        self.operand_id_matrices = operand_id_matrices
        self.operand_ids = [ids.flatten() for ids in operand_id_matrices]
        self.all_ids = np.concatenate(self.operand_ids)
        self.num_elements = len(self.all_ids)
        # Build op_targets and op_sources as tensors
        # --- inside BatchedTorchParametricSolver.__init__ ---
        target_ids   = operand_id_matrices[-1]
        K            = operand_id_matrices[0].shape[1]          # inner dimension
        op_targets   = []
        op_sources   = []

        # --- build one C-only op  +  K A*B ops per D tile -----------------
        target_ids = operand_id_matrices[-1]          # D
        K          = operand_id_matrices[0].shape[1]  # inner dimension

        op_targets  = []
        op_sources  = []

        for i in range(target_ids.shape[0]):          # rows of D
            for j in range(target_ids.shape[1]):      # cols of D
                C_id = operand_id_matrices[2][i, j]   # C(i,j)

                # bias / accumulator once
                op_targets.append(target_ids[i, j])
                op_sources.append([C_id])             # <-- just C

                # K separate multiplies
                for k in range(K):
                    op_targets.append(target_ids[i, j])
                    op_sources.append([
                        operand_id_matrices[0][i, k],   # A(i,k)
                        operand_id_matrices[1][k, j],   # B(k,j)
                    ])

        # Turn the nested Python lists into tensors
        self.op_targets = torch.tensor(op_targets, dtype=torch.long, device='cuda')
        # Pad the ragged source lists with a sentinel (-1) so they can form a tensor
        max_src = max(len(srcs) for srcs in op_sources)   # = K or 1
        src_tensor = -torch.ones(len(op_sources), max_src, dtype=torch.long)
        for row, srcs in enumerate(op_sources):
            src_tensor[row, :len(srcs)] = torch.tensor(srcs)
        self.op_sources = src_tensor.to('cuda')

        self.num_ops     = self.op_targets.shape[0]
        self.num_sources = max_src          # now 1 or 2

        # Update the pretty-op list so it matches the ragged sources
        self.pretty_ops = []
        for tgt, srcs in zip(op_targets, op_sources):
            if len(srcs) == 1:
                self.pretty_ops.append(f"D{tgt} += C{srcs[0]}")
            else:
                self.pretty_ops.append(f"D{tgt} += A{srcs[0]} * B{srcs[1]}")

        # Cache index grids and constants for build_ops_batch
        MAX_M, MAX_K = operand_id_matrices[0].shape
        MAX_N = operand_id_matrices[1].shape[1]

        self.MAX_M = MAX_M
        self.MAX_K = MAX_K
        self.MAX_N = MAX_N

        self.register_buffer('row_idx', torch.arange(MAX_M).view(1, MAX_M, 1, 1))
        self.register_buffer('col_idx', torch.arange(MAX_N).view(1, 1, MAX_N, 1))
        self.register_buffer('k_idx', torch.arange(MAX_K).view(1, 1, 1, MAX_K))

        # --- Max shape/capacity for padding ---
        if max_matrix_shape is None:
            max_matrix_shape = tuple(max(mat.shape[i] for mat in operand_id_matrices) for i in range(2))
        self.max_matrix_shape = max_matrix_shape
        if max_total_elements is None:
            max_total_elements = sum(np.prod(max_matrix_shape) for _ in operand_id_matrices)
        self.max_total_elements = max_total_elements

        # Per-matrix convs (all matrices padded to max shape)
        self.per_matrix_convs = nn.ModuleList([
            nn.Conv2d(1, 8, kernel_size=3, padding=1) for _ in operand_id_matrices
        ])
        self.token_proj = nn.Linear(8, attn_embed_dim)
        # Memory space conv (memory space padded to max possible shape)
        self.memory_space_conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.memory_space_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.memory_space_proj = nn.Linear(16 * 4 * 4, self.num_ops)
        self.memory_lane_width = 8

        # Transformer for global reasoning
        self.attn_embed_dim = attn_embed_dim
        self.attn_layers = attn_layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=attn_embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        self.op_proj = nn.Linear(attn_embed_dim, self.num_ops)

    # === micro batching support =========================================
    def _attn_forward(self, tokens_proj, attn_mask, micro):
        """
        Run the encoder in chunks of `micro` along the *batch* dimension
        to cap peak memory.  If micro <=0 we use the plain call.
        """
        if micro <= 0 or tokens_proj.size(0) <= micro:
            return self.transformer(tokens_proj,
                                    src_key_padding_mask=attn_mask)
        chunks = []
        for s in range(0, tokens_proj.size(0), micro):
            e = s + micro
            chunk = self.transformer(tokens_proj[s:e],
                                     src_key_padding_mask=attn_mask[s:e])
            chunks.append(chunk)
        return torch.cat(chunks, 0)


    def from_logits(self, mem_logits_batch, reinforce=True, matrix_shapes=None, matrix_masks=None, attn_micro=0):
        batch_size = mem_logits_batch.shape[0]
        device = mem_logits_batch.device
        B, N = mem_logits_batch.shape

        # --- Batched Gumbel sampling + argsort
        gumbel = -torch.empty_like(mem_logits_batch).exponential_().log()
        sampled_mem_order = torch.argsort(mem_logits_batch + gumbel, dim=1)  # (B, N)

        # --- Batched mem_ranks
        mem_ranks = torch.zeros_like(sampled_mem_order)
        mem_ranks.scatter_(1, sampled_mem_order, torch.arange(N, device=device).unsqueeze(0).expand(B, N))

        # --- Per-matrix padding/masking ---
        max_H, max_W = self.max_matrix_shape
        per_matrix_features = []
        per_matrix_masks = []
        start = 0
        for idx, mat in enumerate(self.operand_id_matrices):
            H, W = mat.shape
            num = H * W
            mat_indices = mem_ranks[:, start:start+num].reshape(B, 1, H, W).float()
            # Pad to (B, 1, max_H, max_W)
            pad = (0, max_W - W, 0, max_H - H)
            mat_padded = F.pad(mat_indices, pad)
            feats = F.relu(self.per_matrix_convs[idx](mat_padded))  # (B, 8, max_H, max_W)
            per_matrix_features.append(feats)
            # Mask: 1 for valid, 0 for pad
            mask = torch.zeros(B, 1, max_H, max_W, device=device)
            mask[:, :, :H, :W] = 1
            per_matrix_masks.append(mask)
            start += num

        # --- Concatenate all per-matrix features and masks ---
        all_feats = torch.cat([f.view(B, 8, -1) for f in per_matrix_features], dim=2)  # (B, 8, total)
        all_masks = torch.cat([m.view(B, 1, -1) for m in per_matrix_masks], dim=2)     # (B, 1, total)
        total_tokens = all_feats.shape[2]

        # --- Memory space conv: build memory space from all per-matrix features ---
        # For simplicity, treat memory space as a single (B, 8, max_mem_rows, max_mem_cols) tensor
        lane_width = self.memory_lane_width
        n_tiles = self.max_total_elements
        n_rows = int(np.ceil(n_tiles / lane_width))
        memory_space = torch.zeros(B, 8, n_rows, lane_width, device=device)
        # Fill memory_space with per-matrix features (flattened, padded)
        flat_feats = all_feats.view(B, 8, -1)
        flat_mask = all_masks.view(B, 1, -1)
        # Only fill up to n_rows*lane_width
        mem_tokens = min(flat_feats.shape[2], n_rows * lane_width)
        memory_space_flat = memory_space.view(B, 8, -1)
        memory_space_flat[:, :, :mem_tokens] = flat_feats[:, :, :mem_tokens] * flat_mask[:, :, :mem_tokens]
        memory_space = memory_space_flat.view(B, 8, n_rows, lane_width)
        # Mask for memory space
        memory_mask = torch.zeros(B, 1, n_rows * lane_width, device=device)
        memory_mask[:, :, :mem_tokens] = flat_mask[:, :, :mem_tokens]

        # --- Memory space conv and pooling ---
        mem_conv = F.relu(self.memory_space_conv(memory_space))
        mem_pooled = self.memory_space_pool(mem_conv)
        mem_flat = mem_pooled.view(B, -1)
        mem_proj = self.memory_space_proj(mem_flat)  # (B, num_ops)

        # --- Transformer block over all tokens (per-matrix + memory space) ---
        # Prepare transformer input: concat per-matrix tokens and memory tokens
        # (B, 8, total_tokens) + (B, 8, n_rows*lane_width) -> (B, seq_len, embed)
        per_matrix_tokens = all_feats.permute(0, 2, 1)  # (B, total_tokens, 8)
        memory_tokens = memory_space.view(B, 8, -1).permute(0, 2, 1)  # (B, n_rows*lane_width, 8)
        tokens = torch.cat([per_matrix_tokens, memory_tokens], dim=1)  # (B, seq_len, 8)
        # Project to attn_embed_dim
        tokens_proj = F.relu(self.token_proj(tokens))
        # Build mask: 1 for valid, 0 for pad
        tokens_mask = torch.cat([
            all_masks.view(B, -1),
            memory_mask.view(B, -1)
        ], dim=1)  # (B, seq_len)
        attn_mask = (tokens_mask == 0)  # True for pad
        # Transformer
        x = self._attn_forward(tokens_proj, attn_mask, micro= getattr(torch._C._get_tracing_state(), 'attn_micro', None)
                                      or 0)   # fallback when scripted
        # Pooling: mean over non-masked positions
        mask_float = (~attn_mask).float()
        x_pooled = (x * mask_float.unsqueeze(2)).sum(dim=1) / (mask_float.sum(dim=1, keepdim=True) + 1e-6)
        op_logits = self.op_proj(x_pooled) + mem_proj  # (B, num_ops)

        gumbel_op = -torch.empty_like(op_logits).exponential_().log()
        sampled_op_order = torch.argsort(op_logits + gumbel_op, dim=1)  # (B, num_ops)

        # -------------------------------- dump hooks ------------------------
        if getattr(self, '_dump_stage', None) == 'sampled':
            if (not torch.distributed.is_available() or
                not torch.distributed.is_initialized() or
                torch.distributed.get_rank() == 0):
                for row, op_idx in enumerate(sampled_op_order[0]):
                    tgt  = int(self.op_targets[op_idx])
                    srcs = [int(s) for s in self.op_sources[op_idx] if s>=0]
                    if len(srcs)==1:
                        print(f"{row:3d}: D{tgt} += C{srcs[0]}")
                    else:
                        print(f"{row:3d}: D{tgt} += A{srcs[0]} * B{srcs[1]}")
            import sys
            sys.exit(0)
        # -------------------------------------------------------------------

        # --- Compute batched REINFORCE log-probs for memory assignment ---
        log_probs_mem = torch.log_softmax(mem_logits_batch, dim=1)
        sorted_log_probs_mem = torch.gather(log_probs_mem, 1, sampled_mem_order)
        mem_logprobs_batch = sorted_log_probs_mem.sum(dim=1)  # (B,)

        # --- Compute batched REINFORCE log-probs for operation ordering ---
        log_probs_op = torch.log_softmax(op_logits, dim=1)
        sorted_log_probs_op = torch.gather(log_probs_op, 1, sampled_op_order)
        op_logprobs_batch = sorted_log_probs_op.sum(dim=1)  # (B,)

        # --- Build sequences for penalty computation
        op_sources_exp = self.op_sources.unsqueeze(0).expand(B, -1, -1)  # (B, num_ops, num_sources)
        op_targets_exp = torch.gather(
            self.op_targets.unsqueeze(0).expand(B, -1), 
            1, sampled_op_order
        )  # (B, num_ops)
        seq = torch.cat([op_sources_exp, op_targets_exp.unsqueeze(2)], dim=2)  # (B, num_ops, num_sources+1)

        # Map seq IDs to addresses
        mem_ranks_exp = mem_ranks.unsqueeze(1).expand(-1, seq.shape[1], -1)  # (B, num_ops, N)
        # NEW – safe gather
        seq_valid   = seq.ge(0)                    # True where real id
        seq_clamped = seq.clamp(min=0)             # -1 → 0 (any valid index)
        seq_addrs   = torch.gather(mem_ranks_exp, 2, seq_clamped)
        seq_addrs   = seq_addrs * seq_valid        # mask out dummies

        # --- Compute penalties
        hops = seq_addrs[:,:,1:] - seq_addrs[:,:,:-1]
        intra_forward_hops = torch.relu(hops)
        intra_backward_hops = torch.relu(-hops)

        first_srcs = seq_addrs[:,:,0]
        last_dsts = seq_addrs[:,:,-1]
        inter_forward_hops = torch.relu(first_srcs[:,1:] - last_dsts[:,:-1])
        inter_backward_hops = torch.relu(last_dsts[:,:-1] - first_srcs[:,1:])

        # --- Dump hops if requested ---
        if getattr(self, '_dump_stage', None) == 'hops':
            if (not torch.distributed.is_available() or
                not torch.distributed.is_initialized() or
                torch.distributed.get_rank() == 0):
                # Only print for the first batch element
                op_order = sampled_op_order[0]
                seq_addrs0 = seq_addrs[0]
                for row, op_idx in enumerate(op_order):
                    tgt  = int(self.op_targets[op_idx])
                    srcs = [int(s) for s in self.op_sources[op_idx] if s>=0]
                    addrs = [int(seq_addrs0[row, i].item()) for i in range(len(srcs)+1)]
                    hop_str = ''
                    if len(srcs) == 1:
                        hop_str = f"hops: {addrs[1] - addrs[0]}"
                        print(f"{row:3d}: D{tgt} += C{srcs[0]}   {hop_str}")
                    else:
                        hop_str = f"hops: {addrs[1] - addrs[0]}, {addrs[2] - addrs[1]}"
                        print(f"{row:3d}: D{tgt} += A{srcs[0]} * B{srcs[1]}   {hop_str}")
            import sys
            sys.exit(0)

        def staged_penalty(hops, exp):
            cache_line, l1, l2, l3 = 2, 4, 8, 16
            base = hops ** exp
            staged = torch.where(hops <= cache_line, base,
                      torch.where(hops <= l1, base * 1.5,
                      torch.where(hops <= l2, base * 2.0,
                      torch.where(hops <= l3, base * 3.0, base * 5.0))))
            return staged.mean(dim=(-1, -2) if staged.dim() > 2 else -1)

        inter_pen_batch = staged_penalty(inter_forward_hops, 1.0) + staged_penalty(inter_backward_hops, 2.0)
        intra_pen_batch = staged_penalty(intra_forward_hops, 1.0) + staged_penalty(intra_backward_hops, 2.0)

        return (
            inter_pen_batch.requires_grad_(),
            intra_pen_batch.requires_grad_(),
            mem_ranks.detach(),
            sampled_op_order.detach(),
            op_logprobs_batch,
            mem_logprobs_batch
        )

    def route_per_matrix_features(self, per_matrix_features, addr_map, lane_width, device):
        # Build flat memory ranks tensor
        mem_ranks = torch.tensor([addr_map[label] for label in self.all_ids], device=device)
        rows = (mem_ranks // lane_width).long()
        cols = (mem_ranks % lane_width).long()
        n_rows = int(torch.max(rows).item() + 1)

        # Build flat features
        feats = torch.cat([feat.view(8, -1) for feat in [f.squeeze(0) for f in per_matrix_features]], dim=1)

        # Prepare memory space and scatter
        memory_space = torch.zeros(1, 8, n_rows, lane_width, device=device)
        channels = torch.arange(8, device=device).view(-1, 1)
        memory_space[0].index_put_((channels, rows, cols), feats)
        return memory_space

    def build_per_matrix_inputs(self, addr_map, addr_to_rank, device):
        label_idx_lookup = torch.tensor([addr_map.get(label, 0) for label in self.all_ids], device=device)
        per_matrix_inputs = []
        start = 0
        for mat in self.operand_id_matrices:
            H, W = mat.shape
            num = H * W
            mat_indices = label_idx_lookup[start:start+num].reshape(H, W)
            per_matrix_inputs.append(mat_indices.unsqueeze(0).float())
            start += num
        return per_matrix_inputs
    def _build_ops_batch(self, MNK, A_ids, B_ids, D_ids):
        """
        MNK:  (B, 3)  tensor with M,N,K per sample   (on cuda)
        A_ids,B_ids,D_ids : (MAX_M,MAX_K) / (MAX_K,MAX_N) / (MAX_M,MAX_N)

        returns
            op_tgt   (B, L_max)          int64
            op_src   (B, L_max, 2)       int64
            valid    (B, L_max)          bool   (mask)
        """
        B = MNK.size(0)
        M, N, K = MNK[:,0], MNK[:,1], MNK[:,2]          # (B,)

        # ---------- broadcast indices to (B, MAX_M, MAX_N, MAX_K) ----------
        I = self.row_idx        # (1,MAX_M,1,1)
        J = self.col_idx        # (1,1,MAX_N,1)
        Kk= self.k_idx          # (1,1,1,MAX_K)

        # keep only rows / cols / ks that are inside each sample's bounds
        keep = (I < M.view(B,1,1,1)) & \
            (J < N.view(B,1,1,1)) & \
            (Kk< K.view(B,1,1,1))                       # (B, M,N,K)

        # flatten to (B, L_max)
        keep_f = keep.view(B, -1)
        L_max  = keep_f.sum(1).max().item()               # longest sequence

        # build flat rank tensors once
        flat_A = A_ids.view(-1)           # (MAX_M*MAX_K,)
        flat_B = B_ids.view(-1)
        flat_D = D_ids.view(-1)

        # linearised indices into A/B/D
        a_lin = (I * self.MAX_K + Kk).view(-1)                 # length MAX_M*MAX_K*MAX_N
        b_lin = (Kk* self.MAX_N + J ).view(-1)
        d_lin = (I * self.MAX_N + J ).view(-1)

        # gather and reshape back to (B, M,N,K)
        src0 = flat_A[a_lin].view_as(keep)                # int64
        src1 = flat_B[b_lin].view_as(keep)
        tgt  = flat_D[d_lin].view_as(keep)

        # apply mask & pad to L_max
        def _pad(x):
            x_f = x.view(B,-1)
            padded = torch.zeros(B, L_max, dtype=torch.long, device=x.device)
            for b in range(B):                        # still python loop *over batch only*
                n = keep_f[b].sum().item()
                padded[b,:n] = x_f[b, keep_f[b]]
            return padded

        op_tgt = _pad(tgt)
        op_src = torch.stack((_pad(src0), _pad(src1)), dim=2)  # (B,L_max,2)

        valid_mask = op_tgt != 0    # or keep_f padded the same way

        return op_tgt, op_src, valid_mask


# ======================
# Main (Unified: single and batched)
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--attn_micro', type=int, default=64,
                        help='inner mini-batch size for the transformer; '
                             '<=0 disables micro-batching')
    parser.add_argument('--fsdp_offload', action='store_true',
                        help='wrap the transformer in FSDP with cpu_offload')
    parser.add_argument('--autocast', action='store_true', help='Enable torch.cuda.amp.autocast for mixed precision')
    parser.add_argument('--size', type=int, default=20)
    parser.add_argument('--anneal', action='store_true')
    parser.add_argument('--anneal_steps', type=int, default=200)
    parser.add_argument('--sinkhorn_temp_start', type=float, default=0.5)
    parser.add_argument('--sinkhorn_temp_end', type=float, default=0.05)
    parser.add_argument('--display', type=str, default='side_by_side')
    parser.add_argument('--batch', type=int, default=0)

    parser.add_argument('--dump_stage', choices=['asm','sampled','hops'],
                    help='Dump ops right after assembly (asm), after the '
                         'sampled ordering (sampled) or after address '
                         'translation / hop-building (hops) and exit')

    # --- Logits controls ---
    parser.add_argument('--logits_lr', type=float, default=0.1)
    parser.add_argument('--logits_lr_anneal', type=str, default=None, help='Sympy expr for logits learning rate')
    parser.add_argument('--logits_grad_clip', type=str, default=None, help='Sympy expr for logits grad norm cap')
    parser.add_argument('--logits_grad_accum', type=str, default=None, help='Sympy expr for logits grad accumulation steps')
    parser.add_argument('--logits_optim', type=str, default='adam', choices=['adam','adamw','sgd','rmsprop'], help='Optimizer for logits')
    # --- Network controls ---
    parser.add_argument('--net_lr', type=float, default=0.1)
    parser.add_argument('--net_lr_anneal', type=str, default=None, help='Sympy expr for network learning rate')
    parser.add_argument('--net_grad_clip', type=str, default=None, help='Sympy expr for network grad norm cap')
    parser.add_argument('--net_grad_accum', type=str, default=None, help='Sympy expr for network grad accumulation steps')
    parser.add_argument('--net_optim', type=str, default='adam', choices=['adam','adamw','sgd','rmsprop'], help='Optimizer for network')
    # --- Other controls ---
    parser.add_argument('--mutation_anneal', type=str, default=None, help='Sympy expression for mutation std, e.g. "0.2/(1+0.01*epoch)"')
    parser.add_argument('--elite_freeze', type=str, default='1', help='Sympy expr for how many rounds to freeze elite, e.g. "3" or "2+epoch//100"')
    parser.add_argument(
        '--preset',
        type=str,
        choices = [
            "explore30xAdam", "explore30xAdamW", "explore30xRMS",
            "explore30xRMS_AdamW"],
        default=None,
        help='Convenience presets for common experiments')
    args, unknown = parser.parse_known_args()
    display_mode = args.display.replace('-', '_')

    # ────────────────────────────────────────────────────────────────
    #  PRESETS – all tuned for batch = 4 000
    # ----------------------------------------------------------------
    preset = getattr(args, "preset", None)
    default_size = getattr(args, "size", 20)
    default_batch = getattr(args, "batch", 2000)



    def _setdefault(attr, value):
        if getattr(args, attr) is None:
            setattr(args, attr, value)

    if preset == "explore30xAdam":                         # (kept name, new batch)
        args.size   = default_size
        args.batch  = default_batch
        args.steps  = 8000

        # ─── optimise logits with Adam ───────────────────────────────
        _setdefault("logits_optim",        "adam")
        _setdefault("logits_lr_anneal",    "0.05*exp(-0.0002*step)")
        _setdefault("logits_grad_clip",    "1.0")
        _setdefault("logits_grad_accum",   "8")

        # ─── optimise network with Adam ──────────────────────────────
        _setdefault("net_optim",           "adam")
        _setdefault("net_lr_anneal",       "0.05*exp(-0.0002*step)")
        _setdefault("net_grad_clip",       "1.0")
        _setdefault("net_grad_accum",      "16")

        # ─── genetic / mutation knobs ────────────────────────────────
        _setdefault("mutation_anneal",     "0.25 / (1 + 0.0001*step)")
        _setdefault("elite_freeze",        "1")


    elif preset == "explore30xAdamW":
        args.size  = default_size
        args.batch = default_batch
        args.steps = 8000

        # AdamW everywhere (decoupled L2 ­– good when logits wander)
        _setdefault("logits_optim",        "adamw")
        _setdefault("net_optim",           "adamw")

        _setdefault("logits_lr_anneal",    "0.04*exp(-0.00025*step)")
        _setdefault("net_lr_anneal",       "0.04*exp(-0.00025*step)")

        _setdefault("logits_grad_clip",    "1.0")
        _setdefault("net_grad_clip",       "1.0")

        _setdefault("logits_grad_accum",   "8")
        _setdefault("net_grad_accum",      "32")

        _setdefault("mutation_anneal",     "0.25 / (1 + 0.0001*step)")
        _setdefault("elite_freeze",        "1")


    elif preset == "explore30xRMS":
        args.size  = default_size
        args.batch = default_batch
        args.steps = 9000

        # RMSprop for the high-variance REINFORCE logits,
        # Adam for the smoother CNN parameters.
        _setdefault("logits_optim",        "rmsprop")
        _setdefault("net_optim",           "adam")

        _setdefault("logits_lr_anneal",    "0.06*exp(-0.00022*step)")
        _setdefault("net_lr_anneal",       "0.05*exp(-0.00022*step)")

        _setdefault("logits_grad_clip",    "0.8")
        _setdefault("net_grad_clip",       "1.0")

        _setdefault("logits_grad_accum",   "8")
        _setdefault("net_grad_accum",      "16")

        _setdefault("mutation_anneal",     "0.25 / (1 + 0.0001*step)")
        _setdefault("elite_freeze",        "1")


    elif preset == "explore30xRMS_AdamW":          # NEW combo you asked for
        args.size  = default_size
        args.batch = default_batch
        args.steps = 9000

        # RMSprop → logits  |  AdamW → network
        _setdefault("logits_optim",        "rmsprop")
        _setdefault("net_optim",           "adamw")

        _setdefault("logits_lr_anneal",    "0.06*exp(-0.00022*step)")
        _setdefault("net_lr_anneal",       "0.04*exp(-0.00025*step)")

        _setdefault("logits_grad_clip",    "0.8")
        _setdefault("net_grad_clip",       "1.0")

        _setdefault("logits_grad_accum",   "8")
        _setdefault("net_grad_accum",      "32")

        _setdefault("mutation_anneal",     "0.25 / (1 + 0.0001*step)")
        _setdefault("elite_freeze",        "1")
    # ────────────────────────────────────────────────────────────────


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    M,N,K = args.size, args.size, args.size
    offset = 0
    A_ids, offset = label_matrix_ids((M,K), offset)
    B_ids, offset = label_matrix_ids((K,N), offset)
    C_ids, offset = label_matrix_ids((M,N), offset)
    D_ids, offset = label_matrix_ids((M,N), offset)
    logical_order = np.concatenate([A_ids.flatten(), B_ids.flatten(), C_ids.flatten(), D_ids.flatten()])
    # --------------------------------------------------------------------
    global matrix_of
    matrix_of = {}
    for m_id, _ in enumerate(['A','B','C','D']):          # 0,1,2,3
        for lbl in (A_ids, B_ids, C_ids, D_ids)[m_id].flatten():
            matrix_of[int(lbl)] = m_id                      # e.g. 0 for A
    # --------------------------------------------------------------------

    hsv_config = {
        'H': 'temporal_order',
        'S': 'memory_index',
        'V': 'memory_location',
        'line_start': {'H': 0.0, 'S': 'memory_distance', 'V': 1.0, 'A': 0.5},
        'line_end': {'H': 0.6, 'S': 'memory_distance', 'V': 1.0, 'A': 0.5},
    }
    border_hsv_config = {'H': 'matrix_index', 'S': 'matrix_location', 'V': 'matrix_location'}

    batch_size = args.batch if args.batch > 0 else 1
    # --- Pin host memory for mem_logits_batch and use non_blocking copy to device ---
    
    model = BatchedTorchParametricSolver([A_ids,B_ids,C_ids,D_ids]).to(device)
    # ---------- FSDP (param-on-CPU) wrapper ----------------------------------
    if args.fsdp_offload:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        model.transformer = FSDP(
            model.transformer,
            auto_wrap_policy=transformer_auto_wrap_policy,
            cpu_offload=True,                      # params & grads live on host
            mixed_precision=torch.float16,         # moved as fp16
        )
    # ------------------------------------------------------------------------
    mem_logits_batch_cpu = torch.randn(batch_size, model.num_elements, pin_memory=True)
    model.mem_logits_batch = nn.Parameter(mem_logits_batch_cpu.to(device, non_blocking=True))
    model.register_parameter('mem_logits_batch', model.mem_logits_batch)

    # --- Pre-allocate pad buffers for per-matrix and memory space ---
    max_H, max_W = model.max_matrix_shape
    per_matrix_pad_buffers = []
    per_matrix_mask_buffers = []
    for mat in model.operand_id_matrices:
        H, W = mat.shape
        pad = (0, max_W - W, 0, max_H - H)
        per_matrix_pad_buffers.append(torch.zeros(batch_size, 1, max_H, max_W, device=device))
        mask = torch.zeros(batch_size, 1, max_H, max_W, device=device)
        mask[:, :, :H, :W] = 1
        per_matrix_mask_buffers.append(mask)
    lane_width = model.memory_lane_width
    n_tiles = model.max_total_elements
    n_rows = int(np.ceil(n_tiles / lane_width))
    memory_space_buffer = torch.zeros(batch_size, 8, n_rows, lane_width, device=device)
    memory_mask_buffer = torch.zeros(batch_size, 1, n_rows * lane_width, device=device)
    # --- Split parameters ---
    logits_params = [model.mem_logits_batch]
    net_params = [p for n, p in model.named_parameters() if n != 'mem_logits_batch']
    # --- Optimizer selection logic ---
    def make_optimizer(opt_name, params, lr):
        if opt_name == 'adam':
            return torch.optim.Adam(params, lr=lr)
        elif opt_name == 'adamw':
            return torch.optim.AdamW(params, lr=lr)
        elif opt_name == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        elif opt_name == 'rmsprop':
            return torch.optim.RMSprop(params, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    logits_optimizer = make_optimizer(args.logits_optim, logits_params, args.logits_lr)
    net_optimizer = make_optimizer(args.net_optim, net_params, args.net_lr) if net_params else None

    global_best_loss = float('inf')
    global_best_state = {}

    import random
    # Sympy parsing utility
    def make_sympy_fn(expr, default=None):
        if expr is None:
            return lambda **kwargs: default
        syms = sympy.symbols('epoch step batch batch_size grad_accum substep')
        expr = sympy.sympify(expr, locals={"exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt, "sin": sympy.sin, "cos": sympy.cos, "pi": sympy.pi})
        def fn(**kwargs):
            return float(expr.evalf(subs=kwargs))
        return fn

    # --- Logits schedules ---
    logits_lr_fn = make_sympy_fn(args.logits_lr_anneal, default=args.logits_lr)
    logits_grad_clip_fn = make_sympy_fn(args.logits_grad_clip, default=None)
    logits_grad_accum_fn = make_sympy_fn(args.logits_grad_accum, default=1)
    # --- Network schedules ---
    net_lr_fn = make_sympy_fn(args.net_lr_anneal, default=args.net_lr)
    net_grad_clip_fn = make_sympy_fn(args.net_grad_clip, default=None)
    net_grad_accum_fn = make_sympy_fn(args.net_grad_accum, default=1)
    # --- Other ---
    mutation_fn = make_sympy_fn(args.mutation_anneal, default=0.2)
    elite_freeze_fn = make_sympy_fn(args.elite_freeze, default=1)

    def mutate_logits(logits, std=0.2):
        return logits + torch.randn_like(logits) * std

    def image_sequence():
        global global_best_loss, global_best_state
        population_size = model.mem_logits_batch.shape[0]
        elite_frac = 0.25
        random_frac = 0.1
        freeze_counter = torch.zeros(population_size, dtype=torch.long, device=device)
        # --- Separate accumulators for logits and net ---
        logits_grad_accum = logits_grad_accum_fn(epoch=0, step=0, batch=0, batch_size=population_size, grad_accum=1, substep=0)
        logits_accum_steps = int(logits_grad_accum)
        logits_accum_loss = 0.0
        logits_accum_count = 0
        net_grad_accum = net_grad_accum_fn(epoch=0, step=0, batch=0, batch_size=population_size, grad_accum=1, substep=0)
        net_accum_steps = int(net_grad_accum)
        net_accum_loss = 0.0
        net_accum_count = 0
        epoch = 0
        substep = 0
        # --- CUDA stream for overlapping datapath ---
        stream = torch.cuda.Stream() if device.type == 'cuda' else None
        autocast_enabled = args.autocast and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=autocast_enabled)
        for step in range(args.steps):
            # Progressive epoch tick for mutation schedule
            if step % population_size == 0 and step > 0:
                epoch += 1
            # --- Compute annealed lr, mutation std, grad clip, grad accum for each group ---
            logits_lr = logits_lr_fn(epoch=epoch, step=step, batch=step, batch_size=population_size, grad_accum=logits_accum_steps, substep=substep)
            logits_grad_clip = logits_grad_clip_fn(epoch=epoch, step=step, batch=step, batch_size=population_size, grad_accum=logits_accum_steps, substep=substep)
            new_logits_accum = logits_grad_accum_fn(epoch=epoch, step=step, batch=step, batch_size=population_size, grad_accum=logits_accum_steps, substep=substep)
            if int(new_logits_accum) != logits_accum_steps:
                logits_accum_steps = int(new_logits_accum)
                logits_accum_count = 0
            for param_group in logits_optimizer.param_groups:
                param_group['lr'] = logits_lr

            net_lr = net_lr_fn(epoch=epoch, step=step, batch=step, batch_size=population_size, grad_accum=net_accum_steps, substep=substep)
            net_grad_clip = net_grad_clip_fn(epoch=epoch, step=step, batch=step, batch_size=population_size, grad_accum=net_accum_steps, substep=substep)
            new_net_accum = net_grad_accum_fn(epoch=epoch, step=step, batch=step, batch_size=population_size, grad_accum=net_accum_steps, substep=substep)
            if int(new_net_accum) != net_accum_steps:
                net_accum_steps = int(new_net_accum)
                net_accum_count = 0
            if net_optimizer:
                for param_group in net_optimizer.param_groups:
                    param_group['lr'] = net_lr

            mutation_std = mutation_fn(epoch=epoch, step=step, batch=step, batch_size=population_size, grad_accum=logits_accum_steps, substep=substep)

            # --- Zero grads for each group as needed ---
            if logits_accum_count == 0:
                logits_optimizer.zero_grad(set_to_none=True)
            if net_optimizer and net_accum_count == 0:
                net_optimizer.zero_grad(set_to_none=True)

            # ──────────────────────────────────────────────────────────────
            #  Overlap datapath: enqueue from_logits on CUDA stream
            # ──────────────────────────────────────────────────────────────
            if autocast_enabled:
                autocast_ctx = torch.amp.autocast('cuda')
            else:
                autocast_ctx = torch.amp.autocast('cuda', enabled=False)
            if stream is not None:
                with torch.cuda.stream(stream):
                    with autocast_ctx:
                        forward_out = model.from_logits(model.mem_logits_batch, attn_micro=args.attn_micro)
            else:
                with autocast_ctx:
                    forward_out = model.from_logits(model.mem_logits_batch, attn_micro=args.attn_micro)

            with torch.no_grad():
                try:
                    torch._C._get_tracing_state().attn_micro = args.attn_micro
                except AttributeError:
                    pass

            # --- CPU-side work (mutation, bookkeeping, etc.) can go here if needed ---

            if stream is not None:
                stream.synchronize()  # Ensure forward is done before backward

            inter_pen_batch, intra_pen_batch, mem_ranks_batch, sampled_ops_batch, op_logprobs_batch, mem_logprobs_batch = forward_out
            loss_batch = inter_pen_batch + intra_pen_batch
            base_loss   = (- loss_batch * (op_logprobs_batch + mem_logprobs_batch)).mean()

            # scale once for logits, later rescale net grads in-place
            loss_scaled = base_loss / logits_accum_steps
            scaler.scale(loss_scaled).backward()

            # if the two groups use different accum_steps, rescale net grads
            if net_optimizer and net_accum_steps != logits_accum_steps:
                scale = logits_accum_steps / net_accum_steps   # e.g. 2/1 doubles net grads
                with torch.no_grad():
                    for p in net_params:
                        if p.grad is not None:
                            p.grad.mul_(scale)

            logits_accum_count += 1
            net_accum_count    += 1  # we always produced grads for both

            # ── clip+step only when each group hit its accumulation quota ──
            if logits_accum_count >= logits_accum_steps:
                if logits_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(logits_params, logits_grad_clip)
                scaler.step(logits_optimizer); scaler.update()
                logits_optimizer.zero_grad(set_to_none=True)
                logits_accum_count = 0

            if net_optimizer and net_accum_count >= net_accum_steps:
                if net_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(net_params, net_grad_clip)
                scaler.step(net_optimizer); scaler.update()
                net_optimizer.zero_grad(set_to_none=True)
                net_accum_count = 0
                substep += 1

            # Find best index in this batch
            best_idx = torch.argmin(loss_batch).item()
            mem_ranks_best = mem_ranks_batch[best_idx]
            sampled_ops_best = sampled_ops_batch[best_idx]

            addr_map_best = {int(model.all_ids[i]): int(mem_ranks_best[i].cpu())
                            for i in range(len(model.all_ids))}
            sampled_targets = model.op_targets[sampled_ops_best]
            temporal_usage_best = {int(sampled_targets[i].cpu()): i for i in range(sampled_targets.shape[0])}
            max_time_best = max(1, max(temporal_usage_best.values(), default=0))

            # Build pseudo operations on demand for drawing lines
            operations_best = []
            for op_idx in sampled_ops_best:
                tgt_id  = int(model.op_targets[op_idx])
                dst_addr= int(mem_ranks_best[tgt_id])
                raw_src = model.op_sources[op_idx]
                real_src= [(int(s), int(mem_ranks_best[int(s)]))
                           for s in raw_src if s >= 0]          # drop -1
                op_tag  = "+=C" if len(real_src)==1 else "+=A*B"
                operations_best.append((op_tag, (tgt_id, dst_addr), real_src))
                # Validate that sources map to the correct matrices
                for tag, (_, _), srcs in operations_best:
                    if tag == '+=C':
                        # bias op: single source must come from C (matrix id 2)
                        assert len(srcs) == 1 and matrix_of[srcs[0][0]] == 2, \
                            f"Expected C-source for bias, got {srcs}"
                    else:
                        # multiply op: sources must be A then B (ids 0 and 1)
                        assert len(srcs) == 2 and matrix_of[srcs[0][0]] == 0 and matrix_of[srcs[1][0]] == 1, \
                            f"Expected A then B sources for multiply, got {srcs}"

            # Now update global best
            if loss_batch[best_idx].item() < global_best_loss:
                global_best_loss = loss_batch[best_idx].item()
                global_best_state['addr_map'] = addr_map_best
                global_best_state['temporal_usage'] = temporal_usage_best
                global_best_state['max_time'] = max_time_best
                if display_mode in ('side_by_side', 'best_only'):
                    global_best_state['img_best'] = generate_overlay_image(
                        addr_map_best, operations_best, logical_order, D_ids,
                        hsv_config=hsv_config,
                        border_hsv_config=border_hsv_config,
                        temporal_usage=temporal_usage_best,
                        max_time=max_time_best
                    )

            # --- Genetic replacement: replace worst with mutated elite or random, freeze elite for N rounds ---
            # ----------------------------------------------
            # torch-only genetic step (no Python loops)
            # ----------------------------------------------
            with torch.no_grad():
                fitness = loss_batch                                # (B,)
                elite_n   = max(1, int(population_size * elite_frac))
                random_n  = max(1, int(population_size * random_frac))

                # ── rank population ─────────────────────────
                elite_val, elite_idx = torch.topk(-fitness, elite_n, largest=True)

                # ── update freeze counters ──────────────────
                freeze_rounds = int(elite_freeze_fn(
                    epoch=epoch, step=step, batch=step,
                    batch_size=population_size,
                    grad_accum=logits_accum_steps,
                    substep=substep))
                new_freeze = torch.zeros_like(freeze_counter)
                new_freeze[elite_idx] = freeze_rounds
                freeze_counter = torch.maximum(freeze_counter - 1, new_freeze)

                # ── choose rows that may be replaced ────────
                not_frozen = torch.nonzero(freeze_counter == 0, as_tuple=True)[0]
                n_mutate   = max(not_frozen.numel() - random_n, 0)

                # ── mutation from elite parents ─────────────
                if n_mutate > 0:
                    parent_rows = elite_idx[torch.randint(elite_n, (n_mutate,), device=device)]
                    noise       = torch.randn((n_mutate, model.num_elements), device=device) * mutation_std
                    model.mem_logits_batch.index_copy_(0, not_frozen[:n_mutate], 
                                                    model.mem_logits_batch[parent_rows] + noise)

                # ── random re-seed rows  ────────────────────
                if random_n > 0 and not_frozen.numel() >= random_n:
                    rand_rows = not_frozen[-random_n:]
                    model.mem_logits_batch[rand_rows] = torch.randn((random_n, model.num_elements), device=device)
                else:
                    rand_rows = torch.tensor([], dtype=torch.long, device=device)

                # ── clear Adam moments for *all* changed rows
                adam_state = logits_optimizer.state.get(model.mem_logits_batch, None)
                mutated_rows = torch.cat((not_frozen[:n_mutate], rand_rows)) if rand_rows.numel() else not_frozen[:n_mutate]
                if adam_state is not None and mutated_rows.numel():
                    if 'exp_avg' in adam_state:
                        adam_state['exp_avg'][mutated_rows]  = 0
                        adam_state['exp_avg_sq'][mutated_rows] = 0
                    if 'step' in adam_state:
                        if isinstance(adam_state['step'], torch.Tensor):
                            adam_state['step'].zero_()
                        else:
                            adam_state['step'] = 0


            # Build current image only if display mode requires it
            img_current = None
            img_best = global_best_state.get('img_best')

            if display_mode in ('side_by_side', 'current_only'):
                img_current = generate_overlay_image(
                    addr_map_best, operations_best, logical_order, D_ids,
                    hsv_config=hsv_config,
                    border_hsv_config=border_hsv_config,
                    temporal_usage=temporal_usage_best,
                    max_time=max_time_best
                )

            tag = '[BATCH+GEN]' if population_size > 1 else ''
            print(f"{tag} Step {step:4d} Loss={loss_batch[best_idx].item():.2f} Best={global_best_loss:.2f} "
                  f"LogitsLR={logits_lr:.4f} NetLR={net_lr:.4f} MutStd={mutation_std:.4f} "
                  f"LogitsClip={logits_grad_clip} NetClip={net_grad_clip} "
                  f"LogitsAccum={logits_accum_steps} NetAccum={net_accum_steps} Freeze={freeze_rounds}")

            if display_mode == 'side_by_side' and img_best is not None:
                yield (img_current, img_best)
            elif display_mode == 'current_only':
                yield img_current
            elif display_mode == 'best_only' and img_best is not None:
                yield img_best


    if args.dump_stage == 'asm':
        for line in model.pretty_ops:
            print(line)
        import sys
        sys.exit(0)
    elif args.dump_stage:
        # remember choice so from_logits can act
        model._dump_stage = args.dump_stage


    run_pygame_gui(image_sequence, display_mode=display_mode)
