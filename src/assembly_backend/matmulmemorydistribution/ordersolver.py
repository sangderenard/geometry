#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os

# ======================
# Output directories
# ======================
OUTPUT_BASE_DIR = "output"
SPECTRUM_DIR = os.path.join(OUTPUT_BASE_DIR, "spectrum")
ORIGINAL_RANK_DIR = os.path.join(OUTPUT_BASE_DIR, "original_rank")
FULL_OVERLAY_DIR = os.path.join(OUTPUT_BASE_DIR, "full_overlay")
for _dir in (OUTPUT_BASE_DIR, SPECTRUM_DIR, ORIGINAL_RANK_DIR, FULL_OVERLAY_DIR):
    os.makedirs(_dir, exist_ok=True)

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

# ======================
# Compute penalties
# ======================
def compute_penalty_metrics_with_intra(operations):
    inter_penalties, inter_running_sum, intra_penalties, intra_running_sum = [], [], [], []
    inter_reversals, intra_reversals = [], []

    acc_inter, acc_intra = 0, 0
    prev_endpoint = None
    last_direction_inter = None

    for op_type, (_, dst_addr), sources in operations:
        seq = [s[1] for s in sources] + [dst_addr]

        # inter
        if prev_endpoint is not None:
            inter_hop = seq[0] - prev_endpoint
            inter_penalty = abs(inter_hop)
            acc_inter += inter_penalty
            inter_penalties.append(inter_penalty)
            inter_running_sum.append(acc_inter)
            direction = np.sign(inter_hop) if inter_hop != 0 else last_direction_inter
            if last_direction_inter is not None and direction * last_direction_inter < 0:
                inter_reversals.append(inter_penalty)
            last_direction_inter = direction
        else:
            inter_running_sum.append(0)

        # intra
        last_direction_intra = None
        for i in range(len(seq)-1):
            hop = seq[i+1] - seq[i]
            penalty = abs(hop)
            acc_intra += penalty
            intra_penalties.append(penalty)
            intra_running_sum.append(acc_intra)
            direction = np.sign(hop) if hop != 0 else last_direction_intra
            if last_direction_intra is not None and direction * last_direction_intra < 0:
                intra_reversals.append(penalty)
            last_direction_intra = direction

        prev_endpoint = seq[-1]

    return sum(inter_penalties), sum(intra_penalties)

# ======================
# Solver class
# ======================
class ParametricMemoryOrderSolver:
    def __init__(self, A_labels, B_labels, C_labels, D_labels):
        self.A_labels = A_labels.flatten()
        self.B_labels = B_labels.flatten()
        self.C_labels = C_labels.flatten()
        self.D_labels = D_labels.flatten()
        self.all_labels = np.concatenate([self.A_labels, self.B_labels, self.C_labels, self.D_labels])
        self.num_elements = len(self.all_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        self.operations = build_flat_operations(A_labels, B_labels, C_labels, D_labels, A_labels.shape[0], D_labels.shape[1], A_labels.shape[1])

    def apply_memory(self, mem_params):
        sort_order = np.argsort(mem_params)
        addr_map = {self.all_labels[i]: idx for idx, i in enumerate(sort_order)}
        return addr_map

    def apply_op_order(self, op_params):
        sort_order = np.argsort(op_params)
        return [self.operations[i] for i in sort_order]

    def evaluate(self, mem_params, op_params):
        addr_map = self.apply_memory(mem_params)
        ordered_ops = self.apply_op_order(op_params)
        get_addr = lambda lbl: addr_map[lbl]
        # Materialize addresses
        materialized_ops = [(op_type, (dst_label, get_addr(dst_label)), [(src_label, get_addr(src_label)) for src_label,_ in sources])
                            for op_type, (dst_label,_), sources in ordered_ops]
        inter_pen, intra_pen = compute_penalty_metrics_with_intra(materialized_ops)
        return inter_pen, intra_pen, materialized_ops, addr_map

# ======================
# Plots (on demand)
# ======================
def plot_overlay(addr_map, operations, logical_order, D_labels, sample_index):
    unique_addrs = sorted(set(addr_map.values()))
    addr_to_rank = {addr: i for i, addr in enumerate(unique_addrs)}
    grid_size = int(np.ceil(np.sqrt(len(unique_addrs))))
    def addr_to_xy(addr):
        rank = addr_to_rank[addr]
        return divmod(rank, grid_size)

    flat_D_labels = D_labels.flatten()
    dst_label_to_rank = {label: i for i, label in enumerate(flat_D_labels)}
    total_d_elements = len(flat_D_labels)

    # build background
    grid = np.zeros((grid_size, grid_size, 3))
    gradient_values = np.linspace(0.3, 1.0, len(logical_order))
    for i, label in enumerate(logical_order):
        addr = addr_map[label]
        x, y = addr_to_xy(addr)
        intensity = gradient_values[i]
        if label.startswith("A"):
            grid[x,y,:] = [intensity,0,0]
        elif label.startswith("B"):
            grid[x,y,:] = [0,intensity,0]
        elif label.startswith("C"):
            grid[x,y,:] = [0,0,intensity]
        elif label.startswith("D"):
            grid[x,y,:] = [intensity,intensity,0]

    # plot
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(grid, origin='lower')
    for idx, (op_type, (dst_label, dst_addr), _) in enumerate(operations):
        x, y = addr_to_xy(dst_addr)
        angle = 2*np.pi * (dst_label_to_rank.get(dst_label,0)/total_d_elements)
        dx, dy = 0.3*np.cos(angle), 0.3*np.sin(angle)
        px, py = y+dx, x+dy
        if op_type == "+=C":
            marker = mpatches.RegularPolygon((px, py), numVertices=4, radius=0.2, orientation=np.pi/4, color='blue', alpha=0.7)
        else:
            marker = mpatches.RegularPolygon((px, py), numVertices=3, radius=0.2, orientation=np.pi/2, color='red', alpha=0.7)
        ax.add_patch(marker)
        ax.text(px, py, str(idx), ha='center', va='center', fontsize=5, color='white')
    ax.set_title(f"Parametric Memory Overlay Sample {sample_index}")
    plt.savefig(os.path.join(FULL_OVERLAY_DIR, f"param_overlay_sample{sample_index}.png"))
    plt.close()

# ======================
# CLI + random search
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-spectrum', action='store_true')
    parser.add_argument('--show-original', action='store_true')
    parser.add_argument('--show-overlay', action='store_true')
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()

    M,N,K = 5,5,5
    k_syll = ["ka", "ki", "ku", "ke", "ko"]
    s_syll = ["sa", "si", "su", "se", "so"]
    w_syll = ["wa", "wi", "wu", "we", "wo"]
    d_syll = ["da", "di", "du", "de", "do"]

    A_labels = label_matrix("A", (M,K), k_syll)
    B_labels = label_matrix("B", (K,N), s_syll)
    C_labels = label_matrix("C", (M,N), w_syll)
    D_labels = label_matrix("D", (M,N), d_syll)
    all_labels = np.concatenate([A_labels.flatten(), B_labels.flatten(), C_labels.flatten(), D_labels.flatten()])
    logical_order = list(all_labels)

    solver = ParametricMemoryOrderSolver(A_labels,B_labels,C_labels,D_labels)

    records = []
    for sample_idx in range(args.samples):
        mem_params = np.random.rand(solver.num_elements)
        op_params = np.random.rand(len(solver.operations))
        inter_pen, intra_pen, mat_ops, addr_map = solver.evaluate(mem_params, op_params)
        records.append({"sample": sample_idx, "inter_pen": inter_pen, "intra_pen": intra_pen})

        if args.show_overlay and sample_idx % 20 == 0:
            plot_overlay(addr_map, mat_ops, logical_order, D_labels, sample_idx)

    df = pd.DataFrame(records)
    print(df.sort_values(["inter_pen", "intra_pen"]).head(10))
    df.to_csv(os.path.join(OUTPUT_BASE_DIR, "random_search_results.csv"), index=False)
