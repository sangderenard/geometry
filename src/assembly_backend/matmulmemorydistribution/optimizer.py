#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

# ======================
# Output directories
# ======================
OUTPUT_BASE_DIR = "output"
FULL_OVERLAY_DIR = os.path.join(OUTPUT_BASE_DIR, "full_overlay")
os.makedirs(FULL_OVERLAY_DIR, exist_ok=True)

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
# Solver class
# ======================
class TorchParametricSolver(nn.Module):
    def __init__(self, A_labels, B_labels, C_labels, D_labels):
        super().__init__()
        self.A_labels = A_labels.flatten()
        self.B_labels = B_labels.flatten()
        self.C_labels = C_labels.flatten()
        self.D_labels = D_labels.flatten()
        self.all_labels = np.concatenate([self.A_labels, self.B_labels, self.C_labels, self.D_labels])
        self.num_elements = len(self.all_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        self.operations = build_flat_operations(A_labels, B_labels, C_labels, D_labels, A_labels.shape[0], D_labels.shape[1], A_labels.shape[1])
        
        self.mem_params = nn.Parameter(torch.randn(self.num_elements))
        self.op_params = nn.Parameter(torch.randn(len(self.operations)))

    def forward(self):
        op_sort_idx = torch.argsort(self.op_params)

        addr_values = self.mem_params
        addr_map = {self.all_labels[i]: addr_values[i] for i in range(self.num_elements)}

        ordered_ops = [self.operations[i] for i in op_sort_idx.detach().cpu().numpy()]
        materialized_ops = [(op_type, (dst_label, addr_map[dst_label]), [(src_label, addr_map[src_label]) for src_label,_ in sources])
                            for op_type, (dst_label,_), sources in ordered_ops]

        inter_pen, intra_pen = self.compute_penalties(materialized_ops)
        return inter_pen, intra_pen, materialized_ops, addr_map


    def compute_penalties(self, operations,
                        intra_forward_exp=1.0, intra_backward_exp=2.0,
                        inter_forward_exp=1.0, inter_backward_exp=2.0):
        inter_forward_hops = []
        inter_backward_hops = []
        intra_forward_hops = []
        intra_backward_hops = []

        prev_endpoint = None
        for op_type, (_, dst_addr), sources in operations:
            seq = [s[1] for s in sources] + [dst_addr]

            # inter-op hop
            if prev_endpoint is not None:
                hop = seq[0] - prev_endpoint
                inter_forward_hops.append(torch.relu(hop))
                inter_backward_hops.append(torch.relu(-hop))
            prev_endpoint = seq[-1]

            # intra-op hops
            for i in range(len(seq)-1):
                hop = seq[i+1] - seq[i]
                intra_forward_hops.append(torch.relu(hop))
                intra_backward_hops.append(torch.relu(-hop))

        # Convert to tensors
        if inter_forward_hops:
            inter_forward_hops = torch.stack(inter_forward_hops)
            inter_backward_hops = torch.stack(inter_backward_hops)
        else:
            inter_forward_hops = torch.zeros(1, device=self.mem_params.device)
            inter_backward_hops = torch.zeros(1, device=self.mem_params.device)

        if intra_forward_hops:
            intra_forward_hops = torch.stack(intra_forward_hops)
            intra_backward_hops = torch.stack(intra_backward_hops)
        else:
            intra_forward_hops = torch.zeros(1, device=self.mem_params.device)
            intra_backward_hops = torch.zeros(1, device=self.mem_params.device)

        # Apply exponents and sum up
        inter_penalties = torch.sum(inter_forward_hops ** inter_forward_exp) + \
                        torch.sum(inter_backward_hops ** inter_backward_exp)
        intra_penalties = torch.sum(intra_forward_hops ** intra_forward_exp) + \
                        torch.sum(intra_backward_hops ** intra_backward_exp)

        return inter_penalties, intra_penalties



# ======================
# Overlay plot
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
    ax.set_title(f"PyTorch Optim Overlay {sample_index}")
    plt.savefig(os.path.join(FULL_OVERLAY_DIR, f"torch_overlay_{sample_index}.png"))
    plt.close()

# ======================
# Main run
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--show-overlay', action='store_true')
    parser.add_argument('--lr', type=float, default=0.1)
    args = parser.parse_args()

    M,N,K = 3,3,3
    k_syll = ["ka", "ki", "ku", "ke", "ko"]
    s_syll = ["sa", "si", "su", "se", "so"]
    w_syll = ["wa", "wi", "wu", "we", "wo"]
    d_syll = ["da", "di", "du", "de", "do"]

    A_labels = label_matrix("A", (M,K), k_syll)
    B_labels = label_matrix("B", (K,N), s_syll)
    C_labels = label_matrix("C", (M,N), w_syll)
    D_labels = label_matrix("D", (M,N), d_syll)
    logical_order = list(np.concatenate([A_labels.flatten(), B_labels.flatten(), C_labels.flatten(), D_labels.flatten()]))

    model = TorchParametricSolver(A_labels,B_labels,C_labels,D_labels)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    best_config = None
    records = []

    for step in range(args.steps):
        optimizer.zero_grad()
        inter_pen, intra_pen, mat_ops, addr_map = model()
        loss = inter_pen + 0.5 * intra_pen  # weighted composite
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        records.append({"step": step, "inter_pen": inter_pen, "intra_pen": intra_pen, "total_loss": loss_val})

        if loss_val < best_loss:
            best_loss = loss_val
            best_config = (mat_ops, addr_map)
            if args.show_overlay:
                plot_overlay(addr_map, mat_ops, logical_order, D_labels, step)

        if step % 50 == 0:
            print(f"Step {step:4d}: Loss={loss_val:.2f} (Inter={inter_pen}, Intra={intra_pen})")

    # Write out best stats
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_BASE_DIR, "torch_optimizer_progress.csv"), index=False)
    print("\nBest total loss:", best_loss)
