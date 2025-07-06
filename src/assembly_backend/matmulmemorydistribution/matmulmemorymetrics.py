import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
# Output directories for saving plots
OUTPUT_BASE_DIR = "output"
SPECTRUM_DIR = os.path.join(OUTPUT_BASE_DIR, "spectrum")
ORIGINAL_RANK_DIR = os.path.join(OUTPUT_BASE_DIR, "original_rank")
FULL_OVERLAY_DIR = os.path.join(OUTPUT_BASE_DIR, "full_overlay")
for _dir in (OUTPUT_BASE_DIR, SPECTRUM_DIR, ORIGINAL_RANK_DIR, FULL_OVERLAY_DIR):
    os.makedirs(_dir, exist_ok=True)

# --- Strategies unchanged ---
class MemoryMapStrategy:
    probabilistic = False
    def __init__(self, labels, element_size=8):
        self.labels = labels
        self.element_size = element_size

    def generate(self):
        raise NotImplementedError
class StripedMatrixMemoryMap(MemoryMapStrategy):
    probabilistic = False
    def __init__(self, A_labels, B_labels, C_labels, D_labels, n=4, element_size=8):
        # Flatten each matrix
        self.A_labels = A_labels.flatten()
        self.B_labels = B_labels.flatten()
        self.C_labels = C_labels.flatten()
        self.D_labels = D_labels.flatten()
        self.n = n
        self.element_size = element_size
        
        # Compose super labels list for uniform interface
        self.labels = np.concatenate([self.A_labels, self.B_labels, self.C_labels, self.D_labels])

    def generate(self):
        addr_map = {}
        idx = 0

        # Local cursors
        a_idx, b_idx, c_idx, d_idx = 0, 0, 0, 0
        total_A, total_B, total_C, total_D = len(self.A_labels), len(self.B_labels), len(self.C_labels), len(self.D_labels)

        while a_idx < total_A or b_idx < total_B or c_idx < total_C or d_idx < total_D:
            # A
            for _ in range(self.n):
                if a_idx < total_A:
                    addr_map[self.A_labels[a_idx]] = idx * self.element_size
                    idx += 1
                    a_idx += 1
            # B
            for _ in range(self.n):
                if b_idx < total_B:
                    addr_map[self.B_labels[b_idx]] = idx * self.element_size
                    idx += 1
                    b_idx += 1
            # C
            for _ in range(self.n):
                if c_idx < total_C:
                    addr_map[self.C_labels[c_idx]] = idx * self.element_size
                    idx += 1
                    c_idx += 1
            # D
            for _ in range(self.n):
                if d_idx < total_D:
                    addr_map[self.D_labels[d_idx]] = idx * self.element_size
                    idx += 1
                    d_idx += 1

        return addr_map

class ContiguousMemoryMap(MemoryMapStrategy):
    probabilistic = False
    def generate(self):
        return {label: idx * self.element_size for idx, label in enumerate(self.labels)}

class RandomStartDisparateMemoryMap(MemoryMapStrategy):
    probabilistic = True
    def __init__(self, labels, element_size=8, min_gap=512):
        super().__init__(labels, element_size)
        self.min_gap = min_gap
    def generate(self):
        #np.random.seed(42)
        starts = np.random.permutation(len(self.labels)) * self.min_gap
        return {label: starts[i] for i, label in enumerate(self.labels)}

class ShuffledContiguousMemoryMap(MemoryMapStrategy):
    probabilistic = True
    def generate(self):
        #np.random.seed(42)
        shuffled = np.random.permutation(len(self.labels))
        return {label: idx * self.element_size for idx, label in zip(shuffled, self.labels)}

class OperationOrderStrategy:
    probabilistic = False
    def sort(self, operations):
        raise NotImplementedError

class DefaultOutputScanOrder(OperationOrderStrategy):
    def sort(self, operations): return operations

class RandomOperationOrder(OperationOrderStrategy):
    probabilistic = True
    def sort(self, operations):
        #np.random.seed(42)
        shuffled = operations.copy()
        np.random.shuffle(shuffled)
        return shuffled

class AlphabeticalOperationOrder(OperationOrderStrategy):
    probabilistic = False
    def sort(self, operations):
        return sorted(operations, key=lambda op: f"{op[0]} {op[1][0]}")

# --- Label matrix generator ---
def label_matrix(prefix, shape, syllables):
    labels = np.empty(shape, dtype=object)
    idx = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            syll = syllables[idx % len(syllables)]
            labels[i, j] = f"{prefix}{syll}{i}{j}"
            idx += 1
    return labels

# --- Build operations ---
def build_flat_operations(A_labels, B_labels, C_labels, D_labels, M, N, K, get_addr):
    flat = []
    for i in range(M):
        for j in range(N):
            dst_label, dst_addr = D_labels[i, j], get_addr(D_labels[i, j])
            flat.append(("+=C", (dst_label, dst_addr), [(C_labels[i, j], get_addr(C_labels[i, j]))]))
            for k in range(K):
                a_label, a_addr = A_labels[i, k], get_addr(A_labels[i, k])
                b_label, b_addr = B_labels[k, j], get_addr(B_labels[k, j])
                flat.append(("+=A*B", (dst_label, dst_addr), [(a_label, a_addr), (b_label, b_addr)]))
    return flat

# --- Compute penalties with intra + inter
def compute_penalty_metrics_with_intra(operations):
    inter_penalties, inter_reversals, inter_running_sum = [], [], []
    intra_penalties, intra_reversals, intra_running_sum = [], [], []

    acc_inter = 0
    acc_intra = 0
    prev_endpoint = None
    last_direction_inter = None

    for op_type, (_, dst_addr), sources in operations:
        seq = [s[1] for s in sources] + [dst_addr]

        # inter hop
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

        # intra hops
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

    return (inter_running_sum, inter_penalties, inter_reversals,
            intra_running_sum, intra_penalties, intra_reversals)
def plot_memory_spectrum_vs_actual_layout(mem_map, logical_order, mem_strategy_name, order_strategy_name, sample_index):
    """
    logical_order: list of labels in contiguous logical order
    mem_map: dict of label -> physical memory address
    """

    all_addrs = [addr for addr in mem_map.values()]
    min_addr, max_addr = min(all_addrs), max(all_addrs)
    addr_range = max_addr - min_addr + 1

    grid_size = int(np.ceil(np.sqrt(addr_range)))
    grid = np.zeros((grid_size, grid_size, 3))  # RGB

    # Build gradient values: darkest to brightest across logical order
    gradient_values = np.linspace(0.2, 1.0, len(logical_order))

    def compute_addr_positions(mem_map):
        sorted_addrs = sorted(set(mem_map.values()))
        addr_to_rank = {addr: i for i, addr in enumerate(sorted_addrs)}
        return addr_to_rank

    addr_to_rank = compute_addr_positions(mem_map)

    def addr_to_xy(addr):
        rank = addr_to_rank[addr]
        return divmod(rank, grid_size)

    for i, label in enumerate(logical_order):
        addr = mem_map[label]
        x, y = addr_to_xy(addr)
        intensity = gradient_values[i]
        grid[x, y, :] = [intensity, intensity, intensity]

    # Plot
    plt.figure(figsize=(8,8))
    plt.imshow(grid, origin='lower')
    plt.title(f"Saturation Spectrum vs Actual Layout\n{mem_strategy_name} + {order_strategy_name} Sample {sample_index}")
    plt.axis('off')
    plt.savefig(os.path.join(SPECTRUM_DIR, f"spectrum_{mem_strategy_name}_{order_strategy_name}_sample{sample_index}.png"))
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
def plot_memory_original_rank_spectrum(mem_map, logical_order, mem_strategy_name, order_strategy_name, sample_index):
    """
    Shows each memory tile color-coded by its matrix (A, B, C, D),
    with the brightness purely from the original flat matrix logical rank.
    This reveals exactly how the original matrix order was scattered by the memory strategy.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Map memory addresses to ranks just for layout
    unique_addrs = sorted(set(mem_map.values()))
    addr_to_rank = {addr: i for i, addr in enumerate(unique_addrs)}
    num_addrs = len(unique_addrs)
    grid_size = int(np.ceil(np.sqrt(num_addrs)))

    def addr_to_xy(addr):
        rank = addr_to_rank[addr]
        return divmod(rank, grid_size)

    # Create gradient purely from original logical matrix flat order
    gradient_values = np.linspace(0.3, 1.0, len(logical_order))
    grid = np.zeros((grid_size, grid_size, 3))

    # Fill based on original matrix sequence
    for i, label in enumerate(logical_order):
        addr = mem_map[label]
        x, y = addr_to_xy(addr)
        intensity = gradient_values[i]

        if label.startswith("A"):
            grid[x,y,:] = [intensity, 0, 0]  # shades of red
        elif label.startswith("B"):
            grid[x,y,:] = [0, intensity, 0]  # shades of green
        elif label.startswith("C"):
            grid[x,y,:] = [0, 0, intensity]  # shades of blue
        elif label.startswith("D"):
            grid[x,y,:] = [intensity, intensity, 0]  # shades of yellow

    # Plot
    plt.figure(figsize=(8,8))
    plt.imshow(grid, origin='lower')
    plt.title(f"Original Matrix Rank → Memory Layout\n{mem_strategy_name} + {order_strategy_name} Sample {sample_index}")
    plt.axis('off')
    plt.savefig(os.path.join(ORIGINAL_RANK_DIR, f"original_rank_spectrum_{mem_strategy_name}_{order_strategy_name}_sample{sample_index}.png"))
    plt.close()
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
def plot_memory_full_overlay(mem_map, logical_order, operations, mem_strategy_name, order_strategy_name, sample_index, D_labels):
    """
    Now plots orbit offset by the rank of destination matrix elements only (D matrix).
    Shapes orbit around tiles based on the destination's original matrix flat rank,
    revealing how scatter order over the output is tied to memory.
    """
    # Compute memory layout
    unique_addrs = sorted(set(mem_map.values()))
    addr_to_rank = {addr: i for i, addr in enumerate(unique_addrs)}
    num_addrs = len(unique_addrs)
    grid_size = int(np.ceil(np.sqrt(num_addrs)))

    def addr_to_xy(addr):
        rank = addr_to_rank[addr]
        return divmod(rank, grid_size)

    # Compute rank only for D matrix (the output)
    flat_D_labels = D_labels.flatten()
    dst_label_to_rank = {label: i for i, label in enumerate(flat_D_labels)}
    total_d_elements = len(flat_D_labels)

    # Build background grid
    grid = np.zeros((grid_size, grid_size, 3))
    gradient_values = np.linspace(0.3, 1.0, len(logical_order))
    for i, label in enumerate(logical_order):
        addr = mem_map[label]
        x, y = addr_to_xy(addr)
        intensity = gradient_values[i]
        if label.startswith("A"):
            grid[x,y,:] = [intensity, 0, 0]
        elif label.startswith("B"):
            grid[x,y,:] = [0, intensity, 0]
        elif label.startswith("C"):
            grid[x,y,:] = [0, 0, intensity]
        elif label.startswith("D"):
            grid[x,y,:] = [intensity, intensity, 0]

    # Start plot
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(grid, origin='lower')
    ax.set_xlim(-0.5, grid_size-0.5)
    ax.set_ylim(-0.5, grid_size-0.5)
    ax.invert_yaxis()
    ax.axis('off')

    # Orbit markers: offset by rank order among D matrix
    for idx, (op_type, (dst_label, dst_addr), sources) in enumerate(operations):
        x, y = addr_to_xy(dst_addr)

        # Determine angle just from the destination matrix rank order
        d_rank = dst_label_to_rank.get(dst_label, 0)
        angle = 2 * np.pi * (d_rank / total_d_elements)
        dx = 0.3 * np.cos(angle)
        dy = 0.3 * np.sin(angle)
        plot_x, plot_y = y + dx, x + dy

        # Draw
        if op_type == "+=C":
            marker = mpatches.RegularPolygon((plot_x, plot_y), numVertices=4, radius=0.2, 
                                             orientation=np.pi/4, color='blue', alpha=0.7)
        elif op_type == "+=A*B":
            marker = mpatches.RegularPolygon((plot_x, plot_y), numVertices=3, radius=0.2, 
                                             orientation=np.pi/2, color='red', alpha=0.7)
        ax.add_patch(marker)
        ax.text(plot_x, plot_y, str(idx), ha='center', va='center', fontsize=5, color='white', weight='bold')

    ax.set_title(f"Output Matrix Rank → Orbit\n{mem_strategy_name} + {order_strategy_name} Sample {sample_index}")
    plt.savefig(os.path.join(FULL_OVERLAY_DIR, f"full_overlay_{mem_strategy_name}_{order_strategy_name}_sample{sample_index}.png"))
    plt.close()



def main():
    M, N, K = 5, 5, 5
    k_syll = ["ka", "ki", "ku", "ke", "ko"]
    s_syll = ["sa", "si", "su", "se", "so"]
    w_syll = ["wa", "wi", "wu", "we", "wo"]
    d_syll = ["da", "di", "du", "de", "do"]

    A_labels = label_matrix("A", (M, K), k_syll)
    B_labels = label_matrix("B", (K, N), s_syll)
    C_labels = label_matrix("C", (M, N), w_syll)
    D_labels = label_matrix("D", (M, N), d_syll)
    all_labels = np.concatenate([A_labels.flatten(), B_labels.flatten(), C_labels.flatten(), D_labels.flatten()])

    mem_strategies = [
        ContiguousMemoryMap(all_labels, element_size=1),
        ShuffledContiguousMemoryMap(all_labels, element_size=1),
        StripedMatrixMemoryMap(A_labels, B_labels, C_labels, D_labels, n=4, element_size=1),
    ]
    order_strategies = [
        DefaultOutputScanOrder(),
        RandomOperationOrder(),
        AlphabeticalOperationOrder(),
    ]

    NUM_SAMPLES = 5
    records = []

    for mem_strat in mem_strategies:
        for order_strat in order_strategies:
            samples = NUM_SAMPLES if (mem_strat.probabilistic or order_strat.probabilistic) else 1
            for s in range(samples):
                addr_map = mem_strat.generate()
                logical_order = list(all_labels)  # in the order they were originally flattened
                plot_memory_spectrum_vs_actual_layout(
                    addr_map, logical_order, 
                    mem_strat.__class__.__name__, order_strat.__class__.__name__, s+1
                )
                plot_memory_original_rank_spectrum(
                    addr_map, logical_order,
                    mem_strat.__class__.__name__, order_strat.__class__.__name__, s+1
                )

                get_addr = lambda label, m=addr_map: m[label]
                ops = build_flat_operations(A_labels, B_labels, C_labels, D_labels, M, N, K, get_addr)
                sorted_ops = order_strat.sort(ops)
                plot_memory_full_overlay(
                    addr_map, logical_order, sorted_ops,
                    mem_strat.__class__.__name__, order_strat.__class__.__name__, s+1, D_labels
                )

                (inter_running_sum, inter_penalties, inter_reversals,
                 intra_running_sum, intra_penalties, intra_reversals) = compute_penalty_metrics_with_intra(sorted_ops)

                intra_idx = 0
                for idx, (op_type, (dst_label, dst_addr), sources) in enumerate(sorted_ops):
                    if op_type == "+=C":
                        src_label, src_addr = sources[0]
                        op_str = f"{dst_label}@{dst_addr} += {src_label}@{src_addr}"
                    else:
                        (a_label, a_addr), (b_label, b_addr) = sources
                        op_str = f"{dst_label}@{dst_addr} += ({a_label}@{a_addr} * {b_label}@{b_addr})"

                    inter_pen = inter_penalties[idx-1] if idx > 0 and (idx-1) < len(inter_penalties) else 0
                    inter_run = inter_running_sum[idx-1] if idx > 0 and (idx-1) < len(inter_running_sum) else 0
                    inter_rev = inter_reversals[idx-1] if idx > 0 and (idx-1) < len(inter_reversals) else 0

                    intra_sum, intra_max, intra_rev_count = 0, 0, 0
                    for _ in range(len(sources)):
                        if intra_idx < len(intra_penalties):
                            p = intra_penalties[intra_idx]
                            intra_sum += p
                            intra_max = max(intra_max, p)
                            if intra_idx < len(intra_reversals) and intra_reversals[intra_idx] > 0:
                                intra_rev_count += 1
                            intra_idx += 1
                    if intra_idx < len(intra_penalties):
                        p = intra_penalties[intra_idx]
                        intra_sum += p
                        intra_max = max(intra_max, p)
                        if intra_idx < len(intra_reversals) and intra_reversals[intra_idx] > 0:
                            intra_rev_count += 1
                        intra_idx += 1

                    # --- new: memory span
                    addrs = [dst_addr] + [s[1] for s in sources]
                    mem_span = max(addrs) - min(addrs)

                    records.append({
                        "idx": idx,
                        "operation": op_str,
                        "inter_penalty": inter_pen,
                        "inter_running": inter_run,
                        "inter_reversal": inter_rev,
                        "intra_sum": intra_sum,
                        "intra_max": intra_max,
                        "intra_rev_count": intra_rev_count,
                        "memory_span": mem_span,
                        "memory_strategy": mem_strat.__class__.__name__,
                        "order_strategy": order_strat.__class__.__name__,
                        "sample": s+1 if samples > 1 else 1
                    })

    # --- Build DataFrame
    df = pd.DataFrame(records)
    print(df.head(10))

    # Example group by
    summary = df.groupby(["memory_strategy", "order_strategy"]).agg({
        "inter_penalty":"sum",
        "intra_sum":"sum",
        "memory_span":"mean"
    }).reset_index()
    print("\n--- Summary by strategy ---")
    print(summary)

if __name__ == "__main__":
    main()
