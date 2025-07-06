import torch
import torch.nn as nn
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

    def from_logits(self, mem_logits_batch):
        batch_size = mem_logits_batch.shape[0]
        device = mem_logits_batch.device

        addr_maps_batch = []
        mat_ops_batch = []
        inter_pen_batch = []
        intra_pen_batch = []

        for b in range(batch_size):
            logits = mem_logits_batch[b]
            sort_idx = torch.argsort(logits)
            addr_map = {self.all_labels[i]: sort_idx[i] for i in range(self.num_elements)}

            unique_addrs_sorted = sorted(float(addr_map[self.all_labels[i]].item() if hasattr(addr_map[self.all_labels[i]], 'item') else addr_map[self.all_labels[i]]) for i in range(self.num_elements))
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
                        mem_rank = int(float(addr_map[label].item()) if hasattr(addr_map[label], 'item') else addr_map[label])
                        row = mem_rank // lane_width
                        col = mem_rank % lane_width
                        memory_space[0, :, row, col] = feat[:, i, j]

            mem_conv = F.relu(self.memory_space_conv(memory_space))
            mem_pooled = self.memory_space_pool(mem_conv)
            mem_flat = mem_pooled.view(1, -1)
            op_logits = self.memory_space_proj(mem_flat).squeeze(0)

            op_sort_idx = torch.argsort(op_logits)
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

        return torch.stack(inter_pen_batch), torch.stack(intra_pen_batch), mat_ops_batch, addr_maps_batch

    def build_per_matrix_inputs(self, addr_map, addr_to_rank, device):
        per_matrix_inputs = []
        for mat in self.operand_label_matrices:
            H, W = mat.shape
            inp = torch.zeros(1, H, W, device=device)
            for i in range(H):
                for j in range(W):
                    label = mat[i, j]
                    if label in addr_map:
                        addr_val = addr_map[label]
                        inp[0, i, j] = float(addr_val.item()) if hasattr(addr_val, 'item') else float(addr_val)
                    else:
                        inp[0, i, j] = 0.0
            per_matrix_inputs.append(inp)
        return per_matrix_inputs

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
            if prev_endpoint is not None:
                hop = seq[0] - prev_endpoint
                hop = hop.float() if hasattr(hop, 'float') else torch.tensor(float(hop), device=self.per_matrix_convs[0].weight.device)
                inter_forward_hops.append(torch.relu(hop))
                inter_backward_hops.append(torch.relu(-hop))
            prev_endpoint = seq[-1]
            for i in range(len(seq) - 1):
                a = seq[i]
                b = seq[i + 1]
                hop = b - a
                hop = hop.float() if hasattr(hop, 'float') else torch.tensor(float(hop), device=self.per_matrix_convs[0].weight.device)
                intra_forward_hops.append(torch.relu(hop))
                intra_backward_hops.append(torch.relu(-hop))

        device = self.per_matrix_convs[0].weight.device
        inter_forward_hops = torch.stack(inter_forward_hops).requires_grad_(True) if inter_forward_hops else torch.zeros(1, device=device, requires_grad=True)
        inter_backward_hops = torch.stack(inter_backward_hops).requires_grad_(True) if inter_backward_hops else torch.zeros(1, device=device, requires_grad=True)
        intra_forward_hops = torch.stack(intra_forward_hops).requires_grad_(True) if intra_forward_hops else torch.zeros(1, device=device, requires_grad=True)
        intra_backward_hops = torch.stack(intra_backward_hops).requires_grad_(True) if intra_backward_hops else torch.zeros(1, device=device, requires_grad=True)

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
