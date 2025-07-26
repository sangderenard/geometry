import sympy

class ILPScheduler:
    def __init__(self, process_graph):
        self.G = process_graph.G
        self.operator_signatures = process_graph.role_schemas if hasattr(process_graph, 'role_schemas') else {}
        self.levels_asap = {}
        self.levels_alap = {}
        self.level_symbols = {}
        self.phase_symbols = {}
        self.ilp_constraints = []
        self.phase_constraints = []

    # ----------------------
    # Integer level scheduling
    # ----------------------
    def compute_asap_levels(self):
        levels = {}
        def dfs(n):
            if n in levels:
                return levels[n]
            preds = [p for p, _ in self.G.nodes[n]['parents']]
            lvl = 0 if not preds else 1 + max(dfs(p) for p in preds)
            levels[n] = lvl
            return lvl
        for nid in list(self.G.nodes):
            dfs(nid)
        self.levels_asap = levels
        return levels

    def compute_alap_levels(self):
        levels = {}
        def dfs(n):
            if n in levels:
                return levels[n]
            children = [c for c, _ in self.G.nodes[n]['children']]
            lvl = 0 if not children else 1 + max(dfs(c) for c in children)
            levels[n] = lvl
            return lvl
        for nid in list(self.G.nodes):
            dfs(nid)
        self.levels_alap = levels
        return levels

    # ----------------------
    # Integer symbolic constraints
    # ----------------------
    def build_level_symbols(self):
        self.level_symbols = {nid: sympy.Symbol(f"L_{nid}", integer=True) for nid in self.G.nodes}

    def build_ilp_constraints(self):
        """
        Build classic scheduling constraints: L_parent + 1 <= L_child for each edge.
        """
        constraints = []
        for nid in self.G.nodes:
            L_nid = self.level_symbols[nid]
            for child, _ in self.G.nodes[nid]['children']:
                L_child = self.level_symbols[child]
                constraints.append(L_nid + 1 <= L_child)
        self.ilp_constraints = constraints
        return constraints

    def solve_ilp_symbolically(self):
        solution = sympy.solve(self.ilp_constraints, list(self.level_symbols.values()), dict=True)
        return solution

    # ----------------------
    # Phase (rotor) harmonic system
    # ----------------------
    def build_phase_symbols(self):
        self.phase_symbols = {nid: sympy.Symbol(f"θ_{nid}", real=True) for nid in self.G.nodes}

    def build_harmonic_constraints(self, harmonics=3):
        """
        Build harmonic phase lock constraints:
        sum_m sin(m*(θ_c - θ_p - latency)) / m
        captures multi-harmonic coupling between parent and child.
        """
        constraints = []
        for nid in self.G.nodes:
            θ_n = self.phase_symbols[nid]
            for child, _ in self.G.nodes[nid]['children']:
                θ_c = self.phase_symbols[child]
                harmonic_sum = sum(sympy.sin(m * (θ_c - θ_n - 1)) / m for m in range(1, harmonics+1))
                constraints.append(harmonic_sum)
        self.phase_constraints = constraints
        return constraints

    def solve_symbolic_phase_system(self):
        solution = sympy.solve(self.phase_constraints, list(self.phase_symbols.values()), dict=True)
        return solution

    # ----------------------
    # Diagnostics
    # ----------------------
    def print_asap_levels(self):
        print("\n=== ASAP Levels ===")
        for nid, lvl in sorted(self.levels_asap.items(), key=lambda x: x[1]):
            print(f"Node {nid}: Level {lvl}")

    def print_alap_levels(self):
        print("\n=== ALAP Levels ===")
        for nid, lvl in sorted(self.levels_alap.items(), key=lambda x: x[1]):
            print(f"Node {nid}: Level {lvl}")

    def print_ilp_constraints(self):
        print("\n=== ILP Constraints ===")
        for c in self.ilp_constraints:
            print(c)

    def print_phase_constraints(self):
        print("\n=== Harmonic Phase Constraints ===")
        for c in self.phase_constraints:
            print(c)
