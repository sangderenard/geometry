import sympy
from sympy import symbols, Sum, IndexedBase, Idx, Lambda
import networkx as nx
from colorama import Fore, Style, init
from solver_types import Node, NodeSet, Operation, Edge

init(autoreset=True)
colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]

# -------------------------------------------------------
# Global graph + level maps for process graph extraction
# -------------------------------------------------------
G = nx.DiGraph()
levels = {}
node_map = {}

# ----------------------------
# Build symbolic dependency graph
# ----------------------------
def full_recombinatorics(expr):
    expr = expr.doit()
    expr = sympy.expand(expr, power_exp=True, log=True, multinomial=True, complex=True, trig=True)
    expr = sympy.expand_mul(expr)
    expr = sympy.expand_power_exp(expr)
    expr = sympy.expand_log(expr)
    expr = sympy.trigsimp(expr)
    expr = sympy.cancel(expr)
    try:
        expr = sympy.apart(expr)
    except:
        pass
    expr = sympy.factor(expr)
    expr = sympy.simplify(expr)
    return expr

def build_graph(node, parent=None):
    node_id = id(node)
    if node_id not in G:
        G.add_node(node_id,
                   label=str(node),
                   type=type(node).__name__,
                   expr_obj=node,
                   parents=set(),
                   children=set())
        node_map[node_id] = node
    if parent is not None:
        G.add_edge(node_id, parent)
        G.nodes[node_id]['children'].add(parent)
        G.nodes[parent]['parents'].add(node_id)
    for arg in getattr(node, 'args', []):
        build_graph(arg, node_id)

def compute_level(node):
    preds = list(G.predecessors(node))
    if not preds:
        levels[node] = 0
    else:
        levels[node] = 1 + max(compute_level(p) for p in preds)
    return levels[node]
def parse_process_graph_for_nodeset_requirements(graph):
    operation_nodes = graph['nodes']
    operation_levels = graph['levels']

    # Build reverse mapping from op_id -> level
    op_id_to_level = {}
    for lvl, ids in operation_levels.items():
        for nid in ids:
            op_id_to_level[nid] = lvl
    
    input_nodes = {} # how many original inputs are required
    intermediary_nodes = {} # how many implicit intermediary nodes are required
    output_nodes = {} # how many original outputs are required
    operations = {}
    
    for operation_node in operation_nodes.items():
        op_id, op_data = operation_node
        op_level = op_id_to_level[op_id]
        operation = Operation(op_data['parents'], len(op_data['parents']),
                              op_data['children'], len(op_data['children']),
                              op_id, op_data['type'],op_level, 0.0)
        operations[op_id] = operation

        for parent in op_data['parents']:
            grandparents = list(G.predecessors(parent))
            if grandparents:
                # This is an intermediary node
                if op_level not in intermediary_nodes:
                    intermediary_nodes[op_level] = {}
                if op_data['type'] not in intermediary_nodes[op_level]:
                    intermediary_nodes[op_level][op_data['type']] = {}
                if op_id not in intermediary_nodes[op_level][op_data['type']]:
                    intermediary_nodes[op_level][op_data['type']][op_id] = set()
                intermediary_nodes[op_level][op_data['type']][op_id].update(grandparents)
            else:
                # This is an input node
                if op_level not in input_nodes:
                    input_nodes[op_level] = {}
                if op_data['type'] not in input_nodes[op_level]:
                    input_nodes[op_level][op_data['type']] = {}
                if op_id not in input_nodes[op_level][op_data['type']]:
                    input_nodes[op_level][op_data['type']][op_id] = set()
                input_nodes[op_level][op_data['type']][op_id].add(parent)

                
        for child in op_data['children']:
            grandchildren = list(G.successors(child))
            if grandchildren:
                # This is an intermediary node
                if op_level not in intermediary_nodes:
                    intermediary_nodes[op_level] = {}
                if op_data['type'] not in intermediary_nodes[op_level]:
                    intermediary_nodes[op_level][op_data['type']] = {}
                if op_id not in intermediary_nodes[op_level][op_data['type']]:
                    intermediary_nodes[op_level][op_data['type']][op_id] = set()
                intermediary_nodes[op_level][op_data['type']][op_id].update(grandchildren)
            else:
                # This is an output node
                if op_level not in output_nodes:
                    output_nodes[op_level] = {}
                if op_data['type'] not in output_nodes[op_level]:
                    output_nodes[op_level][op_data['type']] = {}
                if op_id not in output_nodes[op_level][op_data['type']]:
                    output_nodes[op_level][op_data['type']][op_id] = set()
                output_nodes[op_level][op_data['type']][op_id].add(child)

    

    return input_nodes, intermediary_nodes, output_nodes, operations


def extract_full_process_graph():
    """
    Returns a structured dict representing the entire symbolic process graph,
    including node meta-data and level organization.
    """
    nodes = {}
    for nid in G.nodes:
        data = G.nodes[nid]
        nodes[nid] = {
            'type': data['type'],
            'label': data['label'],
            'expr_obj': data['expr_obj'],
            'parents': list(data['parents']),
            'children': list(data['children']),
            'level': levels[nid],
        }
    levels_map = {}
    for nid, lvl in levels.items():
        levels_map.setdefault(lvl, []).append(nid)
    return {
        'nodes': nodes,
        'levels': levels_map
    }

# ----------------------------
# Dry execution on actual data
# ----------------------------
def dry_execute(data_sources, operator_funcs=None):
    """
    Executes the built graph G on actual data.
    data_sources: dict mapping node labels to numeric values or arrays.
    operator_funcs: optional dict mapping sympy node type names to callables.
    """
    import numpy as np
    if operator_funcs is None:
        operator_funcs = {}
    results = {}

    # default operator implementations
    def op_add(*args):
        return sum(args)
    def op_mul(*args):
        res = args[0]
        for a in args[1:]:
            res = res * a
        return res
    def op_indexed(base, *idx):
        return base[tuple(int(i) if hasattr(i, 'is_Integer') else i for i in idx)]
    def op_sum(value, limits):
        # naive sum wrapper: ignores limits, assumes value aggregated
        return value

    defaults = {'Add': op_add, 'Mul': op_mul, 'Indexed': op_indexed, 'Sum': op_sum}

    # execute by increasing levels
    for lvl in sorted(set(levels.values())):
        for nid in (n for n, l in levels.items() if l == lvl):
            node = G.nodes[nid]
            expr_obj = node['expr_obj']
            node_type = type(expr_obj).__name__
            deps = list(node['parents'])
            if not deps:
                results[nid] = data_sources.get(node['label'], expr_obj)
            else:
                args = [results[d] for d in deps]
                func = operator_funcs.get(node_type)
                if func is None:
                    func = defaults.get(node_type)

                if func is None:
                    raise TypeError(f"No operator function for node type: {node_type} at level {lvl}, node {node['label']}, args = {args}")

                if node_type == 'Sum':
                    results[nid] = func(args[0], expr_obj.limits)
                else:
                    results[nid] = func(*args)

    # return root
    root = max(levels, key=lambda n: levels[n])
    return results[root]
def lateral_graph_merge(graphs_with_meta):
    """
    Merges multiple graphs into a single big graph G, offsetting IDs.
    graphs_with_meta: list of tuples (G_local, levels_local, node_map_local)
    """
    global G, levels, node_map

    current_max_id = max(G.nodes) + 1 if G.nodes else 0

    for G_local, levels_local, node_map_local in graphs_with_meta:
        id_mapping = {}
        for old_id in G_local.nodes:
            new_id = current_max_id
            current_max_id += 1
            id_mapping[old_id] = new_id

            data = G_local.nodes[old_id]
            G.add_node(new_id,
                       label=data['label'],
                       type=data['type'],
                       expr_obj=data['expr_obj'],
                       parents=set(),
                       children=set())
            node_map[new_id] = node_map_local[old_id]
            levels[new_id] = levels_local[old_id]

        # Reconnect edges
        for u, v in G_local.edges:
            u_new = id_mapping[u]
            v_new = id_mapping[v]
            G.add_edge(u_new, v_new)
            G.nodes[u_new]['children'].add(v_new)
            G.nodes[v_new]['parents'].add(u_new)
def group_by_level_and_type():
    grouping = {}
    for node_id in G.nodes:
        lvl = levels[node_id]
        op_type = G.nodes[node_id]['type']
        grouping.setdefault(lvl, {}).setdefault(op_type, []).append(node_id)
    return grouping
def build_lateral_graph_across_domain(M, N, expression_builder):
    """
    Builds and merges multiple small symbolic graphs across a domain,
    like all (i,j), merging them laterally into the global graph G.
    
    - M, N: domain extents
    - expression_builder: function (i,j) -> symbolic expression
    """
    graphs_with_meta = []

    for i in range(M):
        for j in range(N):
            # Build a local graph for this (i,j)
            local_G = nx.DiGraph()
            local_levels = {}
            local_node_map = {}

            # Build local graph building function
            def local_build_graph(node, parent=None):
                node_id = id(node)
                if node_id not in local_G:
                    local_G.add_node(node_id,
                                     label=str(node),
                                     type=type(node).__name__,
                                     expr_obj=node,
                                     parents=set(),
                                     children=set())
                    local_node_map[node_id] = node
                if parent is not None:
                    local_G.add_edge(node_id, parent)
                    local_G.nodes[node_id]['children'].add(parent)
                    local_G.nodes[parent]['parents'].add(node_id)
                for arg in getattr(node, 'args', []):
                    local_build_graph(arg, node_id)

            # Build symbolic expression for this (i,j)
            expr = expression_builder(i, j)
            expr = full_recombinatorics(expr)
            local_build_graph(expr)

            def compute_local_level(node):
                preds = list(local_G.predecessors(node))
                if not preds:
                    local_levels[node] = 0
                else:
                    local_levels[node] = 1 + max(compute_local_level(p) for p in preds)
                return local_levels[node]

            for node in local_G.nodes:
                compute_local_level(node)

            graphs_with_meta.append((local_G, local_levels, local_node_map))

    # Now merge all local graphs laterally into global G
    lateral_graph_merge(graphs_with_meta)

# ----------------------------
# Demo execution (compartmentalized to main)
# ----------------------------
def main():
    global G, levels, node_map
    G.clear()
    levels.clear()
    node_map.clear()

    M_val = 2
    N_val = 2
    K_val = 3
    M, N, K = symbols('M N K', integer=True)
    i, j, k = Idx('i', M), Idx('j', N), Idx('k', K)
    A = IndexedBase('A')
    B = IndexedBase('B')
    C = IndexedBase('C')

    build_lateral_graph_across_domain(
        M_val, N_val,
        lambda ii, jj: Sum(A[ii,k]*B[k,jj], (k,0,K_val-1)) + C[ii,jj]
    )

    # ----------------------------
    # 1. Symbolic expression
    # ----------------------------
    D = IndexedBase('D')



    for node in G.nodes:
        compute_level(node)

    # ----------------------------
    # 2. Organize by level -> op type -> nodes (parallel bands)
    # ----------------------------
    serialization = {}
    for node, lvl in levels.items():
        op_type = G.nodes[node]['type']
        label = G.nodes[node]['label']
        serialization.setdefault(lvl, {}).setdefault(op_type, []).append(label)

    print("\n=== Serialized execution plan (parallel bands) ===")
    for lvl in sorted(serialization.keys()):
        color = colors[lvl % len(colors)]
        print(f"{color}Level {lvl}:{Style.RESET_ALL}")
        for op_type, nodes in serialization[lvl].items():
            print(f"{color}  {op_type}:{Style.RESET_ALL}")
            for n in nodes:
                print(f"{color}    - {n}{Style.RESET_ALL}")

    # ----------------------------
    # 3. Dependency striped view
    # ----------------------------
    history_fabric = {}
    for nid in G.nodes:
        data = G.nodes[nid]
        history_fabric[data['expr_obj']] = {
            'level': levels[nid],
            'type': data['type'],
            'label': data['label'],
            'parents': list(data['parents']),
            'children': list(data['children']),
        }

    print("\n=== Dependency fabric with color-coded parent inclusions ===")
    for i, (expr_obj, record) in enumerate(history_fabric.items()):
        text = record['label']
        for parent_id in record['parents']:
            parent_label = G.nodes[parent_id]['label']
            parent_level = levels[parent_id]
            color = colors[parent_level % len(colors)]
            if parent_label in text:
                text = text.replace(parent_label, f"{color}[{parent_label}]{Style.RESET_ALL}")
        print(f"Expr: {text}")

    # ----------------------------
    # 4. Node detailed relationship data
    # ----------------------------
    print("\n=== Node relationship data ===")
    for nid in sorted(G.nodes, key=lambda n: levels[n]):
        data = G.nodes[nid]
        parents = [G.nodes[p]['label'] for p in data['parents']]
        children = [G.nodes[c]['label'] for c in data['children']]
        print(f"Node: {data['label']}")
        print(f"  Level: {levels[nid]}, Type: {data['type']}")
        print(f"  Parents: {parents}")
        print(f"  Children: {children}")

    # ----------------------------
    # 5. Full process graph extract
    # ----------------------------
    process_graph = extract_full_process_graph()
    print("\n=== Extracted process graph (summary) ===")
    print(f"Total nodes: {len(process_graph['nodes'])}")
    print(f"Levels: {list(process_graph['levels'].keys())}")

    # ----------------------------
    # 6. Dry execution example
    # ----------------------------
    # Define dummy data sources matching the expected node labels
    data_sources = {
        'A[i,k]': 2,
        'B[k,j]': 3,
        'C[i,j]': 5
    }
    try:
        result = dry_execute(data_sources)

        print(f"\n=== Dry execution result ===")
        print(result)
    except Exception as e:
        print(f"Error during dry execution: {e}")


    # ----------------------------
    # 7. Parse process graph for nodeset requirements
    # ----------------------------
    input_nodes, intermediary_nodes, output_nodes, operations = parse_process_graph_for_nodeset_requirements(process_graph)

    print("\n=== Parsed process graph for nodeset requirements ===")
    print("\n--- Input nodes ---")
    for lvl in sorted(input_nodes.keys()):
        print(f"Level {lvl}:")
        for op_type, ops in input_nodes[lvl].items():
            print(f"  {op_type}: {ops}")

    print("\n--- Intermediary nodes ---")
    for lvl in sorted(intermediary_nodes.keys()):
        print(f"Level {lvl}:")
        for op_type, ops in intermediary_nodes[lvl].items():
            print(f"  {op_type}: {ops}")

    print("\n--- Output nodes ---")
    for lvl in sorted(output_nodes.keys()):
        print(f"Level {lvl}:")
        for op_type, ops in output_nodes[lvl].items():
            print(f"  {op_type}: {ops}")

    print("\n--- Operations ---")
    for op in Operation.default_sort(operations.values()):
        print(f"Operation {op.string}: {op}")

    grouping = group_by_level_and_type()

    print("\n=== Logical groupings by level and type ===")
    for lvl in sorted(grouping.keys()):
        print(f"Level {lvl}:")
        for op_type, nodes in grouping[lvl].items():
            print(f"  {op_type}: {[G.nodes[n]['label'] for n in nodes]}")


if __name__ == "__main__":
    main()
