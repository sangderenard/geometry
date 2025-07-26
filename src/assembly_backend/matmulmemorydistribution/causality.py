def generic_tree_to_dag(root, get_children):
    class ProcDAGNode:
        def __init__(self, op_type, args, node_id):
            self.op_type = op_type
            self.args = args
            self.node_id = node_id
            self.deps = []
        def __repr__(self):
            return f"<{self.op_type}#{self.node_id} deps={self.deps} args={self.args}>"

    class ProcDAG:
        def __init__(self):
            self.nodes = []
            self.edges = []
            self.next_id = 0
        def add_node(self, op_type, args):
            node = ProcDAGNode(op_type, args, self.next_id)
            self.nodes.append(node)
            self.next_id += 1
            return node
        def add_edge(self, src, dst):
            self.edges.append((src.node_id, dst.node_id))
            dst.deps.append(src.node_id)
        def __repr__(self):
            return f"ProcDAG(nodes={self.nodes}, edges={self.edges})"

    dag = ProcDAG()
    node_map = {}

    def visit(node, parent=None):
        if isinstance(node, list):
            last = None
            for child in node:
                n = visit(child, parent)
                if last and n:
                    dag.add_edge(last, n)
                last = n
            return last
        else:
            op_type = type(node).__name__
            args = []  # optional: extract fields if desired
            dag_node = dag.add_node(op_type, args)
            node_map[id(node)] = dag_node
            children = get_children(node)
            last_child = None
            for child in children:
                cnode = visit(child, dag_node)
                if last_child and cnode:
                    dag.add_edge(last_child, cnode)
                last_child = cnode
            if last_child:
                dag.add_edge(last_child, dag_node)
            return dag_node

    visit(root)
    return dag

# ---- Usage Example ----

# For Python AST:
def ast_children(node):
    if isinstance(node, ast.Module):
        return node.body
    elif hasattr(node, 'body') and isinstance(node.body, list):
        return node.body
    # Add more rules or return [] as fallback
    return []

# For OOP objects:
def obj_children(obj):
    # If object has an explicit 'children' attribute or method, use it.
    if hasattr(obj, 'children'):
        return obj.children
    # If it has a __dict__ with ordered fields:
    if hasattr(obj, '__dict__'):
        # Optionally filter to only 'node-like' fields
        # For complete indifference, just get all object attributes that are not builtins
        vals = [v for k, v in obj.__dict__.items() if not k.startswith('__')]
        # Flatten any lists of objects
        children = []
        for v in vals:
            if isinstance(v, list):
                children.extend(v)
            else:
                children.append(v)
        return children
    return []

# --- Now you can ingest *anything* as long as you provide the right get_children function! ---

if __name__ == "__main__":
    # Example for Python AST
    import ast
    src = "a=1\nb=2\nc=a+b"
    tree = ast.parse(src)
    dag = generic_tree_to_dag(tree, ast_children)
    for node in dag.nodes:
        print(node)
    print("Edges:", dag.edges)

    # Example for user-defined OOP trees
    class Foo:
        def __init__(self, val, kids=None):
            self.val = val
            self.kids = kids or []
        @property
        def children(self): return self.kids

    t = Foo("root", [Foo("left"), Foo("right", [Foo("leaf")])])
    dag2 = generic_tree_to_dag(t, obj_children)
    for node in dag2.nodes:
        print(node)
    print("Edges:", dag2.edges)
