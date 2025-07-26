class NodeStencilMatcher:
    """
    Future-proof stencil matcher for nodes.
    Will ultimately handle learning, symbolic overlays, live feedback, and more.
    For now: matches each node against all stencils and returns list of matches.
    """

    def __init__(self):
        self.stencils = []

    def register_stencil(self, stencil, name=None):
        """
        Add a stencil. Stencil is a function(node) -> bool or dict.
        Name is optional and used for reporting/labeling.
        """
        self.stencils.append((name, stencil))

    def match(self, node):
        """
        Returns list of (name, stencil) for which stencil(node) returns True (or non-None).
        """
        matches = []
        for name, stencil in self.stencils:
            try:
                result = stencil(node)
                if result:  # Can be bool or richer dict with weights, etc.
                    matches.append((name, result))
            except Exception as e:
                # Optionally log error, continue
                print(f"[StencilMatcher] Error in stencil {name}: {e}")
        return matches

    def best_match(self, node):
        """
        Optionally: pick 'best' match by some policy. For now, just returns the first match.
        """
        matches = self.match(node)
        return matches[0] if matches else None

# Usage example (with AST nodes)
import ast

def is_assign(node):  # Minimal example stencil
    return isinstance(node, ast.Assign)

def is_return(node):
    return isinstance(node, ast.Return)

matcher = NodeStencilMatcher()
matcher.register_stencil(is_assign, name="assign")
matcher.register_stencil(is_return, name="return")

tree = ast.parse("""
def foo(x):
    y = x + 1
    return y
""")
for node in ast.walk(tree):
    match = matcher.match(node)
    if match:
        print(f"Node {type(node).__name__} matches: {match}")
