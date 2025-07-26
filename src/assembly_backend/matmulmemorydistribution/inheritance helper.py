import ast
from collections import defaultdict

def parse_classes(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filename)
    class_parents = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            name = node.name
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    # Handles e.g. module.Base
                    bases.append(base.attr)
                else:
                    bases.append(ast.dump(base))
            class_parents[name] = bases
    return class_parents

def print_tree(class_parents):
    # Build a reverse map: parent → [children]
    parent_to_children = defaultdict(list)
    for child, parents in class_parents.items():
        for parent in parents:
            parent_to_children[parent].append(child)
    # Find roots: classes that are not a child
    roots = [cls for cls in class_parents if not any(cls in children for children in parent_to_children.values())]

    def print_subtree(cls, depth=0, visited=None):
        if visited is None:
            visited = set()
        print("  " * depth + f"- {cls}")
        visited.add(cls)
        for child in parent_to_children.get(cls, []):
            if child not in visited:
                print_subtree(child, depth + 1, visited)

    for root in roots:
        print_subtree(root)

# Main usage:
filename = "bitops.py"  # Path to your file
class_parents = parse_classes(filename)
print("\nClass Inheritance Tree (by AST):\n")
print_tree(class_parents)
