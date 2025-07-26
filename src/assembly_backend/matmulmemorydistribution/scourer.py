import os
import json
import inspect
from typing import Type, Dict, Any, List

def collect_library_reference(roots: List[Type]) -> Dict[str, Dict[str, Any]]:
    """
    Given a list of root classes, recursively discover every subclass
    and record for each:
      - base class name
      - public methods
      - public members/attributes
      - one-line docstring
    Returns a dict mapping class name to its metadata.
    """
    result: Dict[str, Dict[str, Any]] = {}
    visited: set = set()

    def recurse(base: Type) -> None:
        for sub in base.__subclasses__():
            if sub in visited:
                continue
            visited.add(sub)
            members = [name for name, _ in inspect.getmembers(sub)]
            methods = [name for name, member in inspect.getmembers(sub, inspect.isroutine)]

            result[sub.__name__] = {
                "base": base.__name__,
                "members": members,
                "methods": methods,
                "doc": inspect.getdoc(sub).splitlines()[0] if inspect.getdoc(sub) else "",
            }
            recurse(sub)

    for root in roots:
        recurse(root)
    return result

# Broadened companion dictionary list
extra_paths = [
    'ast.AST',
    'sympy.Basic',
    'xml.etree.ElementTree.Element',
    'lxml.etree._Element',
    'anytree.node.Node',
    'sklearn.tree._classes.DecisionTreeClassifier',
    'sklearn.tree._classes.DecisionTreeRegressor',
    'scipy.spatial.KDTree',
    'scipy.spatial.cKDTree',
    'torch.nn.Module',
    'torch.fx.Node'
]

TREE_TYPES: Dict[str, Type] = {}
for path in extra_paths:
    module_path, class_name = path.rsplit('.', 1)
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        TREE_TYPES[path] = cls
    except (ImportError, AttributeError):
        continue

# Base directory for output
base_dir = './tree_references'
for pkg_path, cls in TREE_TYPES.items():
    folder = os.path.join(base_dir, *pkg_path.split('.'))
    os.makedirs(folder, exist_ok=True)
    ref = collect_library_reference([cls])
    with open(os.path.join(folder, 'reference.json'), 'w') as f:
        json.dump(ref, f, indent=2)

# Display summary of created reference folders
import pandas as pd
created = [root for root, dirs, files in os.walk(base_dir) if 'reference.json' in files]
df = pd.DataFrame(created, columns=["Directory"])

