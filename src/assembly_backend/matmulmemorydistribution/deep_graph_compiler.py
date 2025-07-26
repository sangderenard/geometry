# deep_graph_compiler.py
"""Turn a fully‑levelled ProcessGraph into a single Python function.

The emitted function is *pure* (no internal state) and therefore fast in
serial execution loops.  All operator kernels are looked‑up from the
provided `op_table` so the same compiler works for torch, numpy, jax, ….

Example
-------
>>> pg.compute_levels("asap")
>>> compile_pg = GraphDeepCompiler(pg, op_table)
>>> f = compile_pg.build_function()
>>> out1, out2 = f(x=np.ones(3), y=np.arange(3))
"""
from __future__ import annotations

import textwrap, inspect, hashlib, types
from typing import Any, Callable, Dict, List, Tuple

class GraphDeepCompiler:
    """Compile a *level‑sorted* ProcessGraph into one Python function."""

    #: attribute names we expect on ProcessGraph nodes
    _REQ = ("type", "label", "parents")

    def __init__(self, pg: "ProcessGraph", op_table: Dict[str, Callable]):
        self.pg        = pg
        self.op_table  = op_table
        self.op_table["Store"] = lambda a: a  # Store just returns its input
        self._code     = None          # str
        self._fn       = None          # compiled callable

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------
    def build_function(self, device=None) -> Callable:
        """Return (and cache) a function `f(**inputs) -> tuple(outputs)`."""
        if self._fn is not None:
            return self._fn

        src, env, output_names = self._emit_source(device=device)
        code = compile(src, filename="<graph_fn>", mode="exec")
        ns: Dict[str, Any] = {}
        ns.update(env)
        exec(code, ns)
        self._fn = ns["graph_fn"]
        self._code = src
        self._outs = output_names
        return self._fn

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _emit_source(self, *, device):
        """Generate python source for the graph and a globals‑env dict."""
        G        = self.pg.G
        levels   = self.pg.levels               # nid -> level idx
        max_lvl  = max(levels.values())
        run_order: List[int] = []
        for lv in range(max_lvl+1):
            # deterministic order inside a level -> sorted nids
            run_order.extend(sorted(n for n,l in levels.items() if l==lv))

        lines: List[str] = ["def graph_fn(**inputs):"]
        env: Dict[str, Any] = {}
        indent = " " * 4

        for nid in run_order:
            node = G.nodes[nid]
            for k in self._REQ:
                if k not in node:
                    raise KeyError(f"ProcessGraph node missing '{k}' field")

            ntype = node["type"]
            lhs   = f"v{nid}"           # unique local name

            if ntype in ("Symbol", "Input", "Var"):
                # pure argument
                label = node["label"]
                lines.append(f"{indent}{lhs} = inputs[{label!r}]")
                continue
                
            # operator
            fn = self.op_table.get(ntype)
            if fn is None:
                raise KeyError(f"No operator impl for '{ntype}'")
            fn_name = f"op_{nid}"
            env[fn_name] = fn

            # parents come in topo order already
            args = ", ".join(f"v{pid}" for pid, _ in node["parents"])
            lines.append(f"{indent}{lhs} = {fn_name}({args})")

        #  final return – collect nodes marked as outputs / Store
        outputs = [n for n, data in G.nodes(data=True)
                    if data.get("type") in ("Store", "Output")]
        if not outputs:
            # fallback: last node in topo order
            outputs = [run_order[-1]]
        out_expr = ", ".join(f"v{n}" for n in outputs)
        lines.append(f"{indent}return ({out_expr},)\n")

        return textwrap.dedent("\n".join(lines)), env, outputs

    # ------------------------------------------------------------------
    # misc helpers / diagnostics
    # ------------------------------------------------------------------
    def code(self) -> str:
        """Return generated source as text (compiles lazily)."""
        if self._code is None:
            self.build_function()
        return self._code

    def hash(self) -> str:
        """Return a stable hash of the generated source (after build)."""
        src = self.code().encode()
        return hashlib.sha1(src).hexdigest()

# ────────────────────────────────────────────────────────────────────────
# quick self‑test  (run as `python deep_graph_compiler.py`)
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sympy as sp, numpy as np
    from graph_express2 import ProcessGraph
    # toy graph x + y * z
    pg = ProcessGraph()
    x,y,z = sp.symbols("x y z")
    expr = x + y*z
    pg.build_from_expression(expr)
    pg.compute_levels("asap")

    # ops for numpy
    op_table = {
        "Mul": lambda a,b: a*b,
        "Add": lambda a,b: a+b,
    }

    compiler = GraphDeepCompiler(pg, op_table)
    f = compiler.build_function()
    # data
    X = np.array([1,2,3])
    Y = np.array([10,20,30])
    Z = np.array([2,2,2])
    out, = f(x=X, y=Y, z=Z)
    print("result", out)
