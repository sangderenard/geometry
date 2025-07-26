"""
flatdag.py

Flat Directed Acyclic Graph representation with:
- Vocabulary: global string-to-int enumeration with persistence
- DAG: nodes, ports, edges, subgraph extraction, canonical hashing, pattern enumeration
- SCC condensation to collapse cycles into supernodes
- HandlerProfiler for profiling node occurrences
- FilterManager: load per-node templates to split ports into inputs/outputs
- build_port_groups: assign ports to input/output based on templates
- ingestion of Python AST, Sympy, and LLVM IR with configurable options
- SSA correlation helper for emit handler testing
- Corpus processing and unified hub for graph generation, condensation, correlation, and data dumps
"""

import argparse
import ast as pyast
import hashlib
import inspect
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union, Optional

from matplotlib import pyplot as plt
import sympy
import yaml
from llvmlite import ir as llvmir

#───────────────────────────────────────────────────────────────────────────────
# Vocabulary: global, file-backed string → int mapping
#───────────────────────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self):
        self._term_to_id: Dict[str, int] = {}
        self._next_id = 0

    def get_id(self, term: str) -> int:
        if term not in self._term_to_id:
            self._term_to_id[term] = self._next_id
            self._next_id += 1
        return self._term_to_id[term]

    def save(self, path: str):
        with open(path, "w") as f:
            for term, idx in sorted(self._term_to_id.items(), key=lambda t: t[1]):
                f.write(f"{idx}\t{term}\n")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        v = cls()
        with open(path) as f:
            for line in f:
                idx, term = line.rstrip("\n").split("\t", 1)
                v._term_to_id[term] = int(idx)
            v._next_id = max(v._term_to_id.values(), default=-1) + 1
        return v

#───────────────────────────────────────────────────────────────────────────────
# HandlerProfiler: count occurrences of whitelisted node types
#───────────────────────────────────────────────────────────────────────────────
class HandlerProfiler:
    def __init__(self, vocab: Vocabulary, whitelist: Optional[List[str]] = None):
        self.vocab = vocab
        self.whitelist = set(whitelist) if whitelist else set()
        self.counts: Dict[int, int] = defaultdict(int)

    def record(self, name: str):
        if not self.whitelist or name in self.whitelist:
            hid = self.vocab.get_id(name)
            self.counts[hid] += 1

    def save(self, path: str):
        inv = {v: k for k, v in self.vocab._term_to_id.items()}
        data = {inv[hid]: count for hid, count in self.counts.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

#───────────────────────────────────────────────────────────────────────────────
# Flat DAG: Nodes, Ports, Edges, SCC condensation, Subgraph Hashing & Patterns
#───────────────────────────────────────────────────────────────────────────────
NodeID = int
PortID = int
Edge   = Tuple[PortID, PortID]

class DAG:
    def __init__(self):
        self.nodes: Set[NodeID]                 = set()
        self.name_ref: Dict[NodeID, int]        = {}
        self.port_to_node: Dict[PortID, NodeID] = {}
        self.edges: List[Edge]                  = []
        self._next_node_id: NodeID              = 0
        self._next_port_id: PortID              = 0

    def new_node(self, name_vocab_id: int) -> NodeID:
        nid = self._next_node_id
        self._next_node_id += 1
        self.nodes.add(nid)
        self.name_ref[nid] = name_vocab_id
        return nid

    def add_edge(self, src: NodeID, dst: NodeID) -> Edge:
        outp = self._next_port_id; self._next_port_id += 1
        inp  = self._next_port_id; self._next_port_id += 1
        self.port_to_node[outp] = src
        self.port_to_node[inp]  = dst
        self.edges.append((outp, inp))
        self.nodes.update([src, dst])
        return outp, inp

    def all_nodes(self) -> List[NodeID]:
        return list(self.nodes)

    def all_edges(self) -> List[Edge]:
        return list(self.edges)

    def _build_adjacency(self):
        fw, bw = defaultdict(list), defaultdict(list)
        for outp, inp in self.edges:
            s, d = self.port_to_node[outp], self.port_to_node[inp]
            fw[s].append(d); bw[d].append(s)
        return fw, bw

    def extract_subgraph(self, root: NodeID, depth: int) -> Tuple[Set[NodeID], List[Edge]]:
        fw, bw = self._build_adjacency()
        visited, frontier = {root}, {root}
        for _ in range(depth):
            nxt = set()
            for n in frontier:
                for nb in fw[n] + bw[n]:
                    if nb not in visited:
                        visited.add(nb); nxt.add(nb)
            frontier = nxt
            if not frontier: break
        sub_edges = [(o,i) for o,i in self.edges
                     if self.port_to_node[o] in visited and self.port_to_node[i] in visited]
        return visited, sub_edges

    def canonical_subgraph_hash(self, nodes: Set[NodeID], sub_edges: List[Edge]) -> str:
        order = self._toposort(nodes, sub_edges)
        idx = {n:i for i,n in enumerate(order)}
        labels = tuple(self.name_ref[n] for n in order)
        edges_idx = sorted((idx[self.port_to_node[o]], idx[self.port_to_node[i]])
                            for o,i in sub_edges)
        return hashlib.sha256(repr((labels, tuple(edges_idx))).encode()).hexdigest()

    def _toposort(self, nodes: Set[NodeID], sub_edges: List[Edge]) -> List[NodeID]:
        fw, _ = self._build_adjacency()
        indeg = {n:0 for n in nodes}
        for o,i in sub_edges: indeg[self.port_to_node[i]] += 1
        q, order = deque([n for n,d in indeg.items() if d==0]), []
        while q:
            n = q.popleft(); order.append(n)
            for c in fw[n]:
                if c in indeg:
                    indeg[c] -= 1
                    if indeg[c]==0: q.append(c)
        return order

    def condense_scc(self) -> Tuple['DAG', Dict[NodeID, NodeID], List[List[NodeID]]]:
        # Tarjan's algorithm
        fw, _ = self._build_adjacency()
        index = 0; indices = {}; lowlink = {}
        stack, onstack, components = [], set(), []
        def strongconnect(v):
            nonlocal index
            indices[v] = index; lowlink[v] = index; index += 1
            stack.append(v); onstack.add(v)
            for w in fw[v]:
                if w not in indices:
                    strongconnect(w); lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in onstack:
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                comp = []
                while True:
                    w = stack.pop(); onstack.remove(w); comp.append(w)
                    if w == v: break
                components.append(comp)
        for v in self.nodes:
            if v not in indices: strongconnect(v)
        comp_map = {}; superdag = DAG()
        for cid, comp in enumerate(components):
            rep_vid = self.name_ref[comp[0]]
            snode = superdag.new_node(rep_vid)
            for n in comp: comp_map[n] = snode
        seen = set()
        for o,i in self.edges:
            u, v = self.port_to_node[o], self.port_to_node[i]
            cu, cv = comp_map[u], comp_map[v]
            if cu != cv and (cu,cv) not in seen:
                superdag.add_edge(cu, cv); seen.add((cu,cv))
        return superdag, comp_map, components

#───────────────────────────────────────────────────────────────────────────────
# Templates & Filters
#───────────────────────────────────────────────────────────────────────────────
@dataclass
class Template:
    inputs:  Union[List[str], str]
    outputs: Union[List[str], str]

class FilterManager:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab; self._templates = {}; self._default = Template("all","all")
    def load_file(self, path: str):
        raw = yaml.safe_load(open(path))
        if "_defaults" in raw:
            d = raw.pop("_defaults"); self._default = Template(d["inputs"], d["outputs"]);
        for name, spec in raw.items():
            vid = self.vocab.get_id(name)
            self._templates[vid] = Template(spec["inputs"], spec["outputs"])
    def get(self, nid: int) -> Template:
        return self._templates.get(nid, self._default)

def build_port_groups(dag: DAG, fman: FilterManager) -> Dict[NodeID, Dict[str, List[int]]]:
    inmap, outmap = defaultdict(list), defaultdict(list)
    for o,i in dag.edges:
        outmap[dag.port_to_node[o]].append(o);
        inmap[dag.port_to_node[i]].append(i)
    res = {}
    for n in dag.all_nodes():
        tpl = fman.get(dag.name_ref[n])
        ins = inmap[n] if tpl.inputs=="all" else inmap[n][:len(tpl.inputs)]
        outs= outmap[n] if tpl.outputs=="all" else outmap[n][:len(tpl.outputs)]
        in_names = [None]*len(ins) if tpl.inputs=="all" else tpl.inputs
        out_names= [None]*len(outs) if tpl.outputs=="all" else tpl.outputs
        res[n] = {"in_ports":ins, "in_names":in_names, "out_ports":outs, "out_names":out_names}
    return res

#───────────────────────────────────────────────────────────────────────────────
# Multi-language Monte Carlo Correlation Helper (agnostic to handler names)
#───────────────────────────────────────────────────────────────────────────────
import os
import json
from llvmlite import binding

import os
import json
from llvmlite import binding

def verify_graph_input_closure(
    dag: DAG,
    vocab: Vocabulary,
    type_graph_nodes: Set[int]
) -> bool:
    """
    Ensure that every node's inputs ultimately trace back to known type nodes.

    - dag: your full graph
    - vocab: global Vocabulary
    - type_graph_nodes: set of node IDs (int) that represent your canonical type system

    Returns True if closure holds, else False (prints detailed diagnostics).
    """
    broken = False

    # Build direct port-to-type map
    node_type_map = dag.name_ref  # NodeID -> vocab ID (int representing type string)

    # Each node's incoming edges must ultimately lead to a type node
    for n in dag.all_nodes():
        n_type_id = node_type_map.get(n)
        if n_type_id not in type_graph_nodes:
            print(f"⚠️ Node {n} with type '{vocab_term(vocab, n_type_id)}' is not in type graph.")
            broken = True

        # Check all inputs to this node
        incoming = [i for o,i in dag.all_edges() if dag.port_to_node[i] == n]
        for in_port in incoming:
            src_node = dag.port_to_node.get(in_port)
            src_type_id = node_type_map.get(src_node)
            if src_type_id not in type_graph_nodes:
                print(f"❌ Input from node {src_node} (type '{vocab_term(vocab, src_type_id)}') "
                      f"to node {n} (type '{vocab_term(vocab, n_type_id)}') "
                      f"is not covered by type graph.")
                broken = True

    if not broken:
        print("✅ Graph input closure verified: all nodes and inputs trace to known type nodes.")
    else:
        print("❌ Graph input closure failed: see above diagnostics.")

    return not broken


def vocab_term(vocab: Vocabulary, term_id: int) -> str:
    """Helper to reverse lookup term strings from vocab ids."""
    if term_id is None:
        return "<unknown>"
    inv = {v: k for k, v in vocab._term_to_id.items()}
    return inv.get(term_id, f"<unmapped-{term_id}>")


def test_llvm_corpus_with_llvmlite(
    corpus_dir: str,
    run_jit: bool = False,
    result_json_path: str = "llvm_corpus_test_results.json"
):
    """
    Test LLVM corpus .ll files by parsing and optionally JIT executing.
    Returns dictionary of successfully parsed ModuleRef.
    """
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    results = {}
    verified_modules = {}

    for fn in os.listdir(corpus_dir):
        if not fn.endswith(".ll"):
            continue
        filepath = os.path.join(corpus_dir, fn)
        try:
            ir_text = open(filepath).read()
        except Exception as e:
            results[fn] = {"status": "file_error", "error": str(e)}
            continue

        # Try parsing and verifying
        try:
            mod = binding.parse_assembly(ir_text)
            mod.verify()
            results[fn] = {"status": "parsed_and_verified"}
            verified_modules[fn] = mod
        except binding.LLVMException as e:
            results[fn] = {"status": "parse_or_verify_fail", "error": str(e)}
            continue

        # Optionally attempt to JIT and call 'main'
        if run_jit:
            try:
                target = binding.Target.from_default_triple()
                target_machine = target.create_target_machine()
                backing_mod = binding.parse_assembly("")
                engine = binding.create_mcjit_compiler(backing_mod, target_machine)

                engine.add_module(mod)
                engine.finalize_object()
                engine.run_static_constructors()

                try:
                    main_ptr = engine.get_function_address("main")
                    import ctypes
                    cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(main_ptr)
                    retval = cfunc()
                    results[fn].update({"jit_run": "success", "retval": retval})
                except Exception as e:
                    results[fn].update({"jit_run": "main_not_found_or_failed", "error": str(e)})
            except Exception as e:
                results[fn].update({"jit_compile_error": str(e)})

    with open(result_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"LLVM corpus tests complete. Results saved to {result_json_path}")
    return verified_modules

def generate_multi_lang_correlation(
    vocab: Vocabulary,
    correlation_path: str,
    emitter_map: Dict[str, Dict[int, callable]],
    test_types: Optional[List[type]] = None,
    monte_carlo_runs: int = 100
):
    """
    Perform Monte Carlo tests for every handler in every language,
    without filtering by existing vocabulary entries.

    - vocab: the shared Vocabulary
    - correlation_path: JSON file to write/read results
    - emitter_map: {lang: {name_ref_id: emitter_fn}}
    - test_types: types to instantiate per parameter
    - monte_carlo_runs: iterations per handler per lang
    """
    try:
        corr = json.load(open(correlation_path))
    except FileNotFoundError:
        corr = {}

    types = test_types or [int, float, str, bool]

    # Loop through each language and its handlers
    for lang, handlers in emitter_map.items():
        for name_id, emitter in handlers.items():
            # Use the numeric name_id as key for agnostic correlation
            sid = str(name_id)
            entry = corr.setdefault(sid, {})
            stats = entry.setdefault(lang, {'runs': 0, 'ok': 0, 'fail': 0})
            sig = inspect.signature(emitter)
            for i in range(monte_carlo_runs):
                args = []
                for idx, parm in enumerate(sig.parameters.values()):
                    t = types[idx % len(types)]
                    try:
                        args.append(t())
                    except:
                        args.append(None)
                stats['runs'] += 1
                try:
                    emitter(*args)
                    stats['ok'] += 1
                except:
                    stats['fail'] += 1

    with open(correlation_path, 'w') as f:
        json.dump(corr, f, indent=2)


#───────────────────────────────────────────────────────────────────────────────
# SSA Correlation Helper
#───────────────────────────────────────────────────────────────────────────────
def generate_ssa_correlation(
    dag: DAG,
    vocab: Vocabulary,
    correlation_path: str,
    emitter_map: Dict[str,Dict[str, callable]],
    test_types: Optional[List[type]] = None,
    default_lang: str = 'ssa'
):
    """
    Build or update a correlation table mapping node names → SSA emit handlers status.

    - dag: the flat DAG
    - vocab: Vocabulary instance
    - correlation_path: JSON file path for correlation table
    - emitter_map: mapping lang → name → emitter function
    - test_types: list of types to instantiate for args
    - default_lang: key in emitter_map to test against
    """
    inv_vocab = {v:k for k,v in vocab._term_to_id.items()}
    try:
        corr = json.load(open(correlation_path))
    except FileNotFoundError:
        corr = {}
    types = test_types or [int, float, str, bool]
    for nid in dag.all_nodes():
        name = inv_vocab.get(dag.name_ref[nid])
        if not name: continue
        table = corr.setdefault(name, {})
        if default_lang in table: continue
        emitter_lang = emitter_map.get(default_lang, {})
        emitter = emitter_lang.get(name)
        if not emitter: continue
        sig = inspect.signature(emitter)
        found=False
        for t in types:
            args=[]
            for p in sig.parameters:
                try: args.append(t())
                except: args.append(None)
            try:
                emitter(*args)
                table[default_lang]={'status':'ok'}; found=True; break
            except Exception:
                continue
        if not found:
            table[default_lang]={'status':'failed'}
    with open(correlation_path,'w') as f:
        json.dump(corr,f,indent=2)


#───────────────────────────────────────────────────────────────────────────────
# Ingestion: Python AST & Sympy class hierarchies with options
#───────────────────────────────────────────────────────────────────────────────

def ingest_ast(
    dag: DAG,
    vocab: Vocabulary,
    include_private: bool = False,
    field_whitelist: Optional[List[str]] = None,
    include_inheritance: bool = True,
    profiler: Optional[HandlerProfiler] = None
):
    """Build nodes/edges for all ast.AST subclasses and their fields/components."""
    field_whitelist = set(field_whitelist or [])
    class_to_nid: Dict[type, NodeID] = {}

    def collect_ast(cls):
        for sub in cls.__subclasses__():
            if sub not in class_to_nid:
                vid = vocab.get_id(sub.__name__)
                nid = dag.new_node(vid)
                if profiler: profiler.record(sub.__name__)
                class_to_nid[sub] = nid
                collect_ast(sub)

    collect_ast(pyast.AST)

    # optional inheritance edges
    if include_inheritance:
        for cls, nid in class_to_nid.items():
            for base in cls.__bases__:
                if base in class_to_nid:
                    dag.add_edge(class_to_nid[base], nid)

    # field components
    for cls, nid in class_to_nid.items():
        for fld in getattr(cls, '_fields', ()) or ():
            if not include_private and fld.startswith('_') and fld not in field_whitelist:
                continue
            vid = vocab.get_id(fld)
            fnid = dag.new_node(vid)
            dag.add_edge(nid, fnid)
            if profiler: profiler.record(fld)

    # other class components
    for cls, nid in class_to_nid.items():
        for name, member in cls.__dict__.items():
            if not include_private and name.startswith('_') and name not in field_whitelist:
                continue
            if name in getattr(cls, '_fields', ()) or name == '__doc__':
                continue
            vid = vocab.get_id(name)
            cnid = dag.new_node(vid)
            dag.add_edge(nid, cnid)
            if profiler: profiler.record(name)


def ingest_sympy(
    dag: DAG,
    vocab: Vocabulary,
    include_private: bool = False,
    include_inheritance: bool = True,
    profiler: Optional[HandlerProfiler] = None
):
    """Build nodes/edges for all sympy.Basic subclasses and their components."""
    class_to_nid: Dict[type, NodeID] = {}

    def collect_sympy(cls):
        for sub in cls.__subclasses__():
            if sub not in class_to_nid:
                vid = vocab.get_id(sub.__name__)
                nid = dag.new_node(vid)
                if profiler: profiler.record(sub.__name__)
                class_to_nid[sub] = nid
                collect_sympy(sub)

    from sympy import Basic
    collect_sympy(Basic)

    # optional inheritance edges
    if include_inheritance:
        for cls, nid in class_to_nid.items():
            for base in cls.__bases__:
                if base in class_to_nid:
                    dag.add_edge(nid, class_to_nid[base])

    # class components
    for cls, nid in class_to_nid.items():
        for name, member in cls.__dict__.items():
            if not include_private and name.startswith('_'):
                continue
            vid = vocab.get_id(name)
            cnid = dag.new_node(vid)
            dag.add_edge(nid, cnid)
            if profiler: profiler.record(name)
def ingest_llvmlite_module(
    dag: DAG,
    vocab: Vocabulary,
    module: llvmir.Module,
    profiler: Optional[HandlerProfiler] = None
):
    inst_to_nid: Dict[llvmir.Instruction, NodeID] = {}
    def reg(name: str) -> int:
        vid = vocab.get_id(name)
        if profiler: profiler.record(name)
        return vid
    for func in module.functions:
        fvid = reg(f"Function:{func.name}"); fnid = dag.new_node(fvid)
        for block in func.blocks:
            bvid = reg(f"BasicBlock:{func.name}.{block.name}"); bnid = dag.new_node(bvid)
            dag.add_edge(fnid, bnid)
            for inst in block.instructions:
                # Fix: handle missing 'opname' attribute
                if hasattr(inst, "opname"):
                    inst_name = f"Inst:{inst.opname}"
                elif hasattr(inst, "name"):
                    inst_name = f"Inst:{inst.name}"
                else:
                    inst_name = f"Inst:{type(inst).__name__}"
                ivid = reg(inst_name)
                inid = dag.new_node(ivid)
                inst_to_nid[inst] = inid
                dag.add_edge(bnid, inid)
    for inst, inid in inst_to_nid.items():
        for op in inst.operands:
            if isinstance(op, llvmir.Instruction):
                dag.add_edge(inst_to_nid[op], inid)
            else:
                cvid = reg(f"Const:{type(op).__name__}")
                cnid = dag.new_node(cvid); dag.add_edge(cnid, inid)

#───────────────────────────────────────────────────────────────────────────────
# Corpus processors
#───────────────────────────────────────────────────────────────────────────────
def process_python_corpus(
    root: str,
    dag: DAG,
    vocab: Vocabulary,
    profiler: Optional[HandlerProfiler],
    **ingest_kwargs
):
    for dp,_,fns in os.walk(root):
        for fn in fns:
            if not fn.endswith('.py'): continue
            path = os.path.join(dp, fn)
            try:
                src = open(path, 'r', encoding='utf-8').read()
                tree = pyast.parse(src, filename=path)
            except Exception:
                continue
            ingest_ast(dag, vocab, profiler=profiler, **ingest_kwargs)


def process_llvm_corpus(
    root: str,
    dag: DAG,
    vocab: Vocabulary,
    profiler: Optional[HandlerProfiler]
):
    for fn in os.listdir(root):
        if not fn.endswith('.ll'): continue
        ir_text = open(os.path.join(root, fn)).read()
        
        module = binding.parse_assembly(ir_text)
        module.verify()
        ingest_llvmlite_module(dag, vocab, module, profiler)

def save_dag(dag: DAG, path: str):
    data = {
        'nodes': list(dag.nodes),
        'name_ref': dag.name_ref,
        'port_to_node': dag.port_to_node,
        'edges': dag.edges,
    }
    with open(path, 'w') as f:
        json.dump(data, f)

def default_sequence_all(
    python_corpus_dir="./python_samples",
    llvm_corpus_dir="./llvm_samples",
    output_dir="./output",
    run_jit=False
):
    os.makedirs(output_dir, exist_ok=True)

    vocab = Vocabulary()
    profiler = HandlerProfiler(vocab)
    type_dag = DAG()
    full_dag = DAG()

    # Build AST type graph (superset of all possible)
    ingest_ast(type_dag, vocab, profiler=profiler)
    ingest_ast(full_dag, vocab, profiler=profiler)

    # Build Sympy type graph into type_dag + full DAG
    ingest_sympy(type_dag, vocab, profiler=profiler)
    ingest_sympy(full_dag, vocab, profiler=profiler)

    # Add Python corpus (actual instance trees)
    process_python_corpus(python_corpus_dir, full_dag, vocab, profiler)

    # Add LLVM corpus (actual modules)
    process_llvm_corpus(llvm_corpus_dir, full_dag, vocab, profiler)
    verified_modules = test_llvm_corpus_with_llvmlite(llvm_corpus_dir, run_jit=run_jit)

    # Check closure: all nodes inputs must reduce to known types
    type_nodes = set(type_dag.all_nodes())
    closure_ok = verify_graph_input_closure(full_dag, vocab, type_nodes)
    if not closure_ok:
        print("🚨 Aborting correlation: graph failed closure verification.")
        return

    # Build emitter map for correlation
    llvm_emitters = {}
    for filename, mod in verified_modules.items():
        name_id = vocab.get_id(f"llvm_corpus::{filename}")
        def make_emitter(m): return lambda: m.verify()
        llvm_emitters[name_id] = make_emitter(mod)

    emitter_map = {
        "llvm_corpus": llvm_emitters
    }

    # Run correlation
    generate_multi_lang_correlation(
        vocab,
        correlation_path=os.path.join(output_dir, "multi_lang_correlation.json"),
        emitter_map=emitter_map,
        monte_carlo_runs=10
    )

    # Save everything
    vocab.save(os.path.join(output_dir, "vocab.txt"))
    save_dag(full_dag, os.path.join(output_dir, "full_dag.json"))
    save_dag(type_dag, os.path.join(output_dir, "type_dag.json"))

    print("✅ Default sequence complete: all graphs built, verified, correlated.")


#───────────────────────────────────────────────────────────────────────────────
# Main: unified hub
#───────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser("flatdag unified generator hub")
    p.add_argument('--vocab', default='vocab.txt')
    p.add_argument('--out-dag', default='full_dag.json')
    p.add_argument('--condensed-dag', default=None)
    p.add_argument('--ast-profile', default=None)
    p.add_argument('--sympy-profile', default=None)
    p.add_argument('--ssa-corr', default=None)
    p.add_argument('--python-corpus', default=None)
    p.add_argument('--llvm-corpus', default=None)
    p.add_argument('--include-private', action='store_true')
    p.add_argument('--no-inheritance', dest='inheritance', action='store_false')
    p.add_argument('--fields-whitelist', nargs='*', default=None)
    p.add_argument('--profile-whitelist', nargs='*', default=None)
    args = p.parse_args()

    vocab = Vocabulary()
    profiler = HandlerProfiler(vocab, args.profile_whitelist) if args.profile_whitelist else None
    dag = DAG()

    # Ingest AST and Sympy
    ingest_ast(dag, vocab, include_private=args.include_private,
               include_inheritance=args.inheritance,
               field_whitelist=args.fields_whitelist,
               profiler=profiler)
    ingest_sympy(dag, vocab, include_private=args.include_private,
                 include_inheritance=args.inheritance,
                 profiler=profiler)

    # Optional corpus expansions
    ingest_kwargs = dict(include_private=args.include_private,
                         include_inheritance=args.inheritance,
                         field_whitelist=args.fields_whitelist)
    if args.python_corpus:
        process_python_corpus(args.python_corpus, dag, vocab, profiler, **ingest_kwargs)
    if args.llvm_corpus:
        process_llvm_corpus(args.llvm_corpus, dag, vocab, profiler)

    # Save full DAG
    with open(args.out_dag, 'w') as f:
        json.dump({'nodes':list(dag.nodes),'name_ref':dag.name_ref,
                   'port_to_node':dag.port_to_node,'edges':dag.edges}, f)

    # Condense SCC if requested
    if args.condensed_dag:
        superdag, comp_map, comps = dag.condense_scc()
        with open(args.condensed_dag, 'w') as f:
            json.dump({'nodes':list(superdag.nodes),'name_ref':superdag.name_ref,
                       'edges':superdag.edges}, f)

    # SSA correlation
    if args.ssa_corr:
        generate_ssa_correlation(dag, vocab, args.ssa_corr,
                                 emitter_map={}, test_types=[int,float,str,bool])

    # Save profiles
    if profiler and args.ast_profile:
        profiler.save(args.ast_profile)
    if profiler and args.sympy_profile:
        profiler.save(args.sympy_profile)

    # Persist vocabulary
    vocab.save(args.vocab)

if __name__ == '__main__':
    main()

    default_sequence_all(
        python_corpus_dir="./python_samples",
        llvm_corpus_dir="./llvm_samples",
        output_dir="./examiner_output",
        run_jit=True
    )

def export_as_simple_graph(dag: DAG, vocab: Vocabulary):
    """Exports a universal graph dict with resolved type names."""
    inv_vocab = {v: k for k, v in vocab._term_to_id.items()}
    nodes = []
    for nid in dag.all_nodes():
        nodes.append({
            'id': nid,
            'type': inv_vocab.get(dag.name_ref[nid], f"<unknown-{dag.name_ref[nid]}>"),
            'label': inv_vocab.get(dag.name_ref[nid], f"<unknown-{dag.name_ref[nid]}>"),
        })
    edges = []
    for outp, inp in dag.all_edges():
        src = dag.port_to_node[outp]
        tgt = dag.port_to_node[inp]
        edges.append({'src': src, 'tgt': tgt})
    import networkx as nx
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node['id'], type=node['type'], label=node['label'])
    for edge in edges:
        G.add_edge(edge['src'], edge['tgt'])
    #visualize the graph
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black')
    plt.show()
    return {'nodes': nodes, 'edges': edges}

if __name__ == "__main__":
    import sympy
    import ast
    from graph_express2 import ProcessGraph  # Assuming PYTHONPATH set or co-located

    # Step 1: Build universal DAG from source (SymPy and AST)
    vocab = Vocabulary()
    dag = DAG()

    # Example: Ingest a simple sympy and ast example
    sympy_expr = sympy.sympify("a + b * c")
    ingest_sympy(dag, vocab)
    # Optionally, also build a live instance tree of this expression
    # (Not just class hierarchy, but concrete instance tree)

    # Example: Ingest Python AST
    src_code = "def foo(x): return x + 1"
    tree = ast.parse(src_code)
    ingest_ast(dag, vocab)

    # Step 2: Export universal simple graph
    flat_graph = export_as_simple_graph(dag, vocab)

    # Step 3: Ingest into ProcessGraph
    pg = ProcessGraph()
    pg.build_from_ast(dag)  # <--- this method we define below

    pg.finalize_graph_with_outputs()
    pg.compute_levels("asap")

    pg.print_lifespans_ascii()
    