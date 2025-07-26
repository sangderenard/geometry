import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#──────────────────────────────────────────────
# KDE symbolic example
#──────────────────────────────────────────────
x = sp.Symbol('x')
h = sp.Symbol('h', positive=True)
N = sp.Symbol('N', integer=True, positive=True)
idx = sp.IndexedBase('idx')
i = sp.Symbol('i', integer=True)

pdf = sp.Sum(1/(N*h) * (1/sp.sqrt(2*sp.pi)) * sp.exp(-((x - idx[i])/h)**2/2), (i, 0, N-1))
dpdf = sp.diff(pdf, x)
ipdf = sp.integrate(pdf, x)

idx_vals = (-10, 10, N)
idx_syms = sp.IndexedBase('idx')



#──────────────────────────────────────────────
# General symbolic n-D Fourier class
#──────────────────────────────────────────────
class SymbolicFourierND:
    def __init__(self, dims=2, N=3):
        self.dims = dims
        self.N = N
        self.vars = sp.symbols(f'x0:{dims}')
        self.ks = sp.symbols(f'k0:{dims}', integer=True)
        self.c = sp.IndexedBase('c')
        self.expr = self._build_expr()

    def _build_expr(self):
        sum_expr = 1
        for d in range(self.dims):
            sum_expr *= sp.Sum(
                sp.exp(sp.I * self.ks[d] * self.vars[d]) * self.c[self.ks[d]],
                (self.ks[d], -self.N, self.N)
            )
        return sum_expr

    def diff(self, var_idx):
        return sp.diff(self.expr, self.vars[var_idx])

    def integrate(self, var_idx):
        return sp.integrate(self.expr, self.vars[var_idx])

#──────────────────────────────────────────────
# Example: build symbolic Fourier in 2D
#──────────────────────────────────────────────
fourier2d = SymbolicFourierND(dims=10, N=10)
print("\n🚀 10D symbolic Fourier series:")
sp.pprint(fourier2d.expr, use_unicode=True)

from sympy.plotting import plot, plot3d_parametric_line
#──────────────────────────────────────────────
# Symbolic exploration purely in SymPy
#──────────────────────────────────────────────
def explore_symbolic_over_domain(symbolic_funcs, N_val=10, samples=1000):
    # Collect all free symbols across all expressions
    free_params = set()
    for func in symbolic_funcs:
        free_params |= func.free_symbols

    data_vectors = []
    for _ in range(samples):
        param_subs = {}
        for p in free_params:
            if p == h:
                param_subs[p] = np.random.uniform(0.5, 2.0)
            else:
                param_subs[p] = np.random.uniform(-10, 10)

        # Fill in index and coefficient arrays explicitly
        for i in range(N_val):
            param_subs[idx_syms[i]] = np.random.uniform(-10, 10)
        for k in range(-3,4):
            param_subs[fourier2d.c[k]] = np.random.uniform(-1,1) + 1j*np.random.uniform(-1,1)

        values = []
        for func in symbolic_funcs:
            val = sp.re(func.subs(param_subs).evalf())
            values.append(val)
        data_vectors.append(values)

    print(data_vectors)

    data_vectors = np.array(data_vectors)
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(data_vectors)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], s=5, alpha=0.6, cmap='viridis')
    ax.set_title("PCA of Symbolic Functions Over Fully Randomized Free Params")
    plt.show()

#──────────────────────────────────────────────
# Run exploratory visualization
#──────────────────────────────────────────────
from function_analyzer import FunctionAnalyzer
from graph_express2 import ProcessGraph
from torch_dag_runtime import TorchDAGRuntime
if __name__ == '__main__':
    symbolic_funcs = [pdf, dpdf, ipdf, fourier2d.expr]
    for i, symbolic_func in enumerate(symbolic_funcs):
        graph = ProcessGraph(5)
        graph.build_from_expression(symbolic_func)
        graph.finalize_graph_with_outputs()
        graph.compute_levels()
        #graph.print_lifespans_ascii()
        sp.pprint(symbolic_func, use_unicode=True)
        tdr = TorchDAGRuntime(graph, use_compiled=True, backend='numpy')
        
        torch_func = tdr.compiled()
        symbolic_funcs[i] = torch_func
    #var_symbols = [x] + list(fourier2d.vars)
    #ranges = [(-10,10)] * len(var_symbols)
    #explore_symbolic_over_domain(symbolic_funcs)
        #torch_func("test", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        #torch_func()

        analyzer = FunctionAnalyzer(symbolic_funcs[i])
        # Then hammer with 10k random tries
        analyzer.run_random(tries=100)

        # Get a high level summary
        print(analyzer.get_summary())

        # Op   tionally dump the raw data
        analyzer.dump_success_cases("success_cases.json")
