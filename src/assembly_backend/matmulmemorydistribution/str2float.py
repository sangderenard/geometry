import matplotlib.pyplot as plt
import numpy as np
import random
import sympy as sp
import uuid
import time
from collections import OrderedDict

#─────────────────────────────────────────────────────────────
# Core DeclusterMap
#─────────────────────────────────────────────────────────────
class DeclusterMap:
    _registry = OrderedDict()

    def __init__(self, mapping, symbolic_f=None, symbolic_inv=None, ttl=None):
        self.mapping = mapping
        self.symbolic_f = symbolic_f
        self.symbolic_inv = symbolic_inv
        self.created = time.time()
        self.ttl = ttl

    
    @classmethod
    def create(cls, values, low=0.0, high=1.0, symbolic=True, ttl=60, persistence=False):
        if not values:
            raise ValueError("No values provided")

        # Normalize strings into integer sums
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        base = len(alphabet)
        max_length = 20
        char_to_index = {c: i for i, c in enumerate(alphabet)}
        total_space = sum(base ** l for l in range(1, max_length + 1))

        def str_sum(s):
            pos = 0
            for i, c in enumerate(s):
                if i >= max_length:
                    break
                idx = char_to_index.get(c, 0)
                power = max_length - i - 1
                pos += idx * (base ** power)
            return pos

        # Build integer sum list
        int_sums = [str_sum(s) for s in values]
        sorted_pairs = sorted(zip(int_sums, values))  # sort by int sum

        # Build mapping from int sum to uniform space
        N = len(sorted_pairs)
        mapping = [(pair[0], low + (high - low) * (i/(N-1)) if N>1 else (low+high)/2)
                for i, pair in enumerate(sorted_pairs)]
        
        x_data, y_data = zip(*mapping)
        x_sym = sp.Symbol('x')
        y_sym = sp.Symbol('y')
        symbolic_f = symbolic_inv = None

        if symbolic:
            symbolic_f = sp.interpolating_spline(1, list(x_data), list(y_data), x_sym)
            symbolic_inv = sp.interpolating_spline(1, list(y_data), list(x_data), y_sym)

        m = DeclusterMap(mapping, symbolic_f, symbolic_inv, ttl=None if persistence else ttl)
        token = str(uuid.uuid4())
        cls._registry[token] = m
        return token


    @classmethod
    def get(cls, token):
        obj = cls._registry.get(token)
        if obj is None:
            raise KeyError("Token expired or not found.")
        if obj.ttl and (time.time() - obj.created) > obj.ttl:
            del cls._registry[token]
            raise KeyError("Token expired.")
        return obj

    def transform(self, x_query):
        xs, ys = zip(*self.mapping)
        return np.interp(x_query, xs, ys)

    def inverse(self, y_query):
        ys, xs = zip(*self.mapping)
        return np.interp(y_query, ys, xs)

    def backward(self):
        if self.symbolic_f:
            return sp.diff(self.symbolic_f)
        raise RuntimeError("No symbolic function available for backward differentiation")

    def integral(self):
        if self.symbolic_f:
            return sp.integrate(self.symbolic_f)
        raise RuntimeError("No symbolic function available for integration")

#─────────────────────────────────────────────────────────────
# StringLexNormalizer
#─────────────────────────────────────────────────────────────
class StringLexNormalizer:
    def __init__(self, alphabet: str, max_length: int):
        self.alphabet = alphabet
        self.base = len(alphabet)
        self.max_length = max_length
        self.char_to_index = {c: i for i, c in enumerate(alphabet)}
        self.total_space = sum(self.base ** l for l in range(1, max_length + 1))

    def strtofloat(self, s: str, low=0.0, high=1.0) -> float:
        pos = 0
        for i, c in enumerate(s):
            if i >= self.max_length:
                break
            idx = self.char_to_index.get(c, 0)
            power = self.max_length - i - 1
            pos += idx * (self.base ** power)
        norm = pos / self.total_space
        return low + norm * (high - low)

#─────────────────────────────────────────────────────────────
# Torture test driver
#─────────────────────────────────────────────────────────────
def monte_carlo_lex_plot(words, num_samples=500, low=0.0, high=1.0, seed=None, normalize=True, decluster=True):
    lex_norm = StringLexNormalizer('abcdefghijklmnopqrstuvwxyz', max_length=20)
    samples = []

    for _ in range(num_samples):
        w = words[:]
        random.shuffle(w)
        sentence = ''.join(w)
        samples.append(sentence)

    positions = [lex_norm.strtofloat(s) for s in samples]
    if decluster:
        decluster_map = DeclusterMap.create(samples, low=low, high=high)
        positions = [DeclusterMap.get(decluster_map).transform(p) for p in positions]

    if normalize:
        positions = [(p - min(positions)) / (max(positions) - min(positions)) for p in positions]
    plt.figure(figsize=(14, 6))
    plt.scatter(positions, np.zeros_like(positions), marker='o', color='blue')

    for x, label in zip(positions, samples):
        plt.text(x, 0.02, label, rotation=90, verticalalignment='bottom', horizontalalignment='center', fontsize=7)

    plt.yticks([])
    plt.xlabel("Lexicographic Normalized Position")
    plt.title(f"Monte Carlo Shuffle of Words Mapped to Normalized Lex Position\n(samples={num_samples})")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()

#─────────────────────────────────────────────────────────────
# Run it
#─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    words = ["add", "sub", "mul", "div", "load", "store", "call", "ret"]
    monte_carlo_lex_plot(words, num_samples=1000)
