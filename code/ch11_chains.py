"""ch11_chains.py -- identifying a chain of qubits: correlative sparsity is a light cone.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi) and cvxpy with
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

A chain of n qubits with the Hamiltonian
    H = sum_i h_i X_i + sum_i J_i Z_i Z_{i+1},
n local fields h_i and n - 1 couplings J_i, all unknown.  The chain is prepared
in a product state, every qubit in the same state of Bloch vector
(sin 1, 0, cos 1) or (0, sin 0.7, cos 0.7), and the first two time derivatives
at t = 0 of <X_i>, <Y_i>, <Z_i> are estimated at every site, with Gaussian
errors of standard deviation 0.02.  By the Heisenberg equation,
    d<O>/dt = <i[H, O]>,      d^2<O>/dt^2 = <i[H, i[H, O]]>,
the first is linear and the second quadratic in the parameters, and in a
product state the derivatives at the site i involve only the light cone
{h_{i-1}, J_{i-1}, h_i, J_i, h_{i+1}}.  The least-squares fit of the 12n
derivatives is a polynomial problem of degree 4 in 2n - 1 variables, and its
correlative-sparsity graph, in the order h_1, J_1, h_2, ..., h_n, has the n - 2
maximal cliques C_i = {h_{i-1}, J_{i-1}, h_i, J_i, h_{i+1}}, i = 2, ..., n-1,
each sharing three variables with the next: the clique tree is a chain.
At order 2 the dense relaxation has one PSD block of order binom(2n+1, 2) =
n(2n+1) (136 at n = 8, 1176 at n = 24); the sparse one has n - 2 blocks of
order binom(7, 2) = 21.

(a) The parameters on the chain, the light cones (the cliques), and the clique
    tree drawn as a chain of cores in the notation of Chapter 7.
(b) The order of the largest PSD block against n.
(c) The solve times with SCS on this machine: dense for n <= 8, sparse up to
    n = 24.
(d) The fields and couplings of the chain of n = 24 qubits: the true values
    and the estimate read off the first pseudo-moments.
Every SCS status is "optimal"; the dense and sparse bounds agree to 1e-7 for
n <= 8; at every n the first pseudo-moments give a point whose objective
exceeds the bound by less than 1e-7, so the bound is attained and the point is
the global least-squares estimate.  Least squares from the true parameters
(scipy) finds the same value and the same point.  The estimate is within 0.03
of the true parameters at every n, and within 1e-4 when the data are exact.
The derivatives are checked against the commutators on the whole chain and
against finite differences of exp(-iHt) for n = 5.
Writes figures/ch11_chains.tex.
"""
import os
import sys
import time
from functools import reduce

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp
import needham as N
import tikzexport as T
from ch07_tn_networks import GOLD

I2, SX, SY, SZ = T.pauli()
BLOCH = [np.array([np.sin(1.0), 0.0, np.cos(1.0)]), np.array([0.0, np.sin(0.7), np.cos(0.7)])]
SIGMA = 0.02
NS_SPARSE = [3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24]
NS_DENSE = [3, 4, 5, 6, 7, 8]
N_SHOW = 24
OPTS = dict(solver="SCS", eps=1e-8, max_iters=200000)


def names(n):
    """The variables in chain order: ('h', i) and ('J', i), i from 0."""
    out = []
    for i in range(n):
        out.append(("h", i))
        if i < n - 1:
            out.append(("J", i))
    return out


def ket(v):
    th, ph = np.arccos(v[2]), np.arctan2(v[1], v[0])
    return np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])


def string(ops, L):
    """A Pauli string on L sites, given as {site: matrix}."""
    return reduce(np.kron, [ops.get(k, I2) for k in range(L)])


def term(t, lo, L):
    kind, i = t
    return string({i - lo: SX} if kind == "h" else {i - lo: SZ, i + 1 - lo: SZ}, L)


def coefficients(n, i, obs, v, lo=None, hi=None):
    """d<O_i>/dt = sum_a lin[a] z_a and d^2<O_i>/dt^2 = sum_ab quad[a,b] z_a z_b at t = 0.

    Computed on the sites lo..hi (by default the window i-2..i+2, which holds the light cone)."""
    lo = max(0, i - 2) if lo is None else lo
    hi = min(n - 1, i + 2) if hi is None else hi
    L = hi - lo + 1
    idx = [a for a, t in enumerate(names(n)) if lo <= t[1] and t[1] + (t[0] == "J") <= hi]
    psi = reduce(np.kron, [ket(v)] * L)
    O = string({i - lo: obs}, L)
    P = {a: term(names(n)[a], lo, L) for a in idx}
    lin, quad = {}, {}
    for b in idx:
        c1 = 1j * (P[b] @ O - O @ P[b])
        val = np.vdot(psi, c1 @ psi)
        assert abs(val.imag) < 1e-12
        if abs(val.real) > 1e-12:
            lin[b] = val.real
        for a in idx:
            c2 = 1j * (P[a] @ c1 - c1 @ P[a])
            val = np.vdot(psi, c2 @ psi)
            assert abs(val.imag) < 1e-12
            if abs(val.real) > 1e-12:
                quad[(a, b)] = val.real
    return lin, quad


def model(n):
    """The rows of the fit: (site, linear or quadratic, coefficients)."""
    rows = []
    for v in BLOCH:
        for i in range(n):
            for O in (SX, SY, SZ):
                lin, quad = coefficients(n, i, O, v)
                rows += [(i, lin), (i, quad)]
    return [r for r in rows if r[1]]


def evaluate(row, z):
    return sum(w * (z[k[0]] * z[k[1]] if isinstance(k, tuple) else z[k]) for k, w in row[1].items())


def truth(n):
    rng = np.random.default_rng(100 + n)
    z = np.array([rng.uniform(0.6, 1.2) if k == "h" else rng.uniform(0.3, 0.8) for k, _ in names(n)])
    return z, rng


def instance(n, noise=SIGMA):
    z0, rng = truth(n)
    rows = model(n)
    d = np.array([evaluate(r, z0) for r in rows]) + noise * rng.standard_normal(len(rows))
    return z0, rows, d


def objective(n, rows, d):
    zs = psp.polyvar(2 * n - 1)
    f = psp.Polynomial.zero(2 * n - 1)
    for (_, c), di in zip(rows, d):
        p = -di
        for k, w in c.items():
            p = p + w * (zs[k[0]] * zs[k[1]] if isinstance(k, tuple) else zs[k])
        f = f + p ** 2
    return f


def graph(n):
    """The correlative-sparsity graph: pairs of variables that share a monomial of the objective."""
    z0, rows, d = instance(n)
    edges = set()
    for e in objective(n, rows, d).terms:
        vs = [k for k, a in enumerate(e) if a]
        edges |= {(a, c) for a in vs for c in vs if a < c}
    return sorted(edges)


def solve(n, sparse, noise=SIGMA):
    z0, rows, d = instance(n, noise)
    f = objective(n, rows, d)
    t = time.perf_counter()
    res = psp.sos_lower_bound(f, order=2, sparse=sparse, **OPTS)
    return dict(res=res, time=time.perf_counter() - t, z0=z0, rows=rows, d=d, x=res.minimizer())


def fit(rows, d, z):
    """Local least squares (scipy) from the point z: the independent check."""
    r = least_squares(lambda x: np.array([evaluate(row, x) for row in rows]) - d, z, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return r.x, float(np.sum(r.fun ** 2))


def compute():
    out = {"sparse": {n: solve(n, True) for n in NS_SPARSE},
           "dense": {n: solve(n, False) for n in NS_DENSE},
           "exact": solve(8, True, noise=0.0)}
    return out


def checks(out):
    # the light-cone window against the commutators on the whole chain, and those against exp(-iHt)
    n = 5
    z = truth(n)[0]
    H = sum(w * term(t, 0, n) for w, t in zip(z, names(n)))
    for v in BLOCH:
        psi = reduce(np.kron, [ket(v)] * n)
        for i in range(n):
            for O in (SX, SY, SZ):
                lin, quad = coefficients(n, i, O, v)
                full = coefficients(n, i, O, v, lo=0, hi=n - 1)
                assert set(lin) == set(full[0]) and set(quad) == set(full[1])
                assert all(abs(lin[k] - full[0][k]) < 1e-12 for k in lin) and all(abs(quad[k] - full[1][k]) < 1e-12 for k in quad)
                Oi = string({i: O}, n)
                ev = lambda t: np.vdot(expm(-1j * H * t) @ psi, Oi @ expm(-1j * H * t) @ psi).real
                h = 1e-3
                d1, d2 = (ev(h) - ev(-h)) / (2 * h), (ev(h) - 2 * ev(0) + ev(-h)) / h ** 2
                assert abs(d1 - evaluate((i, lin), z)) < 1e-5 and abs(d2 - evaluate((i, quad), z)) < 1e-4
                assert {names(n)[a][1] for a in lin} | {names(n)[a][1] for k in quad for a in k} <= {i - 1, i, i + 1}
    # the cliques: n - 2 light cones of five variables, blocks of order 21; dense n(2n+1)
    windows = [set(range(2 * i - 2, 2 * i + 3)) for i in range(1, 6)]
    assert graph(7) == sorted((a, c) for a in range(13) for c in range(a + 1, 13) if any({a, c} <= w for w in windows))
    for n, s in out["sparse"].items():
        res = s["res"]
        assert sorted(map(sorted, res.cliques)) == [[2 * i - 2, 2 * i - 1, 2 * i, 2 * i + 1, 2 * i + 2] for i in range(1, n - 1)]
        assert len(res.block_sizes) == n - 2 and max(res.block_sizes) == 21
    for n, s in out["dense"].items():
        assert max(s["res"].block_sizes) == n * (2 * n + 1)
    # certified: optimal status, dense = sparse, the first pseudo-moments attain the bound, scipy agrees
    for key in ("sparse", "dense"):
        for n, s in out[key].items():
            res = s["res"]
            assert res.status == "optimal", (key, n, res.status)
            assert abs(res.gap(s["x"])) < 1e-7, (key, n, res.gap(s["x"]))
            xl, fl = fit(s["rows"], s["d"], s["z0"])
            assert abs(fl - res.value) < 1e-7 and np.abs(xl - s["x"]).max() < 1e-3, (key, n, fl, res.value)
            assert np.abs(s["x"] - s["z0"]).max() < 0.03, (key, n)
    for n in NS_DENSE:
        assert abs(out["dense"][n]["res"].value - out["sparse"][n]["res"].value) < 1e-7
    ex = out["exact"]
    assert ex["res"].status == "optimal" and abs(ex["res"].value) < 1e-7 and np.abs(ex["x"] - ex["z0"]).max() < 1e-4


def log_ticks(pl, values, labels):
    out = []
    for v, lab in zip(values, labels):
        p = pl.P(pl.xlim[0], np.log10(v))
        out.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[left, font=\\scriptsize, text=ndInk] {%s};"
                   % (p[0] + 0.06, p[1], p[0] - 0.06, p[1], lab))
    return out


def marks(pl, xs, ys, style, shape="dot"):
    out = []
    for x, y in zip(xs, ys):
        q = pl.P(x, y)
        if shape == "dot":
            out.append("\\fill[%s] (%.3f,%.3f) circle (1.6pt);" % (style, q[0], q[1]))
        else:
            out.append("\\draw[%s, line width=0.7pt, fill=white] (%.3f,%.3f) circle (1.8pt);" % (style, q[0], q[1]))
    return out


def diagram(y0):
    """(a) the parameters on the chain, the light cones, and the clique tree."""
    b = []
    n = 7
    xs = [0.9 + 1.12 * k for k in range(2 * n - 1)]
    for k, (kind, i) in enumerate(names(n)):
        if kind == "h":
            b.append("\\node[circle, draw=ndBlue, fill=ndSky, line width=0.9pt, minimum size=0.62cm, inner sep=0pt, font=\\small] (v%d) at (%.3f,%.3f) {$h_{%d}$};"
                     % (k, xs[k], y0, i + 1))
        else:
            b.append("\\node[rounded corners=2pt, draw=ndRed, fill=ndRose, line width=0.9pt, minimum size=0.5cm, inner sep=1pt, font=\\small] (v%d) at (%.3f,%.3f) {$J_{%d}$};"
                     % (k, xs[k], y0, i + 1))
    for k in range(2 * n - 2):
        b.append("\\draw[ink] (v%d) -- (v%d);" % (k, k + 1))
    for k, j in graph(n):                                    # the other edges of the sparsity graph
        if j > k + 1:
            b.append("\\draw[faint] (v%d.north) to[out=%d, in=%d] (v%d.north);" % (k, 35 + 8 * (j - k), 145 - 8 * (j - k), j))
    b.append("\\node[note, anchor=south] at (%.3f,%.3f) {the correlative-sparsity graph: an arc wherever two parameters meet in a term of the fit};"
             % (0.5 * (xs[0] + xs[-1]), y0 + 1.3))
    cols = ["ndSky", "ndGold!35"]
    for c in range(n - 2):                                   # the light cones of the sites 2, ..., n-1
        lo, hi = xs[2 * c] - 0.32, xs[2 * c + 4] + 0.32
        yc = y0 - 0.62 - 0.26 * (c % 2)
        b.append("\\fill[%s, rounded corners=3pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (cols[c % 2], lo, yc - 0.09, hi, yc + 0.09))
        b.append("\\draw[ndInk!70, line width=0.5pt, rounded corners=3pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (lo, yc - 0.09, hi, yc + 0.09))
    yt = y0 - 1.95
    for c in range(n - 2):
        b.append("\\node[%s, minimum width=0.85cm, minimum height=0.5cm] (c%d) at (%.3f,%.3f) {$C_{%d}$};" % (GOLD, c, xs[2 * c + 2], yt, c + 2))
        b.append("\\draw[pointer, densely dotted] (%.3f,%.3f) -- (c%d.north);" % (xs[2 * c + 2], y0 - 0.62 - 0.26 * (c % 2) - 0.1, c))
        if c > 0:
            b.append("\\draw[ink] (c%d.east) -- node[below, font=\\scriptsize, inner sep=2pt] {$h_{%d},J_{%d},h_{%d}$} (c%d.west);" % (c - 1, c + 1, c + 1, c + 2, c))
    b.append("\\node[note, anchor=west, align=left] at (%.3f,%.3f) {the clique tree is\\\\ a chain: a PSD block\\\\ of order $21$ per core,\\\\ three parameters\\\\ on each bond};"
             % (xs[2 * n - 4] + 0.6, yt + 0.05))
    b.append("\\node[note, anchor=east, align=right] at (%.3f,%.3f) {the light cone\\\\ of the site $i$:\\\\ $C_i=\\{h_{i-1},J_{i-1},$\\\\ $h_i,J_i,h_{i+1}\\}$};"
             % (xs[2] - 0.6, yt + 0.05))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(a) the parameters of a chain of $n=%d$ qubits, the cliques, and the clique tree};"
             % (0.5 * (xs[0] + xs[-1]), yt - 0.95, n))
    return b


def main():
    out = compute()
    checks(out)
    b = diagram(6.95)

    # ---- (b) the order of the largest PSD block (log scale)
    pb = N.Plot((2, 25), (1, 3.2), 3.9, 3.0, origin=(0.6, 0.0))
    b += pb.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize largest PSD block}",
                 xticks=[(4, "$4$"), (8, "$8$"), (16, "$16$"), (24, "$24$")])
    b += log_ticks(pb, [10, 100, 1000], ["$10$", "$10^2$", "$10^3$"])
    nn = np.arange(3, 25)
    b.append(pb.curve(nn, np.log10(nn * (2 * nn + 1)), "line width=0.9pt, ndInk, dash pattern=on 3pt off 2pt"))
    b += marks(pb, NS_DENSE, [np.log10(max(out["dense"][n]["res"].block_sizes)) for n in NS_DENSE], "ndInk")
    b.append(pb.curve([3, 24], [np.log10(21)] * 2, "line width=0.9pt, ndBlue"))
    b += marks(pb, NS_SPARSE, [np.log10(max(out["sparse"][n]["res"].block_sizes)) for n in NS_SPARSE], "ndBlue")
    b.append("\\node[note, anchor=south east] at %s {dense: $n(2n+1)$};" % pb.pt(24, 3.05))
    b.append("\\node[note, anchor=south east, text=ndBlue] at %s {sparse: $21$, $n-2$ blocks};" % pb.pt(24, 1.42))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the size of the SDP};" % pb.pt(13.5, 0.55))

    # ---- (c) the solve times (log scale)
    pc = N.Plot((2, 25), (-1.5, 2.3), 3.9, 3.0, origin=(5.9, 0.0))
    b += pc.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize seconds (SCS)}",
                 xticks=[(4, "$4$"), (8, "$8$"), (16, "$16$"), (24, "$24$")])
    b += log_ticks(pc, [0.1, 1, 10, 100], ["$0.1$", "$1$", "$10$", "$100$"])
    for key, col, ns in [("dense", "ndInk", NS_DENSE), ("sparse", "ndBlue", NS_SPARSE)]:
        ts = [np.log10(out[key][n]["time"]) for n in ns]
        b.append(pc.curve(ns, ts, "line width=0.7pt, %s" % col))
        b += marks(pc, ns, ts, col)
    td, ts8, ts24 = out["dense"][8]["time"], out["sparse"][8]["time"], out["sparse"][24]["time"]
    b.append("\\node[note, anchor=west] (nd) at %s {dense,\\\\ $%.0f$ s at $n=8$};" % (pc.pt(9.6, 1.85), td))
    b.append("\\draw[pointer] (nd.west) to[bend right=20] %s;" % pc.pt(8.3, np.log10(td)))
    b.append("\\node[note, anchor=north, text=ndBlue] (ns) at %s {sparse: $%.1f$ s at $n=8$,\\\\ $%.0f$ s at $n=24$};" % (pc.pt(15.5, -0.45), ts8, ts24))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the solve times};" % pc.pt(13.5, -1.97))

    # ---- (d) the estimate for n = 24
    s = out["sparse"][N_SHOW]
    pd = N.Plot((0.5, N_SHOW + 0.5), (0, 1.5), 4.1, 3.0, origin=(11.2, 0.0))
    b += pd.axes(xlabel="{\\scriptsize site}", ylabel="{\\scriptsize parameter}",
                 xticks=[(1, "$1$"), (12, "$12$"), (24, "$24$")], yticks=[(0.5, "$0.5$"), (1, "$1$")])
    nm = names(N_SHOW)
    for kind, col, off in [("h", "ndBlue", 0.0), ("J", "ndRed", 0.5)]:
        ks = [k for k, t in enumerate(nm) if t[0] == kind]
        sites = [nm[k][1] + 1 + off for k in ks]
        b.append(pd.curve(sites, s["z0"][ks], "line width=0.4pt, %s!50" % col))
        b += marks(pd, sites, s["z0"][ks], col, "open")
        b += marks(pd, sites, s["x"][ks], col)
    err = np.abs(s["x"] - s["z0"]).max()
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndBlue] at %s {$h_i$};" % pd.pt(24.6, s["z0"][-1]))
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndRed] at %s {$J_i$};" % pd.pt(24.6, s["z0"][-2]))
    b += marks(pd, [2.0], [1.42], "ndInk", "open")
    b.append("\\node[font=\\tiny, anchor=west] at %s {true};" % pd.pt(2.6, 1.42))
    b += marks(pd, [8.5], [1.42], "ndInk")
    b.append("\\node[font=\\tiny, anchor=west] at %s {first pseudo-moments};" % pd.pt(9.1, 1.42))
    b.append("\\node[note] at %s {within $%.3f$ of the truth};" % (pd.pt(12.5, 0.17), err))
    b.append("\\node[font=\\small, anchor=north] at %s {(d) the estimate, $n=%d$};" % (pd.pt(12.5, -0.24), N_SHOW))
    N.write("ch11_chains", b)
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    out = main()
    for key in ("dense", "sparse"):
        for n, s in out[key].items():
            res = s["res"]
            print("%-6s n=%2d  bound %.9f  gap %.1e  status %s  block %4d x %2d  error %.4f  %.1f s"
                  % (key, n, res.value, res.gap(s["x"]), res.status, max(res.block_sizes), len(res.block_sizes),
                     np.abs(s["x"] - s["z0"]).max(), s["time"]))
    ex = out["exact"]
    print("exact data, n=8: bound %.1e, error %.1e" % (ex["res"].value, np.abs(ex["x"] - ex["z0"]).max()))
    print("total %.0f s; written figures/ch11_chains.tex" % (time.perf_counter() - t0))
