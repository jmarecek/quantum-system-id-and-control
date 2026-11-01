"""ch11_lifting.py -- a certified prediction band over a long record, by state lifting.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi) and cvxpy with
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

An identified model of the populations of a qubit, observed stroboscopically:
p_k, the population of the state the qubit was prepared in, and 1 - p_k form a
two-state Markov chain whose transition probabilities depend on a parameter
x_k in [-1, 1] that is known only to lie in that interval at each step (a
slowly fluctuating flux, say):
    stay   a(x) = 0.95 - 0.20 x^2,     return   b(x) = 0.05 - 0.05 x^2,
    p_k = b(x_k) + (a(x_k) - b(x_k)) p_{k-1},   p_1 = a(x_1)
(the controlled chain of Balada Gaggioli, Henrion and Korda, arXiv:2604.17563,
Sec. 7.1).  Without disturbance the populations relax towards 1/2 with the
factor a - b = 0.9 per step.  The population after n steps is a polynomial of
degree 2n in the n parameters.  Since a - b > 0 and both b and a - b decrease in
x^2, p_n is maximised by x = 0 at every step and minimised by x = +-1, so the
exact band is
    0.75^n <= p_n <= 1/2 + 0.9^n/2.
A dense moment relaxation of a polynomial of degree 2n in n variables needs a
Gram matrix of order binom(2n, n): 20, 70, 252, ..., 1.4e11 at n = 20.  Lifting
the states s_k = (p_k, 1 - p_k) to new variables, tied to the parameters by
the equalities s_k = F_k(s_{k-1}, x_k), makes the problem chain-sparse: SL-chord
uses the cliques {s_{k-1}, x_k, s_k} of 2 + 1 + 2 = 5 variables, PSD blocks of
order binom(7, 2) = 21 at order 2 whatever n; SL-push certifies one stage at a
time on {s_{k-1}, x_k}, blocks of order binom(5, 2) = 10 at order 2 and
binom(6, 3) = 20 at order 3.

(a) The lifting as a chain: the parameters x_k, the lifted states s_k and the
    overlapping cliques.
(b) The certified upper bounds against n (SL-chord at order 2 and SL-push at
    order 3 are exact to 1e-6; SL-push at order 2 is loose and exceeds 1 for
    large n, and at n = 8 SCS returns "optimal_inaccurate" with a value below
    the true maximum, which is no bound at all), the certified lower bounds of
    SL-push at order 3 for n <= 12, and the exact band.
(c) The order of the largest PSD block against n: dense, solved for n = 3
    and 4 only, against the constant lifted blocks.
(d) The solve times with SCS on this machine.
All bounds are checked against the closed form, all SCS statuses except the
one noted are "optimal", and for the upper bound the parameters recovered from the
pseudo-moments are x = 0 with zero gap, which certifies global optimality.
For the lower bound the first moments vanish, since x and -x are both
optimal; the bound is attained at x = (1, ..., 1).
Writes figures/ch11_lifting.tex.
"""
import os
import sys
import time
from scipy.special import comb as _comb


def comb(n, k):
    return int(_comb(n, k, exact=True))

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp
import needham as N

STAY = lambda x: 0.95 - 0.20 * x ** 2
BACK = lambda x: 0.05 - 0.05 * x ** 2
NS = [4, 6, 8, 10, 12, 16, 20]
NS_MIN = [4, 6, 8, 10, 12]
NS_PUSH3 = [4, 6, 8, 10, 12, 16]
NS_DENSE = [3, 4]
PUSH2_FAIL = 8                                           # SCS returns optimal_inaccurate there
OPTS = dict(solver="SCS", eps=1e-7, max_iters=100000)


def first(s, x):
    return [STAY(x), 1 - STAY(x)]


def step(s, x):
    return [s[0] * STAY(x) + s[1] * BACK(x), s[0] * (1 - STAY(x)) + s[1] * (1 - BACK(x))]


def last(s, x):
    return s[0] * STAY(x) + s[1] * BACK(x)


def chain(n):
    return [first] + [step] * (n - 2) + [last]


def population(xs):
    """p_n for a numerical sequence of parameters."""
    p = STAY(xs[0])
    for x in xs[1:]:
        p = BACK(x) + (STAY(x) - BACK(x)) * p
    return p


def upper(n):
    return 0.5 + 0.5 * 0.9 ** n


def lower(n):
    return 0.75 ** n


def solve(n, method, order, sense):
    t = time.perf_counter()
    res = psp.composition_lower_bound(chain(n), box=1.0, order=order, sense=sense, method=method, **OPTS)
    return res, time.perf_counter() - t


def dense(n):
    """The dense moment relaxation of max p_n over the box, at the order n."""
    xs = psp.polyvar(n)
    s = first([], xs[0])
    for i in range(1, n - 1):
        s = step(s, xs[i])
    p = last(s, xs[n - 1])
    t = time.perf_counter()
    res = psp.sos_lower_bound(-p, ineqs=[1 - x ** 2 for x in xs], order=n, sparse=False, **OPTS)
    return res, time.perf_counter() - t


def compute():
    out = {"chord": {}, "push2": {}, "push3": {}, "push3min": {}, "dense": {}}
    for n in NS:
        out["chord"][n] = solve(n, "chord", 2, "max")
        out["push2"][n] = solve(n, "push", 2, "max")
    for n in NS_PUSH3:
        out["push3"][n] = solve(n, "push", 3, "max")
    for n in NS_MIN:
        out["push3min"][n] = solve(n, "push", 3, "min")
    for n in NS_DENSE:
        out["dense"][n] = dense(n)
    return out


def checks(out):
    rng = np.random.default_rng(0)
    for n in (3, 8, 20):                                   # the closed form of the band
        assert abs(population(np.zeros(n)) - upper(n)) < 1e-12
        assert abs(population(np.ones(n)) - lower(n)) < 1e-12 and abs(population(-np.ones(n)) - lower(n)) < 1e-12
        for _ in range(2000):
            p = population(rng.uniform(-1, 1, n))
            assert lower(n) - 1e-12 <= p <= upper(n) + 1e-12
    for key in ("chord", "push3"):                         # exact certified upper bounds, x = 0 recovered
        for n, (res, _) in out[key].items():
            assert res.status == "optimal", (key, n, res.status)
            assert abs(res.value - upper(n)) < 1e-6, (key, n, res.value)
            x = res.minimizer()
            assert np.abs(x).max() < 1e-6 and abs(res.gap(x)) < 1e-6
            assert all(r == 2 for r in res.ranks)
    for n, (res, _) in out["chord"].items():
        assert max(res.block_sizes) == comb(7, 2)
    for n, (res, _) in out["push3"].items():
        assert max(res.block_sizes) == comb(6, 3)
    for n, (res, _) in out["push2"].items():           # SL-push at order 2: loose, and SCS fails at n = 8
        assert max(res.block_sizes) == comb(5, 2)
        if n == PUSH2_FAIL:
            assert res.status == "optimal_inaccurate" and res.value < upper(n)   # not a bound at all
        else:
            assert res.status == "optimal" and res.value >= upper(n) - 1e-6
    assert out["push2"][20][0].value > 1 and out["push2"][4][0].value - upper(4) < 1e-4
    for n, (res, _) in out["push3min"].items():        # the lower envelope; first moments vanish by symmetry
        assert res.status == "optimal" and abs(res.value - lower(n)) < 1e-6, (n, res.value)
        assert np.abs(res.minimizer()).max() < 1e-6
    for n, (res, _) in out["dense"].items():           # dense relaxation: exact, Gram block binom(2n, n)
        assert res.status == "optimal" and abs(-res.value - upper(n)) < 1e-6
        assert max(res.block_sizes) == comb(2 * n, n)


def log_axis(pl, values, labels, x=True):
    out = []
    for v, lab in zip(values, labels):
        if x:
            p = pl.P(np.log10(v), pl.ylim[0])
            out.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[below, font=\\scriptsize, text=ndInk] {%s};"
                       % (p[0], p[1] + 0.06, p[0], p[1] - 0.06, lab))
        else:
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
        elif shape == "square":
            out.append("\\draw[%s, line width=0.7pt, fill=white] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                       % (style, q[0] - 0.055, q[1] - 0.055, q[0] + 0.055, q[1] + 0.055))
        else:
            out.append("\\draw[%s, line width=0.8pt] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
                       % (style, q[0] - 0.06, q[1] - 0.06, q[0] + 0.06, q[1] + 0.06, q[0] - 0.06, q[1] + 0.06, q[0] + 0.06, q[1] - 0.06))
    return out


def diagram(y0):
    """(a) the chain of stages and the overlapping cliques {s_{k-1}, x_k, s_k}."""
    b = []
    xs_pos = [1.4, 4.2, 7.0, 11.2, 14.0]
    labels = ["1", "2", "3", "n-1", "n"]
    sy, xy = y0 + 0.75, y0 - 0.45                       # rows of the states and of the parameters
    # cliques: {x_1, s_1}, {s_1, x_2, s_2}, {s_2, x_3, s_3}, {..., x_{n-1}, s_{n-1}}, {s_{n-1}, x_n, p_n}
    spans = [(xs_pos[0] - 0.45, xs_pos[0] + 1.6), (xs_pos[0] + 0.65, xs_pos[1] + 1.6), (xs_pos[1] + 0.65, xs_pos[2] + 1.6),
             (xs_pos[3] - 0.95, xs_pos[3] + 1.6), (xs_pos[3] + 0.65, xs_pos[4] + 1.6)]
    for k, (xa, xb) in enumerate(spans):
        col = "ndSky" if k % 2 == 0 else "ndGold!30"
        b.append("\\fill[%s, fill opacity=0.7, rounded corners=9pt] (%.2f,%.2f) rectangle (%.2f,%.2f);"
                 % (col, xa, xy - 0.42 - 0.06 * (k % 2), xb, sy + 0.42 + 0.06 * (k % 2)))
    for i, (x0, lab) in enumerate(zip(xs_pos, labels)):
        b.append("\\node[draw=ndInk, fill=white, circle, inner sep=1.2pt, minimum size=0.62cm, font=\\small] (x%d) at (%.2f,%.2f) {$x_{%s}$};"
                 % (i, x0, xy, lab))
        if i < 4:
            b.append("\\node[draw=ndBlue, line width=0.9pt, fill=white, rounded corners=3pt, inner sep=2pt, minimum height=0.6cm, font=\\small] (s%d) at (%.2f,%.2f) {$s_{%s}$};"
                     % (i, x0 + 1.1, sy, lab))
    b.append("\\node[draw=ndRed, line width=0.9pt, fill=white, rounded corners=3pt, inner sep=2pt, minimum height=0.6cm, font=\\small] (p) at (%.2f,%.2f) {$p_n$};"
             % (xs_pos[4] + 1.1, sy))
    for i in range(5):
        tgt = "s%d" % i if i < 4 else "p"
        b.append("\\draw[->, >=stealth, line width=0.6pt, ndInk!70] (x%d) -- (%s);" % (i, tgt))
        if 0 < i:
            src = "s%d" % (i - 1)
            if i == 3:
                b.append("\\draw[->, >=stealth, line width=0.6pt, ndInk!70, dash pattern=on 2pt off 2pt] (s1) -- (s2.west);")
                b.append("\\node[font=\\small] at (%.2f,%.2f) {$\\cdots$};" % ((xs_pos[2] + xs_pos[3]) / 2 + 0.55, sy))
                continue
            b.append("\\draw[->, >=stealth, line width=0.9pt, ndBlue] (%s) -- (%s);" % (src, tgt))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {$s_k=F(s_{k-1},x_k)$: the populations $(p_k,1-p_k)$ after the step $k$};" % (xs_pos[0] - 0.4, sy + 0.85))
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {each wash is one clique $\\{s_{k-1},x_k,s_k\\}$ of five variables: one PSD block of order $21$, whatever $n$};"
             % ((xs_pos[0] + xs_pos[4]) / 2 + 0.5, xy - 0.55))
    b.append("\\node[font=\\small, anchor=north] at (%.2f,%.2f) {(a) the record as a chain, with its states lifted to variables};"
             % ((xs_pos[0] + xs_pos[4]) / 2 + 0.5, xy - 1.05))
    return b


def main():
    out = compute()
    checks(out)
    b = diagram(6.3)

    # ---- (b) the certified bounds
    pb = N.Plot((2.5, 21), (0, 1.0), 4.3, 3.2, origin=(0.6, 0.0))
    b += pb.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize population after $n$ steps}",
                 xticks=[(4, "$4$"), (10, "$10$"), (20, "$20$")], yticks=[(0.5, "$\\tfrac12$"), (1, "$1$")])
    nn = np.arange(3, 21)
    band = [pb.P(n, upper(n)) for n in nn] + [pb.P(n, lower(n)) for n in nn[::-1]]
    b.append("\\fill[ndSky, fill opacity=0.6] %s -- cycle;" % " -- ".join(N.pt(q) for q in band))
    b.append(pb.curve(nn, [upper(n) for n in nn], "line width=0.9pt, ndInk"))
    b.append(pb.curve(nn, [lower(n) for n in nn], "line width=0.9pt, ndInk, dash pattern=on 3pt off 2pt"))
    p2 = [(n, r.value) for n, (r, _) in out["push2"].items() if n != PUSH2_FAIL]
    p2in = [(n, v) for n, v in p2 if v <= 1.0]
    b += marks(pb, [n for n, _ in p2in], [v for _, v in p2in], "ndRed", "cross")
    for n, v in p2:
        if v > 1.0:
            q = pb.P(n, 1.0)
            b.append("\\draw[->, >=stealth, ndRed, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f) node[above, font=\\tiny, text=ndRed] {%.1f};"
                     % (q[0], q[1] - 0.12, q[0], q[1] + 0.12, v))
    qf = pb.P(PUSH2_FAIL, out["push2"][PUSH2_FAIL][0].value)
    b.append("\\draw[ndRed, line width=0.7pt, fill=white] (%.3f,%.3f) circle (1.8pt);" % (qf[0], qf[1]))
    b += marks(pb, list(out["push3"]), [r.value for r, _ in out["push3"].values()], "ndBlue", "square")
    b += marks(pb, list(out["chord"]), [r.value for r, _ in out["chord"].values()], "ndBlue")
    b += marks(pb, list(out["push3min"]), [r.value for r, _ in out["push3min"].values()], "ndGreen")
    b.append("\\node[note, anchor=south west] at %s {$\\frac12+\\frac12\\,0.9^n$};" % pb.pt(3.0, 0.88))
    b.append("\\node[note, anchor=south west] at %s {$0.75^n$};" % pb.pt(4.6, 0.32))
    keyb = [("dot", "ndBlue", "upper, SL-chord 2"), ("square", "ndBlue", "upper, SL-push 3"), ("cross", "ndRed", "upper, SL-push 2"),
            ("dot", "ndGreen", "lower, SL-push 3"), ("hollow", "ndRed", "SL-push 2 at $n=8$: SCS")]
    for j, (shape, col, lab) in enumerate(keyb):
        y = 0.49 - 0.075 * j
        if shape == "hollow":
            q = pb.P(11.2, y)
            b.append("\\draw[ndRed, line width=0.7pt, fill=white] (%.3f,%.3f) circle (1.8pt);" % (q[0], q[1]))
            b.append("\\node[font=\\tiny, anchor=north west, text=ndInk, align=left] at %s {%s\\\\ inaccurate, not a bound};" % (pb.pt(11.8, y + 0.035), lab))
        else:
            b += marks(pb, [11.2], [y], col, shape)
            b.append("\\node[font=\\tiny, anchor=west, text=ndInk] at %s {%s};" % (pb.pt(11.8, y), lab))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the certified band};" % pb.pt(11.75, -0.24))

    # ---- (c) the order of the largest PSD block (log scale)
    pc = N.Plot((2.5, 21), (0, 12), 4.3, 3.2, origin=(6.3, 0.0))
    b += pc.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize order of the largest PSD block}",
                 xticks=[(4, "$4$"), (10, "$10$"), (20, "$20$")])
    b += log_axis(pc, [1e1, 1e4, 1e8, 1e12], ["$10$", "$10^4$", "$10^8$", "$10^{12}$"], x=False)
    nd = np.arange(3, 21)
    b.append(pc.curve(nd, [np.log10(comb(2 * n, n)) for n in nd], "line width=0.9pt, ndInk, dash pattern=on 3pt off 2pt"))
    b += marks(pc, NS_DENSE, [np.log10(comb(2 * n, n)) for n in NS_DENSE], "ndInk")
    for val, col in [(21, "ndBlue"), (20, "ndBlue!55"), (10, "ndRed")]:
        b.append(pc.curve([3, 20], [np.log10(val)] * 2, "line width=0.9pt, %s" % col))
    b.append("\\node[font=\\tiny, anchor=south east, text=ndBlue] at %s {SL-chord $21$, SL-push 3 $20$};" % pc.pt(20.8, 1.42))
    b.append("\\node[font=\\tiny, anchor=north east, text=ndRed] at %s {SL-push 2 $10$};" % pc.pt(20.8, 0.92))
    b.append("\\node[note, anchor=south west] at %s {lifted:\\\\ constant in $n$};" % pc.pt(12.5, 2.4))
    b.append("\\node[note, anchor=west] at %s {dense: $\\binom{2n}{n}$,\\\\ $1.4\\cdot10^{11}$ at $n=20$};" % pc.pt(3.6, 9.6))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the size of the SDP};" % pc.pt(11.75, -2.0))

    # ---- (d) the solve times (log scale)
    pd = N.Plot((2.5, 21), (-1, 2.5), 4.3, 3.2, origin=(12.0, 0.0))
    b += pd.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize seconds (SCS)}",
                 xticks=[(4, "$4$"), (10, "$10$"), (20, "$20$")])
    b += log_axis(pd, [0.1, 1, 10, 100], ["$0.1$", "$1$", "$10$", "$100$"], x=False)
    for key, col, shape in [("chord", "ndBlue", "dot"), ("push3", "ndBlue", "square"), ("push2", "ndRed", "cross")]:
        ns = [n for n in out[key] if not (key == "push2" and n == PUSH2_FAIL)]
        ts = [np.log10(out[key][n][1]) for n in ns]
        b.append(pd.curve(ns, ts, "line width=0.6pt, %s" % col))
        b += marks(pd, ns, ts, col, shape)
    ns = list(out["dense"])
    ts = [np.log10(t) for _, t in out["dense"].values()]
    b.append(pd.curve(ns, ts, "line width=0.9pt, ndInk, dash pattern=on 3pt off 2pt"))
    b += marks(pd, ns, ts, "ndInk")
    tchord = out["chord"][20][1]
    b.append("\\node[note, anchor=north west] at %s {dense, solved\\\\ for $n=3,4$ only};" % pd.pt(3.2, 2.45))
    b.append("\\node[note, anchor=east, text=ndBlue] (tc) at %s {SL-chord at\\\\ $n=20$: %.0f s};" % (pd.pt(19.6, 0.15), tchord))
    b.append("\\draw[pointer] (tc.east) to[bend right=20] %s;" % pd.pt(20.0, np.log10(tchord) - 0.1))
    key = [("dot", "ndBlue", "SL-chord, order 2"), ("square", "ndBlue", "SL-push, order 3"), ("cross", "ndRed", "SL-push, order 2"), ("dot", "ndInk", "dense")]
    for j, (shape, col, lab) in enumerate(key):
        y = -0.3 - 0.2 * j
        b += marks(pd, [5.2], [y], col, shape)
        b.append("\\node[font=\\tiny, anchor=west, text=ndInk] at %s {%s};" % (pd.pt(5.8, y), lab))
    b.append("\\node[font=\\small, anchor=north] at %s {(d) the solve times};" % pd.pt(11.75, -1.65))
    N.write("ch11_lifting", b)
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    out = main()
    for key in ("chord", "push2", "push3", "push3min", "dense"):
        for n, (res, t) in out[key].items():
            print("%-8s n=%2d  value %.6f  status %s  block %d  %.1f s" % (key, n, res.value if key != "dense" else -res.value,
                                                                     res.status, max(res.block_sizes), t))
    print("total %.0f s; written figures/ch11_lifting.tex" % (time.perf_counter() - t0))
