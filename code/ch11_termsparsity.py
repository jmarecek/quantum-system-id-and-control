"""ch11_termsparsity.py -- term sparsity in Hamiltonian identification: when it pays, and when not.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi) and cvxpy with
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.  Uses code/ch11_chains.py for the chain.

Term sparsity (TSSOS) keeps in the Gram matrix of the certificate only the
pairs of basis monomials whose product is a term of the objective, or reached
from the terms by s steps of support extension (s = ts_order); the blocks are
the cliques of the resulting graph, after a chordal ("MD") or a block closure.

(a) The three-level problem of eq. (ch11:Vunknown): the drift
    diag(0, 0.515916, 1), the coupling V(z) with <1|V|2> = z_1, <2|V|3> = z_2
    (true values 1/sqrt 2 and 1), a constant control lambda = 1 for T = 0.5,
    so that the Magnus series stops at Lambda = -iT(H_0 + V(z)), and the
    exponentials e^{+-Lambda/2} truncated after the terms of order p = 4.
    The tomographic estimate of U* carries Gaussian errors of standard
    deviation 0.01 on every real and imaginary part.  The objective
    ||e^{Lambda/2} - e^{-Lambda/2} U*||^2 has degree 8 in z and all 45
    monomials of degree <= 8 in two variables, so every product of two of the
    15 basis monomials of degree <= 4 is a term: the term-sparsity graph is
    complete, and at every ts_order (MD or block) the relaxation is the dense
    one, one block of order 15.  The bound 1.70197e-3 is attained: the first
    pseudo-moments give z = (0.7151, 0.9929), the global minimiser, which a
    grid search with a local polish confirms.  The distance to the truth is
    the noise of the tomography, 0.008; with exact data the truncation alone
    leaves 4e-4.
(b) What is true instead: term sparsity pays when there are many unknowns
    and few interactions.  The chain of n = 6 qubits of code/ch11_chains.py
    (11 unknowns, objective of degree 4): the order of the largest Gram block
    and the number of blocks against ts_order, for the chordal (MD) and the
    block closures, with the dense (78) and the correlatively sparse (21)
    relaxations for reference.  MD stays at blocks of order 15; the block
    closure reaches the dense block at the second step, and its graph stops
    changing at the third.
(c) The bounds against ts_order, minus the dense bound: within the SCS
    tolerance (2e-8) every step of either closure already gives the dense
    bound, so the hierarchy is flat; and the estimates read off the first
    pseudo-moments agree to 1e-6.
Every SCS status is "optimal" (eps = 1e-8).  Writes figures/ch11_termsparsity.tex.
"""
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp
import needham as N
import ch11_chains as CH
from ch13_qcpop_vs_grape import H0, V as V13, Ttot

Z_TRUE = np.array([V13[0, 1], V13[1, 2]])                     # 1/sqrt 2 and 1
LAM, P, SIGMA = 1.0, 4, 0.01
N_CHAIN = 6
OPTS = dict(solver="SCS", eps=1e-8, max_iters=300000)


def coupling(z):
    V = np.zeros((3, 3), dtype=object)
    V[0, 1] = V[1, 0] = z[0]
    V[1, 2] = V[2, 1] = z[1]
    return V


def matmul(A, B):
    n = A.shape[0]
    C = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            C[i, j] = sum((A[i, k] * B[k, j] for k in range(1, n)), A[i, 0] * B[0, j])
    return C


def halves(z, p=P):
    """cos and sin series of M = (T/2)(H_0 + lambda V(z)) to order p: e^{+-Lambda/2} = C -+ iS."""
    M = (Ttot / 2) * (H0.astype(object) + LAM * coupling(z))
    C = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            C[i, j] = float(i == j) + 0.0 * M[i, j]
    S = 0.0 * M
    Mk, fact = None, 1.0
    for k in range(1, p + 1):
        Mk = M if Mk is None else matmul(Mk, M)
        fact *= k
        if k % 2 == 0:
            C = C + ((-1) ** (k // 2) / fact) * Mk
        else:
            S = S + ((-1) ** (k // 2) / fact) * Mk
    return C, S


def residual(z, U):
    """e^{Lambda/2} - e^{-Lambda/2} U, as real and imaginary parts (object arrays)."""
    C, S = halves(z)
    Ur, Ui = U.real.astype(object), U.imag.astype(object)
    re = C - matmul(C, Ur) + matmul(S, Ui)
    im = -(S + matmul(C, Ui) + matmul(S, Ur))
    return re, im


def three_level_objective(U):
    zs = psp.polyvar(2)
    re, im = residual(zs, U)
    f = psp.Polynomial.zero(2)
    for e in list(re.ravel()) + list(im.ravel()):
        f = f + e ** 2
    return f


def objective_numeric(z, U):
    re, im = residual(np.asarray(z, dtype=float), U)
    return float(np.sum(re.astype(float) ** 2 + im.astype(float) ** 2))


def target(noise=SIGMA):
    U = expm(-1j * Ttot * (H0 + LAM * coupling(Z_TRUE).astype(float)))
    rng = np.random.default_rng(5)
    return U + noise * (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))


def global_search(U):
    """The independent check: a grid on [-2, 2]^2 and a Nelder-Mead polish."""
    g = np.linspace(-2, 2, 81)
    best = min((objective_numeric((a, b), U), a, b) for a in g for b in g)
    r = minimize(lambda x: objective_numeric(x, U), best[1:], method="Nelder-Mead",
                 options=dict(xatol=1e-10, fatol=1e-15, maxiter=4000))
    return r.x, r.fun


def run(f, **kw):
    t = time.perf_counter()
    res = psp.sos_lower_bound(f, **kw, **OPTS)
    return dict(res=res, time=time.perf_counter() - t, x=res.minimizer())


def hierarchy(f, order=None):
    """Dense, and both closures at ts_order = 1, 2, ... until the blocks stop changing."""
    out = {"dense": run(f, order=order, sparse=False)}
    for ts in ("MD", "block"):
        out[ts], s = [], 1
        while True:
            r = run(f, order=order, ts=ts, ts_order=s)
            if r["res"].ts_order < s:                       # stabilised one step earlier
                break
            out[ts].append(r)
            s += 1
    return out


def compute():
    U = target()
    f = three_level_objective(U)
    three = hierarchy(f)
    three["U"], three["f"] = U, f
    three["search"] = global_search(U)
    U0 = target(0.0)
    three["exact"] = run(three_level_objective(U0), sparse=False)
    z0, rows, d = CH.instance(N_CHAIN)
    fc = CH.objective(N_CHAIN, rows, d)
    chain = hierarchy(fc, order=2)
    chain["cs"] = run(fc, order=2, sparse=True)
    chain["z0"], chain["rows"], chain["d"] = z0, rows, d
    return dict(three=three, chain=chain)


def checks(out):
    th = out["three"]
    f, U = th["f"], th["U"]
    assert len(f.terms) == 45 and f.degree() == 8              # every monomial of degree <= 8 in (z_1, z_2)
    for z in (Z_TRUE, np.array([0.3, -1.1])):                  # the polynomial against the numerical objective
        assert abs(f(z) - objective_numeric(z, U)) < 1e-12
    for p in (4, 10):                                           # the split is a truncation of exp(Lambda/2) exp(Lambda/2)^{-1} U
        C, S = halves(Z_TRUE, p)
        Uh = (C.astype(float) + 1j * S.astype(float)).conj()
        exact = expm(-0.5j * Ttot * (H0 + LAM * coupling(Z_TRUE).astype(float)))
        assert np.abs(Uh - exact).max() < (2e-4 if p == 4 else 1e-11)
    xs, fs = th["search"]
    dense = th["dense"]["res"]
    for r in [th["dense"]] + th["MD"] + th["block"]:           # complete graph: one block of 15, the same bound
        assert r["res"].status == "optimal"
        assert r["res"].block_sizes == [15]
        assert abs(r["res"].value - dense.value) < 1e-9
        assert np.abs(r["x"] - xs).max() < 1e-5
    assert len(th["MD"]) == 1 and len(th["block"]) == 1
    assert -1e-8 < fs - dense.value < 1e-8                     # the bound is attained, by the grid search too
    assert abs(dense.gap(th["dense"]["x"])) < 1e-8
    assert 0.005 < np.abs(xs - Z_TRUE).max() < 0.01
    ex = th["exact"]
    assert ex["res"].status == "optimal" and np.abs(ex["x"] - Z_TRUE).max() < 1e-3
    ch = out["chain"]
    dense = ch["dense"]["res"]
    assert max(dense.block_sizes) == 78 and max(ch["cs"]["res"].block_sizes) == 21
    for r in [ch["dense"], ch["cs"]] + ch["MD"] + ch["block"]:
        assert r["res"].status == "optimal"
        assert abs(r["res"].value - dense.value) < 2e-8
        assert np.abs(r["x"] - ch["dense"]["x"]).max() < 1e-6
        assert abs(r["res"].gap(r["x"])) < 2e-8
    assert max(max(r["res"].block_sizes) for r in ch["MD"]) == 15
    assert max(ch["block"][-1]["res"].block_sizes) == 78 and len(ch["block"]) == 3
    xl, fl = CH.fit(ch["rows"], ch["d"], ch["z0"])
    assert abs(fl - dense.value) < 1e-7 and np.abs(xl - ch["dense"]["x"]).max() < 1e-4


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
            out.append("\\draw[%s, line width=0.7pt, fill=white] (%.3f,%.3f) circle (1.8pt);" % (style, q[0], q[1]))
    return out


def panel_three(out, x0, y0):
    """(a) the support of the three-level objective and the complete term-sparsity graph."""
    th = out["three"]
    b = []
    h = 0.42
    P_ = lambda a1, a2: (x0 + h * a1, y0 + h * a2)
    for k in range(9):                                         # the exponent grid
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (*P_(0, k), *P_(8 - k, k)))
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (*P_(k, 0), *P_(k, 8 - k)))
    b.append("\\draw[faint, ->, >=stealth] (%.3f,%.3f) -- (%.3f,%.3f) node[right, font=\\scriptsize] {$\\deg_{z_1}$};" % (*P_(0, 0), *P_(9, 0)))
    b.append("\\draw[faint, ->, >=stealth] (%.3f,%.3f) -- (%.3f,%.3f) node[above, font=\\scriptsize] {$\\deg_{z_2}$};" % (*P_(0, 0), *P_(0, 9)))
    b.append("\\fill[ndSky, fill opacity=0.6] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- cycle;" % (*P_(0, 0), *P_(4, 0), *P_(0, 4)))
    for e in th["f"].terms:
        b.append("\\fill[ndInk] (%.3f,%.3f) circle (1.5pt);" % P_(*e))
    for a1 in range(5):
        for a2 in range(5 - a1):
            b.append("\\draw[ndBlue, line width=0.7pt] (%.3f,%.3f) circle (2.6pt);" % P_(a1, a2))
    for k in (0, 4, 8):
        b.append("\\node[font=\\tiny, below] at (%.3f,%.3f) {$%d$};" % (*P_(k, 0), k))
        b.append("\\node[font=\\tiny, left] at (%.3f,%.3f) {$%d$};" % (*P_(0, k), k))
    b.append("\\node[note, anchor=west, align=left] at (%.3f,%.3f) {every one of the $45$ monomials\\\\ of degree $\\le8$ is a term};" % (x0 + 2.0, y0 + 3.3))
    b.append("\\node[note, anchor=west, align=left, text=ndBlue] at (%.3f,%.3f) {every product of two of the\\\\ $15$ basis monomials (ringed)\\\\ is a term: the graph is complete,\\\\ one block of $15$ at every $s$};"
             % (x0 + 2.0, y0 + 2.2))
    # the ladder
    lx, ly = x0 + 0.1, y0 + 4.45
    for k, (e, lab) in enumerate(zip([0.0, 0.515916, 1.0], ["1", "2", "3"])):
        yy = ly + 1.0 * e
        b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f) node[right, font=\\scriptsize] {$|%s\\rangle$};" % (lx + 0.75 * k, yy, lx + 0.75 * k + 0.5, yy, lab))
    for k, lab in enumerate(["z_1", "z_2"]):
        e0, e1 = [0.0, 0.515916, 1.0][k], [0.0, 0.515916, 1.0][k + 1]
        b.append("\\draw[<->, >=stealth, ndRed, line width=0.9pt] (%.3f,%.3f) -- node[above left=-1pt, font=\\scriptsize, text=ndRed] {$%s$} (%.3f,%.3f);"
                 % (lx + 0.75 * k + 0.35, ly + e0 + 0.05, lab, lx + 0.75 * (k + 1) + 0.15, ly + e1 - 0.05))
    dense = th["dense"]
    b.append("\\node[note, anchor=west, align=left] at (%.3f,%.3f) {fit to $U^\\star$, $p=4$: the bound\\\\ $%.5f\\cdot10^{-3}$ at every $s$,\\\\ attained at $z=(%.4f,\\,%.4f)$;\\\\ the truth is $(%.4f,\\,%.4f)$};"
             % (lx + 2.6, ly + 0.95, dense["res"].value * 1e3, dense["x"][0], dense["x"][1], Z_TRUE[0], Z_TRUE[1]))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(a) three levels: nothing to gain};" % (x0 + 2.6, y0 - 0.5))
    return b


def main():
    out = compute()
    checks(out)
    b = panel_three(out, 0.6, 0.0)
    ch = out["chain"]
    sMD = [r["res"].ts_order for r in ch["MD"]]
    sBL = [r["res"].ts_order for r in ch["block"]]
    smax = max(sMD + sBL)

    # ---- (b) the largest Gram block and the number of blocks against ts_order (log scale)
    pb = N.Plot((0.5, smax + 0.5), (1, 2), 3.5, 4.0, origin=(7.6, 0.0))
    b += pb.axes(xlabel="{\\scriptsize $s$}", ylabel="{\\scriptsize largest Gram block}",
                 xticks=[(s, "$%d$" % s) for s in range(1, smax + 1)])
    for v, lab in [(10, "$10$"), (20, "$20$"), (50, "$50$"), (100, "$100$")]:
        q = pb.P(0.5, np.log10(v))
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[left, font=\\scriptsize, text=ndInk] {%s};" % (q[0] + 0.06, q[1], q[0] - 0.06, q[1], lab))
    dmax, cmax = max(ch["dense"]["res"].block_sizes), max(ch["cs"]["res"].block_sizes)
    b.append(pb.curve([0.5, smax + 0.5], [np.log10(dmax)] * 2, "line width=0.9pt, ndInk, dash pattern=on 3pt off 2pt"))
    b.append(pb.curve([0.5, smax + 0.5], [np.log10(cmax)] * 2, "line width=0.6pt, ndGreen, dash pattern=on 3pt off 2pt"))
    b.append("\\node[note, anchor=south east] at %s {dense: one block of $%d$};" % (pb.pt(smax + 0.5, np.log10(dmax)), dmax))
    b.append("\\node[note, anchor=north east, text=ndGreen] at %s {correlative: $%d\\times%d$};"
             % (pb.pt(smax + 0.5, np.log10(cmax) - 0.01), len(ch["cs"]["res"].block_sizes), cmax))
    for key, col, shape, pos in [("MD", "ndBlue", "dot", "below"), ("block", "ndRed", "square", "above left")]:
        ss = [r["res"].ts_order for r in ch[key]]
        ys = [np.log10(max(r["res"].block_sizes)) for r in ch[key]]
        b.append(pb.curve(ss, ys, "line width=0.7pt, %s" % col))
        b += marks(pb, ss, ys, col, shape)
        for s, y, r in zip(ss, ys, ch[key]):
            q = pb.P(s, y)
            at = "below" if max(r["res"].block_sizes) == dmax else pos
            b.append("\\node[%s, font=\\tiny, text=%s, inner sep=2.5pt] at (%.3f,%.3f) {$%d$};" % (at, col, q[0], q[1], len(r["res"].block_sizes)))
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndBlue] at %s {MD (chordal) closure};" % pb.pt(1.6, 1.05))
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndRed] at %s {block closure};" % pb.pt(2.0, 1.62))
    b.append("\\node[note, anchor=west] at %s {the numbers: how many blocks};" % pb.pt(1.0, 1.45))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(b) a chain of $%d$ qubits: the blocks};" % (pb.o[0] + 1.75, -0.5, N_CHAIN))

    # ---- (c) the bounds minus the dense bound, in units of 1e-8
    pc = N.Plot((0.5, smax + 0.5), (-3, 3), 3.5, 4.0, origin=(12.75, 0.0))
    b += pc.axes(xlabel="{\\scriptsize $s$}", ylabel="{\\scriptsize bound $-$ dense bound}",
                 xticks=[(s, "$%d$" % s) for s in range(1, smax + 1)],
                 yticks=[(-2, "$-2\\cdot10^{-8}$"), (0, "$0$"), (2, "$2\\cdot10^{-8}$")])
    b.append("\\fill[ndSky, fill opacity=0.6] %s rectangle %s;" % (pc.pt(0.5, -2), pc.pt(smax + 0.5, 2)))
    b.append(pc.curve([0.5, smax + 0.5], [0, 0], "faint"))
    dv = ch["dense"]["res"].value
    for key, col, shape in [("MD", "ndBlue", "dot"), ("block", "ndRed", "square")]:
        ss = [r["res"].ts_order for r in ch[key]]
        ys = [(r["res"].value - dv) * 1e8 for r in ch[key]]
        b += marks(pc, ss, ys, col, shape)
    b.append("\\node[note, anchor=north] at %s {the SCS tolerance};" % pc.pt(0.5 + smax / 2, -2.1))
    b.append("\\node[note, anchor=south, align=center] at %s {every step of either closure\\\\ gives the dense bound $%.6f$};" % (pc.pt(0.5 + smax / 2, 2.1), dv))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(c) the chain: the bounds};" % (pc.o[0] + 1.75, -0.5))
    N.write("ch11_termsparsity", b)
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    out = main()
    th = out["three"]
    print("three levels: terms %d, search min %.9e at %s" % (len(th["f"].terms), th["search"][1], np.round(th["search"][0], 5)))
    for key in ("dense", "MD", "block"):
        for r in ([th[key]] if key == "dense" else th[key]):
            print("  %-6s s=%s  bound %.9e  status %s  blocks %s  x %s  %.1f s" % (key, r["res"].ts_order, r["res"].value, r["res"].status,
                                                                             r["res"].block_sizes, np.round(r["x"], 5), r["time"]))
    print("  exact data: x %s, error %.1e" % (np.round(th["exact"]["x"], 6), np.abs(th["exact"]["x"] - Z_TRUE).max()))
    ch = out["chain"]
    for key in ("dense", "cs", "MD", "block"):
        for r in ([ch[key]] if key in ("dense", "cs") else ch[key]):
            print("chain %-6s s=%s  bound %.10f  status %s  largest %d  blocks %d  %.1f s" % (key, r["res"].ts_order, r["res"].value, r["res"].status,
                                                                                    max(r["res"].block_sizes), len(r["res"].block_sizes), r["time"]))
    print("total %.0f s; written figures/ch11_termsparsity.tex" % (time.perf_counter() - t0))
