"""ch11_sweep.py -- is an identified noise model physical at every bias?

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi) and cvxpy with
the solver SCS.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

Correlated dephasing on a chain of n qubits: Lindblad operators Z_1, ..., Z_n
and an n x n Kossakowski matrix Gamma(x), tridiagonal, whose entries are
polynomials in a bias x in [-1, 1], as a fit to measurements would give:
    local rates           Gamma_ii(x)     = kappa(x) = 1 - 0.1 x,
    nearest neighbours    Gamma_i,i+1(x)  = c(x)     = 0.40 + 0.08 x^2 + b w(x),
where w(x) = prod_k (x - x_k) vanishes at the seven biases x_k = -1, -2/3, ..., 1
and b is chosen so that b max w = 0.20 on [-1, 1].  At the seven biases the
model is Gamma_i,i+1 = 0.40 + 0.08 x_k^2, and Gamma(x_k) is positive definite
(for n = 5 its least eigenvalue is 0.27, 0.31, 0.33, 0.31, 0.26, 0.18, 0.07).
Between -1 and -2/3, however, c(x) rises to 0.664, and since Gamma is
tridiagonal and Toeplitz its least eigenvalue is
    lambda_min(Gamma(x)) = kappa(x) - 2 |c(x)| cos(pi/(n + 1)),
which reaches -0.0607 at x = -0.896 for n = 5: the model is not physical there,
although every 2 x 2 principal block [[kappa, c], [c, kappa]] is positive
semidefinite at every bias, since |c| <= 0.664 < 0.9 <= kappa.

The sparse SOS relaxation of pysparsepmi (pmi_optimize, order 4, the
chordal decomposition of Zheng and Fantuzzi with one 2 x 2 clique per pair of
neighbours) bounds min_x lambda_min(Gamma(x)) over [-1, 1] from below:
    Gamma(x) - t I = S_0(x) + (1 - x^2) S_1(x),  S_0, S_1 sums of squares,
each split over the cliques {i, i+1}.  For n = 5 the bound is -0.06066, equal
to the true minimum, and the minimiser read off the pseudo-moments is
x = -0.896.  For a chordal (here path) pattern, a constant matrix is positive
semidefinite if and only if it is a sum of matrices supported on the cliques,
each positive semidefinite (Grone, Johnson, Sa and Wolkowicz 1984; Agler,
Helton, McCullough and Rodman 1988); the 2 x 2 blocks of such a sum share the
diagonal, which is why pairwise checks are not enough:
[[1, .75, 0], [.75, 1, .75], [0, .75, 1]] has both 2 x 2 blocks positive
definite and the least eigenvalue 1 - 0.75 sqrt(2) = -0.0607.

(a) lambda_min(Gamma(x)) for n = 5, the seven biases (all positive), the
    certified bound and the recovered minimiser.
(b) The chordal decomposition for n = 5, and the 3 x 3 example.
(c) The size of the certificate and the time to compute it against n: the
    dense relaxation has one PSD block of order 5n (the basis 1, x, ..., x^4 of
    order 4 times n rows), the chordal one n - 1 blocks of order 10.  Both
    bounds agree with the true minimum wherever both are computed (n <= 24);
    the times are those of SCS on the machine that ran this script.
Writes figures/ch11_sweep.tex.
"""
import os
import sys
import time
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp
import needham as N

XK = np.linspace(-1, 1, 7)                    # the measured biases
W = np.poly(XK)                               # w(x) = prod (x - x_k), coefficients high -> low
_GRID = np.linspace(-1, 1, 40001)
C_COEF = np.zeros(8)
C_COEF[-1], C_COEF[-3] = 0.40, 0.08
C_COEF = C_COEF + (0.20 / np.polyval(W, _GRID).max()) * W
K_COEF = np.array([-0.10, 1.0])               # kappa(x) = 1 - 0.1 x
ORDER = 4
SOLVE = dict(solver="SCS", eps=1e-8, max_iters=200000)
N_DENSE = (4, 6, 8, 12, 16, 20, 24)
N_SPARSE = (4, 6, 8, 12, 16, 20, 24, 32, 48)


def kappa(x):
    return np.polyval(K_COEF, x)


def corr(x):
    return np.polyval(C_COEF, x)


def lam_min(x, n):
    """Least eigenvalue of the tridiagonal Toeplitz Gamma(x), in closed form."""
    return kappa(x) - 2 * np.abs(corr(x)) * np.cos(np.pi / (n + 1))


def gamma_num(x, n):
    G = np.diag(np.full(n, kappa(x)))
    for i in range(n - 1):
        G[i, i + 1] = G[i + 1, i] = corr(x)
    return G


def gamma_poly(n):
    x, = psp.polyvar(1)

    def poly(coef):
        p = psp.Polynomial.zero(1)
        for k, a in enumerate(coef[::-1]):
            if a != 0:
                p = p + (a * x ** k if k else a)
        return p

    zero, c, k = psp.Polynomial.zero(1), poly(C_COEF), poly(K_COEF)
    E = [[zero for _ in range(n)] for _ in range(n)]
    for i in range(n):
        E[i][i] = k
        if i + 1 < n:
            E[i][i + 1] = E[i + 1][i] = c
    return psp.PolyMatrix(E, nvars=1), x


def certify(n, sparse):
    G, x = gamma_poly(n)
    t0 = time.time()
    res = psp.pmi_optimize(G, ineqs=[1 - x ** 2], order=ORDER, sparse=sparse, **SOLVE)
    return res, time.time() - t0


def clique_split(G):
    """Feasibility of G = sum_k E_k^T S_k E_k with PSD 2 x 2 blocks on the cliques {k, k+1}."""
    n = G.shape[0]
    S = [cp.Variable((2, 2), symmetric=True) for _ in range(n - 1)]
    M = 0
    for k, Sk in enumerate(S):
        E = np.zeros((2, n))
        E[0, k] = E[1, k + 1] = 1
        M = M + E.T @ Sk @ E
    prob = cp.Problem(cp.Minimize(0), [M == G] + [Sk >> 0 for Sk in S])
    prob.solve(solver="SCS", eps=1e-9)
    return prob.status


def checks():
    # the model at the measured biases, and between them
    assert np.allclose(corr(XK), 0.40 + 0.08 * XK ** 2, atol=1e-12)
    lk = lam_min(XK, 5)
    assert np.all(lk > 0.06) and abs(lk[-1] - 0.0686) < 1e-3
    for x in (XK[0], -0.896, 0.3, XK[-1]):
        assert abs(np.linalg.eigvalsh(gamma_num(x, 5))[0] - lam_min(x, 5)) < 1e-12
    lam = lam_min(_GRID, 5)
    xstar, lstar = _GRID[lam.argmin()], lam.min()
    assert -1 < xstar < -2 / 3 and abs(xstar + 0.896) < 1e-3 and abs(lstar + 0.06066) < 1e-4
    assert abs(corr(_GRID).max() - 0.664) < 1e-3
    assert np.all(kappa(_GRID) - np.abs(corr(_GRID)) > 0.2)          # every 2 x 2 block is PSD everywhere
    # the 3 x 3 example: PSD 2 x 2 blocks, indefinite matrix
    T = np.array([[1, 0.75, 0], [0.75, 1, 0.75], [0, 0.75, 1]])
    assert np.linalg.eigvalsh(T[:2, :2])[0] > 0 and abs(np.linalg.eigvalsh(T)[0] - (1 - 0.75 * np.sqrt(2))) < 1e-12
    # the clique decomposition exists where Gamma is PSD and not where it is indefinite
    assert clique_split(gamma_num(0.0, 5)) == "optimal"
    assert clique_split(gamma_num(xstar, 5)) == "infeasible"
    assert clique_split(T) == "infeasible"
    # the certificate for n = 5: tight, with the minimiser recovered from the pseudo-moments
    res, _ = certify(5, sparse=True)
    assert res.status == "optimal" and abs(res.value - lstar) < 1e-5
    assert res.cliques == [[0, 1], [1, 2], [2, 3], [3, 4]] and max(res.block_sizes) == 2 * (ORDER + 1)
    xr = float(np.ravel(res.minimizer())[0])
    assert abs(xr - xstar) < 1e-3
    return xstar, lstar, res.value, xr


def scaling():
    rows = {}
    for n in N_SPARSE:
        true = lam_min(_GRID, n).min()
        rs, ts = certify(n, sparse=True)
        assert rs.status == "optimal" and abs(rs.value - true) < 1e-5, (n, rs.value, true)
        assert max(rs.block_sizes) == 2 * (ORDER + 1)
        row = dict(true=true, sparse=rs.value, ts=ts, bs=max(rs.block_sizes))
        if n in N_DENSE:
            rd, td = certify(n, sparse=False)
            assert rd.status == "optimal" and abs(rd.value - rs.value) < 1e-5, (n, rd.value, rs.value)
            assert max(rd.block_sizes) == n * (ORDER + 1)
            row.update(dense=rd.value, td=td, bd=max(rd.block_sizes))
        rows[n] = row
    return rows


def main():
    xstar, lstar, bound, xr = checks()
    rows = scaling()
    b = []
    ycap = 0.25
    # ---- (a) the least eigenvalue against the bias
    pa = N.Plot((-1, 1), (-0.30, 0.48), 5.2, 3.7, origin=(0.9, 1.15))
    b += pa.axes(xlabel="{\\scriptsize bias $x$}", ylabel="{\\scriptsize $\\lambda_{\\min}\\big(\\Gamma(x)\\big)$}",
                 xticks=[(-1, "$-1$"), (0, "$0$"), (1, "$1$")], yticks=[(0, "$0$"), (0.2, "$0.2$"), (0.4, "$0.4$")])
    b.append("\\draw[faint] %s -- %s;" % (pa.pt(-1, 0), pa.pt(1, 0)))
    xs = np.linspace(-1, 1, 801)
    lam = lam_min(xs, 5)
    neg = xs[lam < 0]
    seg = [pa.P(x, min(lam_min(x, 5), 0)) for x in np.linspace(neg[0], neg[-1], 60)]
    b.append("\\fill[ndRose] %s -- %s -- cycle;" % (" -- ".join(N.pt(p) for p in seg), pa.pt(neg[-1], 0)))
    b.append(pa.curve(xs, lam, "line width=1.3pt, ndBlue"))
    b.append("\\draw[line width=0.9pt, ndRed, dash pattern=on 3pt off 2pt] %s -- %s;" % (pa.pt(-1, bound), pa.pt(1, bound)))
    for x in XK:
        b.append("\\node[dot, minimum size=3.6pt] at %s {};" % pa.pt(x, lam_min(x, 5)))
    b.append("\\draw[ndRed, line width=0.8pt] %s circle (2.2pt);" % pa.pt(xr, bound))
    b.append("\\node[note, anchor=south] (sm) at %s {seven measured biases:\\\\ positive definite at each};" % pa.pt(0.2, 0.47))
    b.append("\\draw[pointer] (sm.south) to[bend right=15] %s;" % N.pt(pa.P(1 / 3, lam_min(1 / 3, 5)) + np.array([0.0, 0.08])))
    b.append("\\node[note, anchor=north west, text=ndRed] (cb) at %s {certified bound $%.4f$;\\\\ minimiser $x=%.3f$,\\\\ read off the pseudo-moments};"
             % (pa.pt(-0.72, -0.095), bound, xr))
    b.append("\\draw[pointer] (cb.north west) to[bend left=25] %s;" % N.pt(pa.P(xr, bound) + np.array([0.07, -0.07])))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(a) a chain of five qubits};" % (pa.P(0, 0)[0], ycap))

    # ---- (b) the chordal decomposition
    ox, oy, h = 8.2, 4.55, 0.34
    n5 = 5
    for i in range(n5):
        for j in range(n5):
            fill = "ndSky" if abs(i - j) <= 1 else "white"
            b.append("\\filldraw[fill=%s, draw=ndInk!35, line width=0.3pt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                     % (fill, ox + j * h, oy - i * h, ox + (j + 1) * h, oy - (i + 1) * h))
    cols = ["ndBlue", "ndRed", "ndGreen", "ndGold!85!ndInk"]
    for k in range(n5 - 1):
        b.append("\\draw[%s, line width=1.1pt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                 % (cols[k], ox + k * h + 0.03, oy - k * h - 0.03, ox + (k + 2) * h - 0.03, oy - (k + 2) * h + 0.03))
    b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,%.3f) {$\\Gamma(x)$: tridiagonal};" % (ox + 2.5 * h, oy + 0.05))
    gx, gy = ox + 2.25, oy - 0.85
    for i in range(n5):
        b.append("\\node[circle, draw=ndInk, fill=white, inner sep=1.2pt, font=\\tiny] at (%.3f,%.3f) {%d};" % (gx + 0.5 * i, gy, i + 1))
    for k in range(n5 - 1):
        b.append("\\draw[%s, line width=1.0pt, rounded corners=5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                 % (cols[k], gx + 0.5 * k - 0.19, gy - 0.19 - 0.05 * (k % 2), gx + 0.5 * (k + 1) + 0.19, gy + 0.19 + 0.05 * (k % 2)))
    b.append("\\node[note, anchor=south] at (%.3f,%.3f) {the cliques $\\{i,i+1\\}$\\\\ of a path};" % (gx + 1.0, gy + 0.3))
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {$\\Gamma\\succeq0$ if and only if $\\Gamma=\\sum_kE_k^{\\!\\top}S_kE_k$\\\\ with every $S_k\\succeq0$ of order $2$};"
             % (ox + 2.3, oy - 1.95))
    b.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {$\\left(\\begin{smallmatrix}1&3/4&0\\\\ 3/4&1&3/4\\\\ 0&3/4&1\\end{smallmatrix}\\right)$};" % (ox + 0.5, oy - 3.5))
    b.append("\\node[note, anchor=west] at (%.3f,%.3f) {pairs do not decide:\\\\ both $2\\times2$ blocks are\\\\ positive definite, yet\\\\ $\\lambda_{\\min}=-0.061$};" % (ox + 1.25, oy - 3.5))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(b) the chordal decomposition};" % (ox + 2.3, ycap))

    # ---- (c) the size of the certificate, and the time
    nn = sorted(rows)
    nd = [n for n in nn if "bd" in rows[n]]
    pc = N.Plot((0, 50), (0, 130), 3.6, 1.45, origin=(13.85, 3.2))
    b += pc.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize largest PSD block}",
                 xticks=[(8, "$8$"), (24, "$24$"), (48, "$48$")], yticks=[(10, "$10$"), (120, "$120$")])
    b.append(pc.curve(nd, [rows[n]["bd"] for n in nd], "line width=1.1pt, ndRed"))
    b.append(pc.curve(nn, [rows[n]["bs"] for n in nn], "line width=1.1pt, ndBlue"))
    for n in nd:
        b.append("\\fill[ndRed] %s circle (1.2pt);" % pc.pt(n, rows[n]["bd"]))
    for n in nn:
        b.append("\\fill[ndBlue] %s circle (1.2pt);" % pc.pt(n, rows[n]["bs"]))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at %s {dense, $5n$};" % pc.pt(26, 112))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=south] at %s {chordal, $10$};" % pc.pt(38, 16))
    lt = lambda v: np.log10(v)
    pt_ = N.Plot((0, 50), (-1.3, 2.0), 3.6, 1.45, origin=(13.85, 0.95))
    b += pt_.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize seconds}",
                  xticks=[(8, "$8$"), (24, "$24$"), (48, "$48$")], yticks=[(-1, "$0.1$"), (0, "$1$"), (1, "$10$")])
    b.append(pt_.curve(nd, [lt(rows[n]["td"]) for n in nd], "line width=1.1pt, ndRed"))
    b.append(pt_.curve(nn, [lt(rows[n]["ts"]) for n in nn], "line width=1.1pt, ndBlue"))
    for n in nd:
        b.append("\\fill[ndRed] %s circle (1.2pt);" % pt_.pt(n, lt(rows[n]["td"])))
    for n in nn:
        b.append("\\fill[ndBlue] %s circle (1.2pt);" % pt_.pt(n, lt(rows[n]["ts"])))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(c) dense against chordal};" % (pt_.P(25, 0)[0], ycap))
    N.write("ch11_sweep", b)
    return xstar, lstar, bound, xr, rows


if __name__ == "__main__":
    xstar, lstar, bound, xr, rows = main()
    print("n = 5: true min %.6f at x = %.4f; certified %.6f; recovered x = %.4f" % (lstar, xstar, bound, xr))
    for n in sorted(rows):
        r = rows[n]
        print("n = %2d: true %.6f sparse %.6f (block %d, %.1f s)" % (n, r["true"], r["sparse"], r["bs"], r["ts"])
              + ("  dense %.6f (block %d, %.1f s)" % (r["dense"], r["bd"], r["td"]) if "dense" in r else ""))
    print("written figures/ch11_sweep.tex")
