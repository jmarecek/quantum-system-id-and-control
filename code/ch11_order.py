"""ch11_order.py -- the model-order step, with certified residuals.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi), with cvxpy and
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

The qubit with the hidden defect of Chapter 4 (g = 1, kappa = 0.8, the defect
in |0>).  In the frame rotating at omega the amplitude left in the qubit is
    c(t) = e^{-kappa t/4} (cos W t + kappa/(4W) sin W t),   W = sqrt(g^2 - kappa^2/16)
(code/ch07_defect.py).  Two experiments, each sampled at the 21 times
t_k = 0.3 k in [0, 6] with S shots per time:
    x_k ~ <sigma_x> after preparing |+>       (mean c_k),
    P_k ~ population of |1> after preparing |1>  (mean c_k^2).
c changes sign at t = 1.81 and the population revives after it.  Each model is
judged by its one-step prediction error over k = 2, ..., 19.
(M1) The qubit alone, Markovian: x_{k+1} = lambda x_k, P_{k+1} = mu P_k, a
     channel of amplitude damping and dephasing.  It is completely positive iff
     [[1, lambda], [lambda, mu]] >= 0 and mu <= 1, which is the wedge
     T2 <= 2T1 of Figure ch11_physical in discrete time.  The residual is a
     convex quadratic in (lambda, mu) and the constraint a 2x2 PMI, so
     pysparsepmi.pmi_optimize at order 1 gives the least residual exactly;
     a direct cvxpy solve agrees to 1e-7.  The best Markovian qubit lies on
     the edge lambda^2 = mu, T2 = 2T1.
(M2) The qubit and one hidden mode, the defect: in the single-excitation
     sector the amplitudes (c, defect) evolve linearly, so
         x_{k+1} = a1 x_k + a2 x_{k-1},
         P_{k+1} = (a1^2 + a2) P_k + (a1^2 a2 + a2^2) P_{k-1} - a2^3 P_{k-2},
     the second because c^2 has the modes r1^2, r1 r2, r2^2 when c has r1, r2,
     the roots of s^2 - a1 s - a2.  The defect can only lose the excitation:
     the roots lie in the closed unit disc, the triangle |a1| <= 1 - a2,
     a2 >= -1.  The residual is a polynomial of degree 6 in (a1, a2), not
     convex.  Its minimum is bracketed by the order-3 bound of pmi_optimize
     (below) and a local solution, Nelder-Mead from the best point of a grid
     of the triangle (above).  In each of the 66 cases below SCS reports
     'optimal', the two agree to 1e-8 absolutely (the tolerance of SCS; at
     most 2e-6 relatively), and the pseudo-moments of order one are the local
     minimiser: the moment relaxation is exact here, and the landscape has a
     single minimum.  Truth:
     a1 = 2 e^{-kappa h/4} cos W h = 1.8027, a2 = -e^{-kappa h/2} = -0.8869.
M1 contains no model with a hidden mode, M2 contains the Markovian qubits
without dephasing (a2 = 0), and both have two parameters: what differs is the
order, one amplitude against two.  On the noiseless data the certified least
residual of M1 is 0.324 and that of M2 is 0.
(a) The data for S = 10^4 (dots: x_k filled, P_k open), with the free run of
    the best M1 (red, dashed) from the known preparation and of the best M2
    (blue) from c_0 = 1 and the c_1 that fits x best.  The best Markovian
    qubit has lambda^2 = mu to 1e-12: it sits on the edge T2 = 2T1 (without
    the PMI the least-squares (lambda, mu) of the noiseless data would lie
    outside the wedge, by lambda^2 - mu = 0.023).  It can only decay; M2
    follows the sign change and the revival.  The one-step criterion is biased
    by the noise in its regressors: the fitted (a1, a2) = (1.781, -0.868),
    and the blue curves swing a little less than the data.
(b) The landscape of the M2 residual for the same data, log10 of it in
    contours half a decade apart, over part of the triangle (sky) with the
    region where the amplitude would grow (rose).  Blue: the certified
    minimiser, residual 0.010319 (bound and local solution agree to 1e-6);
    gold: the truth.
(c) The least residuals against S (8 data sets for each S; medians, with the
    range as a band): M1 (certified) levels off at 0.32, the residual of the
    noiseless data, while M2 (certified) follows the noise level, the expected
    residual of the true model (dashed, proportional to 1/S), at 0.58 to 1.11
    times it from S = 1000 on.  The two part where the noise level crosses
    0.324, at S = 301 (thin line): below it the defect cannot be seen in these
    data, above it it cannot be missed; at S = 10^5 the ratio is 570.
Writes figures/ch11_order.tex.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from skimage import measure

try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp

import needham as N
import ch07_defect as D                       # amplitude(t), with g = 1, kappa = 0.8

G, KAPPA = D.E.G, D.E.KAPPA
W = np.sqrt(G ** 2 - KAPPA ** 2 / 16)
H = 0.3
TS = H * np.arange(21)
K = len(TS)
IDX = np.arange(2, K - 1)
C = D.amplitude(TS)
A1, A2 = 2 * np.exp(-KAPPA * H / 4) * np.cos(W * H), -np.exp(-KAPPA * H / 2)
SHOTS = [30, 100, 300, 1000, 3000, 10000, 30000, 100000]
REPS = 8
S_SHOW = 10000


def coeffs(a1, a2):
    """The recurrence of P = c^2 when c obeys c_{k+1} = a1 c_k + a2 c_{k-1}."""
    return a1 * a1 + a2, a1 * a1 * a2 + a2 * a2, -(a2 * a2 * a2)


def data(S, rng):
    x = 2 * rng.binomial(S, (1 + C) / 2) / S - 1
    P = rng.binomial(S, C ** 2) / S
    return x, P


def r1(x, P, lam, mu):
    k = IDX
    return np.sum((x[k + 1] - lam * x[k]) ** 2) + np.sum((P[k + 1] - mu * P[k]) ** 2)


def r2(x, P, a1, a2):
    k = IDX
    b1, b2, b3 = coeffs(a1, a2)
    return (np.sum((x[k + 1] - a1 * x[k] - a2 * x[k - 1]) ** 2)
            + np.sum((P[k + 1] - b1 * P[k] - b2 * P[k - 1] - b3 * P[k - 2]) ** 2))


def markov_cert(x, P):
    v = psp.polyvar(2)
    zero = psp.Polynomial.zero(2)
    f = zero
    for k in IDX:
        e = zero + x[k + 1] - v[0] * x[k]
        f = f + e * e
        e = zero + P[k + 1] - v[1] * P[k]
        f = f + e * e
    one = zero + 1.0
    res = psp.pmi_optimize([[f]], ineqs=[psp.PolyMatrix([[one, v[0]], [v[0], v[1]]], nvars=2), one - v[1]],
                           order=1, solver="SCS", eps=1e-9, max_iters=100000)
    return res.value, res.status, res.minimizer()


def markov_direct(x, P):
    lam, mu = cp.Variable(), cp.Variable()
    k = IDX
    Z = cp.Variable((2, 2), symmetric=True)
    pr = cp.Problem(cp.Minimize(cp.sum_squares(x[k + 1] - lam * x[k]) + cp.sum_squares(P[k + 1] - mu * P[k])),
                    [Z >> 0, Z[0, 0] == 1, Z[0, 1] == lam, Z[1, 1] == mu, mu <= 1])
    pr.solve(solver="SCS", eps=1e-10, max_iters=200000)
    return pr.value, (lam.value, mu.value)


def defect_cert(x, P):
    v = psp.polyvar(2)
    zero = psp.Polynomial.zero(2)
    b1, b2, b3 = coeffs(v[0], v[1])
    f = zero
    for k in IDX:
        e = zero + x[k + 1] - v[0] * x[k] - v[1] * x[k - 1]
        f = f + e * e
        e = zero + P[k + 1] - b1 * P[k] - b2 * P[k - 1] - b3 * P[k - 2]
        f = f + e * e
    one = zero + 1.0
    res = psp.pmi_optimize([[f]], ineqs=[one + v[1], one - v[1] - v[0], one - v[1] + v[0]],
                           order=3, solver="SCS", eps=1e-9, max_iters=200000)
    return res.value, res.status, res.minimizer()


def triangle_grid(n=81):
    pts = []
    for a2 in np.linspace(-1, 1, n):
        for a1 in np.linspace(-(1 - a2), 1 - a2, n):
            pts.append((a1, a2))
    return np.array(pts)


GRID = triangle_grid()


def defect_local(x, P):
    """A local solution: the best grid point, polished by Nelder-Mead (staying in the triangle)."""
    vals = np.array([r2(x, P, a, b) for a, b in GRID])
    q0 = GRID[np.argmin(vals)]
    pen = lambda q: r2(x, P, *q) + 1e6 * (max(0, -1 - q[1]) + max(0, abs(q[0]) - 1 + q[1]))
    r = minimize(pen, q0, method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-15, maxiter=4000))
    assert abs(r.x[0]) <= 1 - r.x[1] + 1e-12 and r.x[1] >= -1 - 1e-12
    return r2(x, P, *r.x), r.x, vals.min()


def noise_level(S):
    """Expected residual of the true model: the shot noise passed through its predictor."""
    vx, vP = (1 - C ** 2) / S, C ** 2 * (1 - C ** 2) / S
    b1, b2, b3 = coeffs(A1, A2)
    k = IDX
    return np.sum(vx[k + 1] + A1 ** 2 * vx[k] + A2 ** 2 * vx[k - 1]) + \
        np.sum(vP[k + 1] + b1 ** 2 * vP[k] + b2 ** 2 * vP[k - 1] + b3 ** 2 * vP[k - 2])


def run(x, P):
    """Both models on one data set, every certified number cross-checked."""
    v1, s1, m1 = markov_cert(x, P)
    d1, (lam, mu) = markov_direct(x, P)
    assert s1 == "optimal" and abs(v1 - d1) < 1e-7 * max(1, d1), (v1, d1, s1)
    assert abs(r1(x, P, *m1) - v1) < 1e-6 * max(1, v1)
    v2, s2, m2 = defect_cert(x, P)
    l2, q2, g2 = defect_local(x, P)
    assert s2 == "optimal", s2
    assert v2 <= l2 + 1e-8 and v2 <= g2 + 1e-8                      # a lower bound (to the tolerance of SCS)
    assert abs(l2 - v2) < 1e-8 + 1e-5 * l2, (v2, l2)                 # and the bracket closes
    assert np.allclose(m2, q2, atol=1e-3)                            # the pseudo-moments give the minimiser
    return dict(m1=v1, lam=m1[0], mu=m1[1], m2lo=v2, m2hi=l2, a=q2, truth=r2(x, P, A1, A2))


def sweep(seed=21):
    rng = np.random.default_rng(seed)
    return {S: [run(*data(S, rng)) for _ in range(REPS)] for S in SHOTS}


def free_markov(lam, mu, t):
    return lam ** (t / H), mu ** (t / H)


def free_defect(a, x, t):
    """c(t) for the fitted recurrence, with c_0 = 1 and the c_1 that fits the x data best."""
    r = np.roots([1, -a[0], -a[1]]).astype(complex)
    assert abs(r[0].imag) > 1e-6                                     # a damped oscillation
    V = np.array([[1, 1], r])                                        # c_k = alpha r1^k + beta r2^k
    basis = []
    for c1 in (0.0, 1.0):
        al = np.linalg.solve(V, [1.0, c1])
        basis.append(np.real(al[0] * r[0] ** np.arange(K) + al[1] * r[1] ** np.arange(K)))
    u = basis[1] - basis[0]
    c1 = np.dot(x - basis[0], u) / np.dot(u, u)
    al = np.linalg.solve(V, [1.0, c1])
    return np.real(al[0] * r[0] ** (t / H) + al[1] * r[1] ** (t / H))


def checks(sw, show):
    # the noiseless data obey the recurrences of M2 exactly, and no Markovian qubit fits them
    b1, b2, b3 = coeffs(A1, A2)
    assert np.allclose(C[2:], A1 * C[1:-1] + A2 * C[:-2]) and np.allclose(C[3:] ** 2, b1 * C[2:-1] ** 2 + b2 * C[1:-2] ** 2 + b3 * C[:-3] ** 2)
    assert abs(A1) <= 1 - A2 and A2 >= -1
    zero = run(C, C ** 2)
    assert zero["m2hi"] < 1e-9 and abs(zero["m1"] - 0.324) < 0.0005
    assert abs(zero["lam"] ** 2 - zero["mu"]) < 1e-9                # on the edge T2 = 2T1
    lu = np.sum(C[IDX + 1] * C[IDX]) / np.sum(C[IDX] ** 2)            # without the constraint: outside the wedge
    mu_ = np.sum(C[IDX + 1] ** 2 * C[IDX] ** 2) / np.sum(C[IDX] ** 4)
    assert abs(lu ** 2 - mu_ - 0.023) < 0.001
    # the sweep: M1 levels off, M2 follows the noise level
    med = lambda S, k: np.median([r[k] for r in sw[S]])
    for S in SHOTS[3:]:
        assert abs(med(S, "m1") - 0.324) < 0.04
        assert 0.5 < med(S, "m2hi") / noise_level(S) < 1.2
    assert med(30, "m1") / med(30, "m2hi") < 2 and 500 < med(100000, "m1") / med(100000, "m2hi") < 650
    Sx = noise_level(1.0) / 0.324
    assert 295 < Sx < 305
    # the data set shown: the best Markovian qubit is on the edge, M2 near the truth
    assert abs(show["lam"] ** 2 - show["mu"]) < 1e-9
    assert np.allclose(show["a"], (1.781, -0.868), atol=5e-4) and abs(show["m2hi"] - 0.010319) < 1e-6
    assert abs(show["m2hi"] - show["m2lo"]) < 1e-6 * show["m2hi"]
    # its landscape has a single local minimum on a fine grid of the triangle
    x, P = show["data"]
    n = 161
    a2s = np.linspace(-1, 1, n)
    a1s = np.linspace(-2, 2, 2 * n - 1)
    Z = np.array([[r2(x, P, a, c) if abs(a) <= 1 - c + 1e-12 else np.inf for a in a1s] for c in a2s])
    mins = [(i, j) for i in range(1, n - 1) for j in range(1, 2 * n - 2)
            if np.isfinite(Z[i, j]) and Z[i, j] <= Z[i - 1:i + 2, j - 1:j + 2].min()]
    assert len(mins) == 1 and np.hypot(a1s[mins[0][1]] - show["a"][0], a2s[mins[0][0]] - show["a"][1]) < 0.02
    return Sx


def landscape(x, P, lim, n=161):
    (x0, x1), (y0, y1) = lim
    a1s, a2s = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    F = np.array([[np.log10(r2(x, P, a, b)) for a in a1s] for b in a2s])
    return a1s, a2s, F


def main():
    sw = sweep()
    rng = np.random.default_rng(4)
    xs, Ps = data(S_SHOW, rng)
    show = run(xs, Ps)
    show["data"] = (xs, Ps)
    Sx = checks(sw, show)
    b = []

    # ---- (a) the data and the free runs of the two fitted models
    tt = np.linspace(0, 6, 241)
    xm, Pm = free_markov(show["lam"], show["mu"], tt)
    cd = free_defect(show["a"], xs, tt)
    for row, (ylim, dat, mk, md, lab, yt) in enumerate([
            ((-0.65, 1.05), xs, xm, cd, "$\\langle\\sigma_x\\rangle$, from $\\ket+$", [(-0.5, "$-0.5$"), (0.5, "$0.5$"), (1, "$1$")]),
            ((0, 1.05), Ps, Pm, cd ** 2, "population of $\\ket1$, from $\\ket1$", [(0.5, "$0.5$"), (1, "$1$")])]):
        hgt = 1.9 if row == 0 else 1.45
        p = N.Plot((0, 6), ylim, 4.0, hgt, origin=(0.9, 3.0 if row == 0 else 0.8))
        b += p.axes(xlabel="{\\scriptsize $t$}", xticks=[(2, "$2$"), (4, "$4$"), (6, "$6$")] if row == 1 else [],
                    yticks=yt, yzero=0 if row == 0 else None)
        b.append(p.curve(tt, mk, "line width=0.9pt, ndRed, dash pattern=on 3pt off 2pt"))
        b.append(p.curve(tt, md, "line width=0.9pt, ndBlue"))
        for t, y in zip(TS, dat):
            b.append(("\\fill[ndInk] %s circle (1.1pt);" if row == 0 else "\\draw[ndInk, fill=white, line width=0.5pt] %s circle (1.2pt);") % p.pt(t, y))
        b.append("\\node[font=\\scriptsize, anchor=north east] at %s {%s};" % (p.pt(6.1, ylim[1] + 0.02), lab))
        if row == 0:
            b.append("\\node[font=\\scriptsize, anchor=west, text=ndRed] at %s {qubit alone};" % p.pt(2.2, 0.5))
            b.append("\\node[font=\\scriptsize, anchor=north, text=ndBlue] at %s {qubit and defect};" % p.pt(3.3, -0.62))
        else:
            b.append("\\node[note, anchor=south] (rv) at %s {revival};" % p.pt(4.6, 0.45))
            b.append("\\draw[pointer] (rv.south) -- %s;" % p.pt(3.4, 0.31))
    b.append("\\node[font=\\small, anchor=north] at (2.9,0.25) {(a) data and two models};")

    # ---- (b) the landscape of the residual of M2
    lim = ((0.9, 2.1), (-1.1, 0.15))
    pb = N.Plot(lim[0], lim[1], 3.9, 3.9 * 1.25 / 1.2, origin=(6.15, 0.8))
    (x0, x1), (y0, y1) = lim
    b.append("\\begin{scope}")
    b.append("\\clip %s rectangle %s;" % (pb.pt(x0, y0), pb.pt(x1, y1)))
    b.append("\\fill[ndRose!70] %s rectangle %s;" % (pb.pt(x0, y0), pb.pt(x1, y1)))
    b.append("\\fill[ndSky!60] %s -- %s -- %s -- cycle;" % (pb.pt(0, 1), pb.pt(2, -1), pb.pt(-2, -1)))
    a1s, a2s, F = landscape(xs, Ps, lim)
    for lev in np.floor(F.min()) + 0.5 + 0.5 * np.arange(8):
        for path in measure.find_contours(F, lev):
            pts = [(np.interp(cj, np.arange(len(a1s)), a1s), np.interp(ci, np.arange(len(a2s)), a2s)) for ci, cj in path]
            b.append(N.polyline([pb.P(*q) for q in pts], "line width=0.5pt, ndInk!55"))
    b.append("\\draw[ink] %s -- %s -- %s;" % (pb.pt(0, 1), pb.pt(2, -1), pb.pt(-2, -1)))
    b.append("\\end{scope}")
    b += pb.axes(xlabel="{\\scriptsize $a_1$}", ylabel="{\\scriptsize $a_2$}", xticks=[(1, "$1$"), (1.5, "$1.5$"), (2, "$2$")],
                 yticks=[(-1, "$-1$"), (-0.5, "$-0.5$"), (0, "$0$")])
    b.append("\\node[dot, fill=ndGold, minimum size=5pt] at %s {};" % pb.pt(A1, A2))
    b.append("\\fill[ndBlue] %s circle (1.7pt);" % pb.pt(*show["a"]))
    b.append("\\node[note, anchor=north east, text=ndRed, fill=ndRose!70, inner sep=1.5pt] at %s {the amplitude\\\\ would grow};" % pb.pt(2.08, 0.1))
    b.append("\\node[note, anchor=west, fill=ndSky!60, inner sep=1.5pt] (m) at %s {certified\\\\ minimum};" % pb.pt(1.0, -0.82))
    b.append("\\draw[pointer] (m.east) to[bend right=15] %s;" % pb.pt(show["a"][0] - 0.03, show["a"][1]))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the defect model};" % N.pt(pb.P(1.5, y0) + np.array([0, -0.55])))

    # ---- (c) least residuals against the number of shots
    pc = N.Plot((1.3, 5.2), (-3.6, 0.4), 3.8, 4.2, origin=(11.3, 0.8))
    lx = np.log10(SHOTS)
    for key, col in (("m1", "ndRed"), ("m2hi", "ndBlue")):
        lo = np.log10([min(r[key] for r in sw[S]) for S in SHOTS])
        hi = np.log10([max(r[key] for r in sw[S]) for S in SHOTS])
        pts = [pc.P(u, v) for u, v in zip(lx, lo)] + [pc.P(u, v) for u, v in zip(lx[::-1], hi[::-1])]
        b.append("\\fill[%s, fill opacity=0.2] %s -- cycle;" % (col, " -- ".join(N.pt(q) for q in pts)))
    b += pc.axes(xlabel="{\\scriptsize $S$}", ylabel="{\\scriptsize least residual}",
                 xticks=[(2, "$10^2$"), (3, "$10^3$"), (4, "$10^4$"), (5, "$10^5$")],
                 yticks=[(-3, "$10^{-3}$"), (-2, "$10^{-2}$"), (-1, "$10^{-1}$"), (0, "$1$")])
    ss = np.linspace(np.log10(noise_level(1.0)) - 0.4, 5.15, 50)                 # inside the frame
    b.append(pc.curve(ss, np.log10([noise_level(10 ** s) for s in ss]), "dashedink"))
    for key, col in (("m1", "ndRed"), ("m2hi", "ndBlue")):
        ys = np.log10([np.median([r[key] for r in sw[S]]) for S in SHOTS])
        b.append(pc.curve(lx, ys, "line width=0.9pt, %s" % col))
        b += ["\\fill[%s] %s circle (1.2pt);" % (col, pc.pt(u, v)) for u, v in zip(lx, ys)]
    b.append("\\node[font=\\scriptsize, anchor=south east, text=ndRed] at %s {qubit alone};" % pc.pt(5.2, -0.42))
    b.append("\\node[font=\\scriptsize, anchor=north west, text=ndBlue, align=left] at %s {qubit and\\\\ defect};" % pc.pt(1.32, -1.9))
    b.append("\\node[note, anchor=south] (nl) at %s {noise level};" % pc.pt(4.35, -1.45))
    b.append("\\draw[pointer] (nl.south) -- %s;" % pc.pt(4.35, np.log10(noise_level(10 ** 4.35)) + 0.04))
    b.append("\\draw[faint] %s -- %s;" % (pc.pt(np.log10(Sx), -3.6), pc.pt(np.log10(Sx), 0.4)))
    b.append("\\node[font=\\scriptsize, anchor=south west, inner sep=1pt] at %s {$%d$};" % (pc.pt(np.log10(Sx), -3.6), round(Sx, -1)))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the least residuals};" % N.pt(pc.P(3.25, -3.6) + np.array([0, -0.55])))
    N.write("ch11_order", b)
    return sw, show, Sx


if __name__ == "__main__":
    sw, show, Sx = main()
    print("truth a1 = %.4f, a2 = %.4f" % (A1, A2))
    print("shown (S = %d): M1 certified %.5f at lambda %.4f, mu %.4f (lambda^2 - mu = %.1e); M2 bracket [%.6f, %.6f] at (%.4f, %.4f); truth residual %.6f"
          % (S_SHOW, show["m1"], show["lam"], show["mu"], show["lam"] ** 2 - show["mu"], show["m2lo"], show["m2hi"], *show["a"], show["truth"]))
    for S in SHOTS:
        rs = sw[S]
        print("S = %6d: M1 median %.5f; M2 median %.6f (largest relative gap of the bracket %.1e); noise level %.6f"
              % (S, np.median([r["m1"] for r in rs]), np.median([r["m2hi"] for r in rs]),
                 max((r["m2hi"] - r["m2lo"]) / r["m2hi"] for r in rs), noise_level(S)))
    print("noise level crosses the M1 floor at S = %.0f" % Sx)
    print("written figures/ch11_order.tex")
