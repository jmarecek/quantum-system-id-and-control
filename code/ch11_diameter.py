"""ch11_diameter.py -- the certified diameter of the consistent models, experiment by experiment.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi), with cvxpy and
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

The qubit of Chapter 5 (omega = 3, gamma1 = 0.2, g_phi = 0.4) has the GKSL
parameters p* = (theta, R, S) of code/ch11_physical.py, twelve numbers, with
theta_z = 2.12, R_xx = R_yy = 0.1, R_zz = 0.4, S_xy = -0.1 and the rest zero.
Its Kossakowski matrix has the eigenvalues 0.4, 0.2 and 0: the qubit does not
heat, and p* lies on the boundary of the cone gamma >= 0.  An experiment
prepares one of |0>, |1>, |+>, |+i> and measures the initial rate of change
of one of <X>, <Y>, <Z>, that is one entry of A x0 + beta, a linear function
l_e . p of the parameters; the twelve experiments are added in that order
(state by state, readouts x, y, z), and each raises the rank of the data map
by one.  The data are d_e = l_e . p* + xi_e with xi_e ~ N(0, 0.01^2), and the
models consistent with them are

    C_n = { p : |l_e . p - d_e| <= eps (e <= n),  |theta_j| <= 3,  |R_jk|, |S_jk| <= 1 },
    eps = 0.03,

with or without complete positivity, [[R, -S], [S, R]] >= 0.  For each n and
each of the twelve parameters, pysparsepmi.pmi_optimize (order 1, exact here
since the problem is a semidefinite programme) bounds min p_k and max p_k:
624 certificates, and 48 more for the scaling below, all ending with SCS
status "optimal" (6 of the 672 report "optimal_inaccurate" at the tolerance
1e-9 and are solved again at 1e-8).  The widths w_k give the
certified upper bound U = ||w|| on the diameter of C_n; the pseudo-moment
minimisers are feasible points (checked by eigenvalues), and the largest
distance between the two extremisers of one parameter is a lower bound L.

  data alone:     L = 6 (theta_z unseen, box |theta_z| <= 3) for n <= 7,
                  2.47 <= L <= U <= 5.11 (rates bounded only by the box) for
                  8 <= n <= 11, and the noise level, L = 0.19, U = 0.34, at
                  n = 12, when the data map reaches rank twelve;
  with gamma>=0:  L = 6 for n <= 7, then L = 0.29, U = 0.54 at n = 8 (the
                  Hamiltonian is seen, and positivity pins every rate), and
                  L = 0.15, U = 0.29 at n = 12.

The largest single width is 6, 2 and 0.15 (data alone) and 6, 0.23 and 0.14
(with positivity) on the three stretches.  The width left by positivity
shrinks like sqrt(eps), the width left by the data like eps: with the noise
and eps divided by ten, the largest width at n = 8 falls by a factor of 3.59,
at n = 12 by 9.99.
Cross-checks: every certified bound agrees with scipy.optimize.linprog (data
alone) or a direct cvxpy solve (with positivity) to 1e-5; at n = 12 the
widths of the data-alone set equal 2 eps sum_e |(E^-1)_ke|, E the matrix
with the rows l_e, as they must for a parallelotope; and a brute-force hit-and-run walk of 2000 steps in each
C_n, with the chords computed from eigenvalues, never leaves the certified
bounding box and never finds two points further apart than U.
(a) The bounds L <= diam C_n <= U against n, data alone (blue) and with
    complete positivity (red), on a logarithmic scale.
(b), (c) The certified widths w_k, one row per parameter and one column per
    n, darker for wider (from 0.02 to 6, logarithmically).
Writes figures/ch11_diameter.tex.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from scipy.spatial.distance import pdist

try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp

import needham as N
import ch05_spectrum as Q                    # the qubit of Chapter 5
import ch11_physical as PH                   # M: (theta, R, S) -> (A, beta); kossakowski

M = PH.M
NAMES = ["\\theta_x", "\\theta_y", "\\theta_z", "R_{xx}", "R_{xy}", "R_{xz}", "R_{yy}", "R_{yz}", "R_{zz}",
         "S_{xy}", "S_{xz}", "S_{yz}"]
BOX = np.array([3.0] * 3 + [1.0] * 9)
SIGMA, EPS = 0.01, 0.03
STATES = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0)]          # |0>, |1>, |+>, |+i>
ROWS = np.array([sum(x0[k] * M[3 * j + k] for k in range(3)) + M[9 + j] for x0 in STATES for j in range(3)])
A5 = Q.qubit_A()
P_TRUE = np.linalg.solve(M, np.concatenate([A5[:3, :3].ravel(), A5[:3, 3]]))
XI = SIGMA * np.random.default_rng(0).normal(size=12)
V = psp.polyvar(12)
ZERO = psp.Polynomial.zero(12)


def kmat(p):
    """The real 12x12 parameters -> 6x6 matrix [[R, -S], [S, R]], PSD iff gamma >= 0."""
    g = PH.kossakowski(p)
    return np.block([[g.real, -g.imag], [g.imag, g.real]])


def linear(row, c):
    e = ZERO + c
    for k in range(12):
        if abs(row[k]) > 1e-14:
            e = e + V[k] * row[k]
    return e


def constraints(n, psd, d, eps=EPS):
    # the box as B^2 - p_k^2 >= 0: exact at order 1, and it keeps the linear monomials in the
    # Gram basis (with only linear constraints, minimizer() returns the pseudo-moments negated)
    cons = [BOX[k] ** 2 - V[k] * V[k] for k in range(12)]
    for e in range(n):
        cons += [linear(-ROWS[e], eps + d[e]), linear(ROWS[e], eps - d[e])]
    if psd:
        K = [[None] * 6 for _ in range(6)]
        cols = [kmat(np.eye(12)[k]) for k in range(12)]
        for i in range(6):
            for j in range(6):
                K[i][j] = sum((V[k] * cols[k][i, j] for k in range(12) if abs(cols[k][i, j]) > 0), ZERO)
        cons = [psp.PolyMatrix(K, nvars=12)] + cons
    return cons


RETRIES = []


def bound(f, cons):
    """pmi_optimize at the SCS tolerance 1e-9, or at 1e-8 if SCS reports "optimal_inaccurate"."""
    res = psp.pmi_optimize([[f]], ineqs=cons, order=1, solver="SCS", eps=1e-9, max_iters=200000)
    if res.status != "optimal":
        RETRIES.append(res.status)
        res = psp.pmi_optimize([[f]], ineqs=cons, order=1, solver="SCS", eps=1e-8, max_iters=200000)
    return res


def certified(n, psd, d, eps=EPS):
    """Certified [lo_k, hi_k] for every parameter, the extremisers, and the SCS statuses."""
    cons = constraints(n, psd, d, eps)
    lo, hi, pts, st = np.zeros(12), np.zeros(12), [], []
    for k in range(12):
        a, b = bound(V[k], cons), bound(ZERO - V[k], cons)
        lo[k], hi[k] = a.value, -b.value
        pts.append((a.minimizer(), b.minimizer()))
        st += [a.status, b.status]
    return lo, hi, pts, st


def direct(n, psd, d):
    """The same bounds by linprog (data alone) or by a direct cvxpy solve (with positivity)."""
    A_ub = np.vstack([ROWS[:n], -ROWS[:n]])
    b_ub = np.concatenate([d[:n] + EPS, EPS - d[:n]])
    lo, hi = np.zeros(12), np.zeros(12)
    if not psd:
        for k in range(12):
            c = np.eye(12)[k]
            lo[k] = linprog(c, A_ub=A_ub if n else None, b_ub=b_ub if n else None,
                            bounds=list(zip(-BOX, BOX)), method="highs").fun
            hi[k] = -linprog(-c, A_ub=A_ub if n else None, b_ub=b_ub if n else None,
                             bounds=list(zip(-BOX, BOX)), method="highs").fun
        return lo, hi
    Z = cp.Variable((6, 6), symmetric=True)
    th = cp.Variable(3)
    R, S = Z[:3, :3], Z[3:, :3]
    par = cp.hstack([th] + [R[l, m] for l, m in PH.PAIRS] + [S[m, l] for l, m in PH.ANTI])
    cons = [Z >> 0, Z[:3, :3] == Z[3:, 3:], S + S.T == 0, cp.abs(par) <= BOX]
    if n:
        cons.append(cp.abs(ROWS[:n] @ par - d[:n]) <= EPS)
    for k in range(12):
        for sgn in (1, -1):
            prob = cp.Problem(cp.Minimize(sgn * par[k]), cons)
            prob.solve(solver="SCS", eps=1e-10, max_iters=200000)
            if sgn == 1:
                lo[k] = prob.value
            else:
                hi[k] = -prob.value
    return lo, hi


def feasible(p, n, psd, d, tol=1e-6):
    ok = np.all(np.abs(p) <= BOX + tol) and np.all(np.abs(ROWS[:n] @ p - d[:n]) <= EPS + tol)
    return ok and (not psd or np.linalg.eigvalsh(kmat(p)).min() >= -tol)


def hit_and_run(n, psd, d, start, steps=2000, seed=0):
    """Brute force: a random walk on random chords of C_n, chords from eigenvalues."""
    rng = np.random.default_rng(seed)
    A = np.vstack([np.eye(12), -np.eye(12), ROWS[:n], -ROWS[:n]])
    c = np.concatenate([BOX, BOX, d[:n] + EPS, EPS - d[:n]])
    p, out = start.copy(), []
    for _ in range(steps):
        u = rng.normal(size=12)
        u /= np.linalg.norm(u)
        au, slack = A @ u, c - A @ p
        t_hi = np.min(slack[au > 1e-14] / au[au > 1e-14])
        t_lo = np.max(slack[au < -1e-14] / au[au < -1e-14])
        if psd:
            L = np.linalg.cholesky(kmat(p))
            Li = np.linalg.inv(L)
            mu = np.linalg.eigvalsh(Li @ kmat(u) @ Li.T)
            if np.any(mu > 1e-14):
                t_lo = max(t_lo, np.max(-1 / mu[mu > 1e-14]))
            if np.any(mu < -1e-14):
                t_hi = min(t_hi, np.min(-1 / mu[mu < -1e-14]))
        p = p + rng.uniform(t_lo, t_hi) * u
        out.append(p)
    return np.array(out)


def staircase(psd, d):
    rows = []
    for n in range(13):
        lo, hi, pts, st = certified(n, psd, d)
        rows.append((lo, hi, pts, st))
    return rows


def bounds(row):
    lo, hi, pts, st = row
    w = hi - lo
    return np.linalg.norm(w), max(np.linalg.norm(a - b) for a, b in pts), w


def checks(runs):
    # the truth: the qubit of Chapter 5, on the boundary of the cone
    ev = np.linalg.eigvalsh(PH.kossakowski(P_TRUE))
    assert np.allclose(ev, [0.0, 0.2, 0.4], atol=1e-12)
    assert abs(P_TRUE[2] - 3 / np.sqrt(2)) < 1e-12 and np.allclose(P_TRUE[[3, 6, 8, 9]], [0.1, 0.1, 0.4, -0.1])
    # each experiment raises the rank of the data map by one; the noise is within eps
    assert [np.linalg.matrix_rank(ROWS[:n]) for n in range(1, 13)] == list(range(1, 13))
    assert np.abs(XI).max() <= EPS
    d = ROWS @ P_TRUE + XI
    for psd, rows in runs.items():
        for n, row in enumerate(rows):
            lo, hi, pts, st = row
            assert set(st) == {"optimal"}, (psd, n, st)
            assert feasible(P_TRUE, n, psd, d) and np.all(lo <= P_TRUE + 1e-6) and np.all(P_TRUE <= hi + 1e-6)
            dlo, dhi = direct(n, psd, d)
            assert np.abs(lo - dlo).max() < 1e-5 and np.abs(hi - dhi).max() < 1e-5, (psd, n, lo - dlo, hi - dhi)
            for k, (a, b) in enumerate(pts):                        # the extremisers are feasible
                assert feasible(a, n, psd, d) and feasible(b, n, psd, d)
                assert abs(a[k] - lo[k]) < 1e-5 and abs(b[k] - hi[k]) < 1e-5
            U, L, w = bounds(row)
            start = np.mean([q for ab in pts for q in ab], axis=0)
            assert feasible(start, n, psd, d, tol=0) and (not psd or np.linalg.eigvalsh(kmat(start)).min() > 1e-9)
            walk = hit_and_run(n, psd, d, start)
            assert np.all(walk >= lo - 1e-6) and np.all(walk <= hi + 1e-6)
            spread = pdist(walk[::4]).max()
            assert L <= U + 1e-9 and spread <= U + 1e-9
    # the step: data alone, the noise level only at n = 12, where C_12 is a parallelotope
    W = np.linalg.inv(ROWS)
    lo, hi = runs[False][12][:2]
    assert np.allclose(hi - lo, 2 * EPS * np.abs(W).sum(axis=1), atol=1e-6)
    wmax = [bounds(r)[2].max() for r in runs[False]]
    assert min(wmax[:8]) >= 6 - 1e-6 and min(wmax[8:12]) >= 2 - 1e-6 and wmax[12] < 0.2
    wmax = [bounds(r)[2].max() for r in runs[True]]
    assert min(wmax[:8]) >= 6 - 1e-6 and max(wmax[8:]) < 0.25
    # sqrt(eps) against eps: the same noise divided by ten
    d10 = ROWS @ P_TRUE + XI / 10
    r8 = bounds(certified(8, True, d10, EPS / 10))[2].max()
    r12 = bounds(certified(12, True, d10, EPS / 10))[2].max()
    f8 = bounds(runs[True][8])[2].max() / r8
    f12 = bounds(runs[True][12])[2].max() / r12
    assert 2.5 < f8 < 5 and 8 < f12 < 12, (f8, f12)
    return f8, f12


def main():
    d = ROWS @ P_TRUE + XI
    runs = {False: staircase(False, d), True: staircase(True, d)}
    f8, f12 = checks(runs)
    b = []
    col = {False: "ndBlue", True: "ndRed"}
    # ---- (a) the staircase
    ylo, yhi = -1.25, 1.25
    pa = N.Plot((-0.5, 12.5), (ylo, yhi), 5.6, 3.7, origin=(1.0, 0.95))
    b += pa.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize $\\operatorname{diam}C_n$}",
                 xticks=[(0, "$0$"), (4, "$4$"), (8, "$8$"), (12, "$12$")],
                 yticks=[(-1, "$0.1$"), (0, "$1$"), (1, "$10$")])
    for psd in (False, True):
        U = np.log10([bounds(r)[0] for r in runs[psd]])
        L = np.log10([bounds(r)[1] for r in runs[psd]])
        for vals, lw in [(U, 0.5), (L, 1.3)]:
            b.append(N.polyline([pa.P(x, u) for n, u in enumerate(vals) for x in (n - 0.5, n + 0.5)],
                                "line width=%.1fpt, %s" % (lw, col[psd])))
    b.append("\\node[note, anchor=west] (ub) at %s {upper bound $\\norm w$};" % pa.pt(1.6, 1.2))
    b.append("\\draw[pointer] (ub.west) to[bend right=20] %s;" % pa.pt(0.9, np.log10(bounds(runs[False][1])[0]) + 0.02))
    b.append("\\node[note, anchor=west] (lb) at %s {lower bound};" % pa.pt(1.6, 0.5))
    b.append("\\draw[pointer] (lb.west) to[bend left=20] %s;" % pa.pt(0.9, np.log10(6.0) - 0.02))
    b.append("\\node[note, anchor=north west, align=left, text=ndBlue] (da) at %s {data alone:\\\\ the noise level\\\\ only at rank $12$};" % pa.pt(9.3, 1.55))
    b.append("\\draw[pointer] (da.south) to[bend left=25] %s;" % pa.pt(12.0, -0.68))
    b.append("\\node[note, anchor=north, text=ndRed] (dp) at %s {with $\\gamma\\succeq0$:\\\\ the rates are pinned\\\\ once the Hamiltonian\\\\ is seen};" % pa.pt(3.3, -0.12))
    b.append("\\draw[pointer] (dp.east) to[bend right=20] %s;" % pa.pt(7.35, -0.36))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the diameter, certified};" % N.pt(pa.P(6, ylo) + np.array([0, -0.55])))

    # ---- (b), (c) the widths
    cw, ch = 0.29, 0.27
    lw0, lw1 = np.log10(0.02), np.log10(6.0)
    for i, psd in enumerate((False, True)):
        x0, y0 = 8.15 + i * 4.35, 1.0
        for n, r in enumerate(runs[psd]):
            w = bounds(r)[2]
            for k in range(12):
                frac = np.clip((np.log10(w[k]) - lw0) / (lw1 - lw0), 0, 1)
                p = (x0 + n * cw, y0 + (11 - k) * ch)
                b.append("\\fill[%s!%d, draw=white, line width=0.3pt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                         % (col[psd], int(4 + 86 * frac), p[0], p[1], p[0] + cw, p[1] + ch))
        for n in (0, 4, 8, 12):
            b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$%d$};" % (x0 + (n + 0.5) * cw, y0 - 0.02, n))
        b.append("\\node[font=\\scriptsize, anchor=west] at (%.3f,%.3f) {$n$};" % (x0 + 13 * cw + 0.03, y0 - 0.17))
        if i == 0:
            for k in range(12):
                b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {$%s$};" % (x0 - 0.03, y0 + (11.5 - k) * ch, NAMES[k]))
        cap = ["(b) widths, data alone", "(c) widths, with $\\gamma\\succeq0$"][i]
        b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {%s};" % (x0 + 6.5 * cw, y0 - 0.4, cap))
    b.append("\\node[note, anchor=south] at (%.3f,%.3f) {dark: wide, pale: pinned};" % (8.15 + 4.35 / 2 + 6.5 * cw, 1.0 + 12 * ch + 0.05))
    N.write("ch11_diameter", b)
    return runs, f8, f12


if __name__ == "__main__":
    import time
    t0 = time.time()
    runs, f8, f12 = main()
    for psd in (False, True):
        print(["data alone", "with gamma >= 0"][psd])
        for n, r in enumerate(runs[psd]):
            U, L, w = bounds(r)
            print("  n = %2d: L = %.4f, U = %.4f, largest width %.4f" % (n, L, U, w.max()))
    print("noise / 10: largest width at n = 8 falls by %.2f, at n = 12 by %.2f" % (f8, f12))
    print("SCS retried at eps = 1e-8: %d of %d bounds" % (len(RETRIES), 2 * 12 * (2 * 13 + 2)))
    print("written figures/ch11_diameter.tex (%.1f s)" % (time.time() - t0))
