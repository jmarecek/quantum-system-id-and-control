"""ch11_physical.py -- certified physicality of an estimated qubit generator.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi), with cvxpy and
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

The coherence vector x_j = Tr(sigma_j rho) of a qubit obeys xdot = A x + beta,
and a GKSL generator with the Hamiltonian H = sum_p theta_p F_p and the
Hermitian Kossakowski matrix gamma = R + iS in the basis F_j = sigma_j/sqrt(2),
    L(rho) = -i[H, rho] + sum_{l,m} gamma_lm (F_l rho F_m - (1/2){F_m F_l, rho}),
gives (A, beta) linearly in the 12 real unknowns (theta, R, S) (R symmetric,
S antisymmetric).  For a qubit this linear map is a bijection (its matrix M
has rank 12 and condition number 2), as the identifiability results of
Chapter 4 lead one to expect.  The generator is physical (completely
positive) iff gamma >= 0, that is iff the real 6x6 matrix [[R, -S], [S, R]] is
positive semidefinite.  For an estimate (A^, beta^) we bound

    rho(A^, beta^) = min ||A(theta, R, S) - A^||_F^2 + ||beta(theta, R, S) - beta^||^2
                     subject to  [[R, -S], [S, R]] >= 0

from below with pysparsepmi.pmi_optimize (the residual is a quadratic
polynomial in the 12 unknowns, the constraint a polynomial matrix
inequality, relaxation order 1), and recover the nearest physical generator
from the first-order pseudo-moments.  The problem is convex, so the order-1
relaxation is exact and a direct cvxpy solve gives the same value; the
script checks that every certified value agrees with the direct solve to
1e-8, and with the closed forms below.  A positive bound certifies that NO
GKSL generator reproduces the estimate; a zero bound, with the exactly
reconstructed gamma >= 0, certifies that one does.

(a) The slice of generators of a qubit relaxing to |0> at the rate
    g1 = 1/T1 and dephasing at g_phi: A = diag(-Gamma, -Gamma, -g1),
    beta = (0, 0, g1), Gamma = 1/T2 = g1/2 + g_phi.  The reconstructed gamma
    has the eigenvalues g1, 0 and g_phi, so the physical part of the slice is
    the wedge 1/T2 >= 1/(2 T1), that is T2 <= 2 T1; the certificate
    (computed on a grid) is 0 inside and positive outside.  Marked: the qubit
    of Chapter 5 (g1 = 0.2, Gamma = 0.5), far inside, and a relaxation-limited
    qubit (g1 = 0.2, g_phi = 0.005, Gamma = 0.105), close to the edge.
(b) Sixty estimates of (1/T1, 1/T2) for the relaxation-limited qubit, each
    from a simulated T1 experiment (|1> prepared, population of |1> measured
    at 8 times in [1, 15] with 300 shots each) and a T2 experiment (|+>
    prepared, <sigma_x> measured at 8 times in [1, 20] with 300 shots each),
    fitted by least squares.  Fifteen of the sixty, 25%, have T2 > 2 T1 and are
    certified non-physical.  For a point outside the wedge the nearest
    physical generator lies on its edge, with the closed form
    g1* = (2 Gamma^ + 4 g1^)/5, Gamma* = g1*/2, beta* = (0, 0, g1*), and
    rho = 2 (Gamma* - Gamma^)^2 + 2 (g1* - g1^)^2; arrows join four of them to
    it.
(c) The qubit with the hidden defect of Chapter 4 (g = 1, kappa = 0.8, the
    defect in |0>), whose reduced dynamics in the rotating frame is amplitude
    damping with the amplitude c(t) and the time-local rate
    gamma(t) = -2 d/dt ln|c(t)| (code/ch08_nested.py): its time-local generator
    is A = diag(-gamma/2, -gamma/2, -gamma), beta = (0, 0, gamma).  The
    certified squared distance to the GKSL generators is 0 while gamma(t) >= 0
    and (5/2) gamma(t)^2 while gamma(t) < 0, the nearest GKSL generator being
    then the zero generator: a certificate, instant by instant, that no
    Markovian model of the qubit alone produces its dynamics between t = 1.81
    and t = 3.21.  Drawn: the rate gamma(t) and the certified distance
    sqrt(rho), with 30 certified values (dots) on the closed form, clipped to
    the window |.| <= 3.
Writes figures/ch11_physical.tex.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvxpy as cp
from scipy.optimize import curve_fit

try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp

import tikzexport as T
import needham as N
import ch08_nested as NE                      # rate(t) of the qubit with the hidden defect

I2, SX, SY, SZ = T.pauli()
PAULI = [SX, SY, SZ]
F = [s / np.sqrt(2) for s in PAULI]
PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]     # entries of R
ANTI = [(0, 1), (0, 2), (1, 2)]                             # entries of S
G1, GPHI, SHOTS = 0.2, 0.005, 300
GAM = G1 / 2 + GPHI


def dissipator(l, m, rho):
    return F[l] @ rho @ F[m] - 0.5 * (F[m] @ F[l] @ rho + rho @ F[m] @ F[l])


def coherence(gen):
    """(A, beta) of xdot = A x + beta for a generator of the master equation."""
    A = np.array([[0.5 * np.real(np.trace(PAULI[j] @ gen(PAULI[k]))) for k in range(3)] for j in range(3)])
    b = np.array([0.5 * np.real(np.trace(PAULI[j] @ gen(I2))) for j in range(3)])
    return A, b


def generator_map():
    """The 12x12 matrix taking (theta, R, S) to (A, beta) (A row by row, then beta)."""
    gens = [lambda r, p=p: -1j * (F[p] @ r - r @ F[p]) for p in range(3)]
    for l, m in PAIRS:
        gens.append((lambda r, l=l: dissipator(l, l, r)) if l == m
                    else (lambda r, l=l, m=m: dissipator(l, m, r) + dissipator(m, l, r)))
    for l, m in ANTI:
        gens.append(lambda r, l=l, m=m: 1j * (dissipator(l, m, r) - dissipator(m, l, r)))
    cols = []
    for g in gens:
        A, b = coherence(g)
        cols.append(np.concatenate([A.ravel(), b]))
    return np.array(cols).T


M = generator_map()


def kossakowski(par):
    R = np.zeros((3, 3))
    S = np.zeros((3, 3))
    for k, (l, m) in enumerate(PAIRS):
        R[l, m] = R[m, l] = par[3 + k]
    for k, (l, m) in enumerate(ANTI):
        S[l, m], S[m, l] = par[9 + k], -par[9 + k]
    return R + 1j * S


def certify(A_hat, b_hat):
    """Certified lower bound on the squared distance to the GKSL generators, and the nearest one."""
    v = psp.polyvar(12)
    zero = psp.Polynomial.zero(12)
    target = np.concatenate([A_hat.ravel(), b_hat])
    resid = zero
    for i in range(12):
        e = zero + (-target[i])
        for k in range(12):
            if abs(M[i, k]) > 1e-14:
                e = e + v[k] * M[i, k]
        resid = resid + e * e
    R = [[v[3], v[4], v[5]], [v[4], v[6], v[7]], [v[5], v[7], v[8]]]
    S = [[zero, v[9], v[10]], [zero - v[9], zero, v[11]], [zero - v[10], zero - v[11], zero]]
    K = [[None] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            K[i][j] = K[i + 3][j + 3] = R[i][j]
            K[i + 3][j] = S[i][j]
            K[i][j + 3] = zero - S[i][j]
    res = psp.pmi_optimize([[resid]], ineqs=[psp.PolyMatrix(K, nvars=12)], order=1,
                           solver="SCS", eps=1e-9, max_iters=100000)
    return res.value, res.status, res.minimizer()


def direct(A_hat, b_hat):
    """The same convex problem solved directly by cvxpy, as a check."""
    Z = cp.Variable((6, 6), symmetric=True)
    th = cp.Variable(3)
    R, S = Z[:3, :3], Z[3:, :3]
    par = cp.hstack([th] + [R[l, m] for l, m in PAIRS] + [S[m, l] for l, m in ANTI])
    target = np.concatenate([A_hat.ravel(), b_hat])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(M @ par - target)), [Z >> 0, Z[:3, :3] == Z[3:, 3:], S + S.T == 0])
    prob.solve(solver="SCS", eps=1e-10, max_iters=200000)
    return prob.value


def slice_generator(g1, gam):
    return np.diag([-gam, -gam, -g1]), np.array([0.0, 0.0, g1])


def closed_form(g1, gam):
    """Squared distance and nearest physical (1/T1, 1/T2) for a point of the slice."""
    if gam >= g1 / 2 and g1 >= 0:
        return 0.0, (g1, gam)
    gs = (2 * gam + 4 * g1) / 5
    return 2 * (gs / 2 - gam) ** 2 + 2 * (gs - g1) ** 2, (gs, gs / 2)


def estimates(n=60, seed=3, npts=8):
    """Least-squares fits of simulated T1 and T2 experiments: rows (1/T1, 1/T2)."""
    rng = np.random.default_rng(seed)
    t1, t2 = np.linspace(1, 15, npts), np.linspace(1, 20, npts)
    out = []
    for _ in range(n):
        p1 = rng.binomial(SHOTS, np.exp(-G1 * t1)) / SHOTS
        p2 = rng.binomial(SHOTS, (1 + np.exp(-GAM * t2)) / 2) / SHOTS
        a = curve_fit(lambda t, g: np.exp(-g * t), t1, p1, p0=[G1])[0][0]
        b = curve_fit(lambda t, g: (1 + np.exp(-g * t)) / 2, t2, p2, p0=[GAM])[0][0]
        out.append((a, b))
    return np.array(out)


def classify(est):
    """Certify every estimate; return (physical flag, bound, nearest slice point)."""
    rows = []
    for g1, gam in est:
        A, b = slice_generator(g1, gam)
        lam = np.linalg.eigvalsh(kossakowski(np.linalg.solve(M, np.concatenate([A.ravel(), b])))).min()
        val, status, x = certify(A, b)
        ref = direct(A, b)
        cf, near = closed_form(g1, gam)
        assert status == "optimal" and abs(val - ref) < 1e-8 and abs(val - cf) < 1e-8, (g1, gam, val, ref, cf)
        physical = lam >= -1e-12
        assert physical == (val < 1e-8)                  # exactly one of the two certificates
        Am = (M @ x)[:9].reshape(3, 3)
        if not physical:                                 # the nearest model lies on the edge of the wedge
            assert np.allclose((-Am[2, 2], -Am[0, 0]), near, atol=1e-5)
            assert np.abs(Am - np.diag(np.diag(Am))).max() < 1e-6
        rows.append((physical, val, (-Am[2, 2], -Am[0, 0])))
    return rows


def defect(ts):
    """Certified distances of the time-local generators of the qubit with the hidden defect."""
    out = []
    for t in ts:
        g = NE.rate(t)
        A, b = np.diag([-g / 2, -g / 2, -g]), np.array([0.0, 0.0, g])
        val, status, x = certify(A, b)
        ref = direct(A, b)
        cf = 2.5 * g ** 2 if g < 0 else 0.0
        assert status == "optimal" and abs(val - ref) < 1e-7 * max(1, ref) and abs(val - cf) < 1e-7 * max(1, cf), (t, g, val, ref, cf)
        if g < 0:                                        # the nearest GKSL generator is the zero generator
            assert np.abs(M @ x).max() < 1e-5
        out.append((t, g, val))
    return np.array(out)


def checks():
    # the map: amplitude damping towards |0> and pure dephasing
    g = 0.3
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    A, b = coherence(lambda r: g * (sm @ r @ sm.conj().T - 0.5 * (sm.conj().T @ sm @ r + r @ sm.conj().T @ sm)))
    assert np.allclose(A, np.diag([-g / 2, -g / 2, -g])) and np.allclose(b, [0, 0, g])
    par = np.zeros(12)
    par[3], par[6], par[9] = g / 2, g / 2, -g / 2
    assert np.allclose(M @ par, np.concatenate([A.ravel(), b]))
    A, b = coherence(lambda r: g / 2 * (SZ @ r @ SZ - r))
    assert np.allclose(A, np.diag([-g, -g, 0])) and np.allclose(b, 0)
    assert np.linalg.matrix_rank(M) == 12 and abs(np.linalg.cond(M) - 2) < 1e-9
    # the wedge: on a grid of the slice the certificate vanishes exactly where 1/T2 >= 1/(2 T1)
    for g1 in np.linspace(0.02, 0.34, 7):
        for gam in np.linspace(0.01, 0.58, 9):
            if abs(gam - g1 / 2) < 0.01:
                continue
            val = certify(*slice_generator(g1, gam))[0]
            assert (val < 1e-8) == (gam > g1 / 2), (g1, gam, val)
            assert abs(val - closed_form(g1, gam)[0]) < 1e-8


def main():
    checks()
    est = estimates()
    rows = classify(est)
    bad = [i for i, r in enumerate(rows) if not r[0]]
    frac = len(bad) / len(rows)
    ts = np.array([t for t in np.linspace(0.1, 4.8, 33) if abs(t - 1.81) > 0.1])
    dd = defect(ts[::1])
    b = []
    # ---- (a) the wedge
    pa = N.Plot((0, 0.36), (0, 0.6), 3.9, 3.9, origin=(0.9, 0.75))
    b.append("\\fill[ndSky!70] %s -- %s -- %s -- %s -- cycle;" % (pa.pt(0, 0), pa.pt(0.36, 0.18), pa.pt(0.36, 0.6), pa.pt(0, 0.6)))
    b.append("\\fill[ndRose!70] %s -- %s -- %s -- cycle;" % (pa.pt(0, 0), pa.pt(0.36, 0), pa.pt(0.36, 0.18)))
    b += pa.axes(xlabel="{\\scriptsize $1/T_1$}", ylabel="{\\scriptsize $1/T_2$}",
                 xticks=[(0.1, "$0.1$"), (0.2, "$0.2$"), (0.3, "$0.3$")], yticks=[(0.2, "$0.2$"), (0.4, "$0.4$")])
    b.append("\\draw[ink] %s -- %s;" % (pa.pt(0, 0), pa.pt(0.36, 0.18)))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % pa.pt(0.2, 0.5))
    b.append("\\node[font=\\scriptsize, anchor=west] at %s {$g_\\phi=0.4$};" % N.pt(pa.P(0.2, 0.5) + np.array([0.08, 0.0])))
    b.append("\\node[dot, fill=ndGold] at %s {};" % pa.pt(G1, GAM))
    x0, x1, y0, y1 = 0.17, 0.23, 0.08, 0.13
    b.append("\\draw[line width=0.5pt, ndInk!70] %s rectangle %s;" % (pa.pt(x0, y0), pa.pt(x1, y1)))
    b.append("\\node[note, anchor=west] at %s {physical:\\\\ $T_2\\le2T_1$};" % N.pt(pa.P(0.03, 0.42)))
    b.append("\\node[note, anchor=south, text=ndRed] at %s {$T_2>2T_1$};" % N.pt(pa.P(0.305, 0.012)))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the physical wedge};" % N.pt(pa.P(0.18, 0) + np.array([0, -0.55])))

    # ---- (b) the estimates
    pb = N.Plot((x0, x1), (y0, y1), 4.4, 3.9, origin=(6.1, 0.75))
    b.append("\\fill[ndSky!70] %s -- %s -- %s -- %s -- cycle;" % (pb.pt(x0, x0 / 2), pb.pt(x1, x1 / 2), pb.pt(x1, y1), pb.pt(x0, y1)))
    b.append("\\fill[ndRose!70] %s -- %s -- %s -- %s -- cycle;" % (pb.pt(x0, y0), pb.pt(x1, y0), pb.pt(x1, x1 / 2), pb.pt(x0, x0 / 2)))
    b += pb.axes(xlabel="{\\scriptsize $1/T_1$}", ylabel="{\\scriptsize $1/T_2$}",
                 xticks=[(0.18, "$0.18$"), (0.2, "$0.20$"), (0.22, "$0.22$")], yticks=[(0.09, "$0.09$"), (0.11, "$0.11$")])
    b.append("\\draw[ink] %s -- %s;" % (pb.pt(x0, x0 / 2), pb.pt(x1, x1 / 2)))
    far = sorted(bad, key=lambda i: rows[i][1], reverse=True)[:4]
    for i in far:
        g1, gam = est[i]
        nx, ny = rows[i][2]
        b.append("\\draw[->, >=stealth, line width=0.6pt, ndRed] %s -- %s;" % (pb.pt(g1, gam), pb.pt(nx, ny)))
        b.append("\\draw[ndRed, fill=white, line width=0.5pt] %s circle (1.3pt);" % pb.pt(nx, ny))
    for i, (g1, gam) in enumerate(est):
        if not (x0 < g1 < x1 and y0 < gam < y1):
            continue
        col = "ndBlue" if rows[i][0] else "ndRed"
        b.append("\\fill[%s] %s circle (1.4pt);" % (col, pb.pt(g1, gam)))
    b.append("\\node[dot, fill=ndGold, minimum size=4.5pt] at %s {};" % pb.pt(G1, GAM))
    b.append("\\node[note, anchor=south west, text=ndBlue] at %s {physical};" % pb.pt(x0 + 0.003, 0.122))
    b.append("\\node[note, anchor=north east, text=ndRed] at %s {certified\\\\ non-physical:\\\\ %d of %d};" % (pb.pt(x1 - 0.002, 0.096), len(bad), len(rows)))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) sixty estimates, $300$ shots};" % N.pt(pb.P(0.2, y0) + np.array([0, -0.55])))

    # ---- (c) the qubit with the hidden defect
    lim = 3.0
    pc = N.Plot((0, 4.9), (-lim, lim), 4.6, 3.9, origin=(12.0, 0.75))
    tt = np.linspace(0.02, 4.88, 1200)
    gg = np.array([NE.rate(t) for t in tt])
    # shade where the rate is negative
    neg = gg < 0
    idx = np.where(neg)[0]
    for sg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
        if len(sg) > 1:
            b.append("\\fill[ndRose!70] %s rectangle %s;" % (pc.pt(tt[sg[0]], -lim), pc.pt(tt[sg[-1]], lim)))
    b += pc.axes(xlabel="{\\scriptsize $t$}", xticks=[(1.81, "$1.81$"), (3.21, "$3.21$")],
                 yticks=[(-2, "$-2$"), (2, "$2$")], yzero=0)
    for vals, sty in [(gg, "line width=0.8pt, ndGold!85!ndInk"), (np.sqrt(2.5) * np.clip(-gg, 0, None), "line width=1.3pt, ndRed")]:
        ok = np.abs(vals) <= lim
        idx = np.where(ok)[0]
        for sg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(sg) > 1:
                b.append(pc.curve(tt[sg], vals[sg], sty))
    for t, g, val in dd:
        y = np.sqrt(max(val, 0))
        if y <= lim:
            b.append("\\fill[ndRed!80!ndInk] %s circle (1.1pt);" % pc.pt(t, y))
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndGold!60!ndInk] at %s {rate $\\gamma(t)$};" % pc.pt(0.3, -1.1))
    b.append("\\node[note, anchor=south, text=ndRed] (cd) at %s {certified distance\\\\ $\\sqrt{5/2}\\,\\abs\\gamma$ where $\\gamma<0$};" % pc.pt(4.05, -2.75))
    b.append("\\draw[pointer] (cd.north) to[bend right=25] %s;" % pc.pt(2.75, np.sqrt(2.5) * abs(NE.rate(2.75)) + 0.05))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the qubit with the hidden defect};" % N.pt(pc.P(2.45, -lim) + np.array([0, -0.55])))
    N.write("ch11_physical", b)
    return frac, len(bad), len(rows), dd


if __name__ == "__main__":
    frac, nbad, n, dd = main()
    print("non-physical estimates: %d of %d (%.0f%%)" % (nbad, n, 100 * frac))
    print("defect: %d certified instants, largest certified squared distance in window %.3f" % (len(dd), dd[:, 2][dd[:, 2] < 9].max()))
    print("written figures/ch11_physical.tex")
