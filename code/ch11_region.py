"""ch11_region.py -- a certified confidence region that respects physics.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi), with cvxpy and
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

The relaxation-limited qubit of Figure ch11_physical (g1 = 1/T1 = 0.2,
g_phi = 0.005, Gamma = 1/T2 = 0.105, no Hamiltonian in the rotating frame),
whose generator is z0 = (A, beta) in the notation of code/ch11_physical.py.
An experiment returns the estimate z^ = z0 + sigma xi of all twelve entries,
with xi standard normal and sigma = 0.006 (as from a tomography of the
generator).  The naive 95% region for (1/T1, 1/T2) = (-A_33, -(A_11 + A_22)/2)
ignores physics: it is the ellipse
    (a - g1^)^2 + 2 (b - Gamma^)^2 <= sigma^2 chi2_2(0.95),   chi2_2(0.95) = 5.99,
the shadow of the ball |z - z^|^2 <= sigma^2 chi2_2(0.95).  The physical region
is the shadow of the generators in the GKSL cone (gamma >= 0) that fit as well
as the best of them, up to the same allowance:
    C = { z = M p : [[R, -S], [S, R]] >= 0,  |z - z^|^2 <= rho_min + sigma^2 chi2_2(0.95) },
with rho_min the certified squared distance of z^ from the cone
(ch11_physical.certify).  C is convex, and so is its shadow; its support
function h(d) = max_{z in C} d . (-A_33, -(A_11 + A_22)/2) is computed in 72
directions d by pysparsepmi.pmi_optimize (a linear objective, the 6x6 PMI and
one quadratic constraint, order 1: exact, since the problem is convex), and
every value is checked against a direct cvxpy solve to 1e-6 (all 144 SCS
statuses 'optimal').  The supporting half-planes, with the half-plane
1/T2 >= 1/(2 T1) that contains every GKSL generator, intersect in the certified
outer polygon (drawn).  (The edge normal itself is not among the directions:
there the maximiser is not unique and SCS reports 'optimal_inaccurate'.)  The
maximisers span an inner polygon; the two differ by less than 2e-4 in every
direction halfway between two of the 72, and by less than 2% in area.

The shadow is not the part of the ellipse inside the wedge T2 <= 2T1: it is
the shadow of a twelve-dimensional set, and a face that the plane does not
show also cuts it.  Every GKSL generator has |beta_3| <= -A_33, since
|gamma_12|^2 <= gamma_11 gamma_22 (checked on 2000 random generators); for a
phase-covariant one this says that the equilibrium polarisation beta_3/g1 lies
in [-1, 1], that the qubit does not relax to a point outside the Bloch ball.
The qubit relaxes to |0> and lies on that face, so the measured beta_3^ bounds
1/T1 from below.
(a) A data set whose estimate (g1^, Gamma^) lies outside the wedge (seed 3):
    62% of the naive ellipse is non-physical; the certified squared distance
    to the GKSL generators is 1.12 sigma^2, and the physical region, cut by the
    edge, has 0.38 of the area of the ellipse.
(b) A data set whose estimate lies inside the wedge, with beta_3^ = 0.1994 >
    g1^ = 0.1955 (seed 11): only 13% of the ellipse is non-physical, but the
    region is cut on the left as well, where 1/T1 >= 0.1872 against 0.1808 for
    the ellipse, and has 0.65 of its area.
Over 300 data sets the naive ellipse contains the truth in 95.0% of cases and
the physical region in 96.7% (direct cvxpy solves of the profile problem):
smaller, and no less reliable.
Gold: the truth; red: the least-squares estimate; blue: the nearest GKSL
generator, with the arrow between them.
Writes figures/ch11_region.tex.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvxpy as cp
from scipy.stats import chi2

try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp

import needham as N
import ch11_physical as PH                    # M, PAIRS, ANTI, certify, direct, slice_generator, kossakowski

G1, GAM = PH.G1, PH.GAM
SIG = 0.006
C2 = chi2.ppf(0.95, 2)
A0, B0 = PH.slice_generator(G1, GAM)
Z0 = np.concatenate([A0.ravel(), B0])
LG1 = np.zeros(12)
LG1[8] = -1.0                                 # 1/T1 = -A_33
LGA = np.zeros(12)
LGA[0] = LGA[4] = -0.5                        # 1/T2 = -(A_11 + A_22)/2
NDIR = 72
SEEDS = (3, 11)


def directions():
    ang = 2 * np.pi * np.arange(NDIR) / NDIR
    return np.stack([np.cos(ang), np.sin(ang)], 1)


def clip(P):
    """The part of the convex polygon P in the wedge 1/T2 >= 1/(2 T1) (Sutherland-Hodgman, one edge)."""
    f = lambda q: q[1] - q[0] / 2
    out = []
    for k in range(len(P)):
        p, q = P[k], P[(k + 1) % len(P)]
        if f(p) >= 0:
            out.append(p)
        if f(p) * f(q) < 0:
            out.append(p + f(p) / (f(p) - f(q)) * (q - p))
    return np.array(out)


def gap(r):
    """Largest excess of the outer polygon over the inner one, in the directions halfway between."""
    D, n = r["D"], len(r["D"])
    out = []
    for k in range(n):
        m = D[k] + D[(k + 1) % n]
        m /= np.linalg.norm(m)
        out.append(np.max(r["outer"] @ m) - np.max(r["inner"] @ m))
    return max(out)


def data(seed):
    return Z0 + SIG * np.random.default_rng(seed).standard_normal(12)


def polys(zh):
    """The 12 variables, the affine forms (M p)_i, and the 6x6 PMI of ch11_physical.certify."""
    v = psp.polyvar(12)
    zero = psp.Polynomial.zero(12)
    lin = []
    for i in range(12):
        e = zero
        for k in range(12):
            if abs(PH.M[i, k]) > 1e-14:
                e = e + v[k] * PH.M[i, k]
        lin.append(e)
    R = [[v[3], v[4], v[5]], [v[4], v[6], v[7]], [v[5], v[7], v[8]]]
    S = [[zero, v[9], v[10]], [zero - v[9], zero, v[11]], [zero - v[10], zero - v[11], zero]]
    K = [[None] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            K[i][j] = K[i + 3][j + 3] = R[i][j]
            K[i + 3][j] = S[i][j]
            K[i][j + 3] = zero - S[i][j]
    return zero, lin, psp.PolyMatrix(K, nvars=12)


def support_cert(zh, rad2, d):
    """Certified upper bound on the support function in the direction d, and a maximiser."""
    zero, lin, K = polys(zh)
    f = zero
    for i in range(12):
        c = d[0] * LG1[i] + d[1] * LGA[i]
        if c:
            f = f + lin[i] * (-c)
    ball = zero + rad2
    for i in range(12):
        e = lin[i] - zh[i]
        ball = ball - e * e
    res = psp.pmi_optimize([[f]], ineqs=[K, ball], order=1, solver="SCS", eps=1e-9, max_iters=200000)
    return -res.value, res.status, res.minimizer()


def cvx_setup():
    Z = cp.Variable((6, 6), symmetric=True)
    th = cp.Variable(3)
    R, S = Z[:3, :3], Z[3:, :3]
    par = cp.hstack([th] + [R[l, m] for l, m in PH.PAIRS] + [S[m, l] for l, m in PH.ANTI])
    return par, [Z >> 0, Z[:3, :3] == Z[3:, 3:], S + S.T == 0]


def support_direct(zh, rad2, d):
    par, cons = cvx_setup()
    z = PH.M @ par
    pr = cp.Problem(cp.Maximize(d[0] * (LG1 @ z) + d[1] * (LGA @ z)), cons + [cp.sum_squares(z - zh) <= rad2])
    pr.solve(solver="SCS", eps=1e-10, max_iters=200000)
    return pr.value


def profile_direct(zh, a, b):
    """Least squared distance from z^ of a physical generator with (1/T1, 1/T2) = (a, b)."""
    par, cons = cvx_setup()
    z = PH.M @ par
    pr = cp.Problem(cp.Minimize(cp.sum_squares(z - zh)), cons + [LG1 @ z == a, LGA @ z == b])
    pr.solve(solver="SCS", eps=1e-10, max_iters=200000)
    return pr.value


def area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def region(seed):
    zh = data(seed)
    rho, status, x = PH.certify(zh[:9].reshape(3, 3), zh[9:])
    assert status == "optimal" and abs(rho - PH.direct(zh[:9].reshape(3, 3), zh[9:])) < 1e-8
    near = PH.M @ x
    rad2 = rho + SIG ** 2 * C2
    D = directions()
    h, inner = [], []
    for d in D:
        val, status, x = support_cert(zh, rad2, d)
        assert status == "optimal", (seed, d, status)
        ref = support_direct(zh, rad2, d)
        assert abs(val - ref) < 1e-6, (seed, d, val, ref)
        z = PH.M @ x
        assert np.linalg.eigvalsh(PH.kossakowski(x)).min() > -1e-6 and np.sum((z - zh) ** 2) < rad2 + 1e-6
        h.append(val)
        inner.append((LG1 @ z, LGA @ z))
    h, inner = np.array(h), np.array(inner)
    n = len(D)
    outer = clip(np.array([np.linalg.solve(np.stack([D[k], D[(k + 1) % n]]), [h[k], h[(k + 1) % n]]) for k in range(n)]))
    return dict(zh=zh, g=(LG1 @ zh, LGA @ zh), rho=rho, near=(LG1 @ near, LGA @ near), rad2=rad2, D=D, h=h,
                inner=inner, outer=outer)


def ellipse(g, n=241):
    th = np.linspace(0, 2 * np.pi, n)
    return np.stack([g[0] + SIG * np.sqrt(C2) * np.cos(th), g[1] + SIG * np.sqrt(C2 / 2) * np.sin(th)], 1)


def wedge_fraction(g, n=400):
    """Fraction of the naive ellipse outside the wedge (a fine grid)."""
    u = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(u, u)
    ok = X ** 2 + Y ** 2 <= 1
    a = g[0] + SIG * np.sqrt(C2) * X[ok]
    b = g[1] + SIG * np.sqrt(C2 / 2) * Y[ok]
    return np.mean(b < a / 2)


def coverage(n=300, seed=1234):
    rng = np.random.default_rng(seed)
    cn = cp_ = 0
    for _ in range(n):
        zh = Z0 + SIG * rng.standard_normal(12)
        cn += (G1 - LG1 @ zh) ** 2 + 2 * (GAM - LGA @ zh) ** 2 <= SIG ** 2 * C2
        rho = PH.direct(zh[:9].reshape(3, 3), zh[9:])
        cp_ += profile_direct(zh, G1, GAM) - rho <= SIG ** 2 * C2
    return cn / n, cp_ / n


def checks(regs, cov):
    # every GKSL generator has |beta_3| <= -A_33: no qubit relaxes outside the Bloch ball
    rng = np.random.default_rng(5)
    for _ in range(2000):
        G = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        gam = G @ G.conj().T
        par = np.concatenate([rng.standard_normal(3), [gam.real[l, m] for l, m in PH.PAIRS], [gam.imag[l, m] for l, m in PH.ANTI]])
        assert np.allclose(PH.kossakowski(par), gam)
        z = PH.M @ par
        assert abs(z[11]) <= -z[8] + 1e-12
    assert abs(Z0[11] + Z0[8]) < 1e-15                              # the truth is on that face (z_eq = 1)
    # the naive region is the shadow of the ball: the support function of the ellipse
    for r in regs:
        ell = ellipse(r["g"], 2001)
        inner_ok = all(np.max(r["inner"] @ d) <= hk + 1e-7 for d, hk in zip(r["D"], r["h"]))
        assert inner_ok
        assert np.max([np.max(r["inner"] @ d) - hk for d, hk in zip(r["D"], r["h"])]) < 1e-7
        # inner and outer polygons agree to a small fraction of the size of the region
        assert 0 <= gap(r) < 2e-4, gap(r)
        # the outer polygon lies in the wedge (the edge normal is one of the directions)
        assert np.all(r["outer"][:, 1] >= r["outer"][:, 0] / 2 - 1e-6)
        assert np.all(r["outer"][:, 0] >= r["zh"][11] - np.sqrt(r["rad2"]) - 1e-6)
    a, b = regs
    assert a["g"][1] < a["g"][0] / 2 and a["rho"] > 0                 # (a) estimate outside the wedge
    assert b["g"][1] > b["g"][0] / 2 and b["zh"][11] > b["g"][0]       # (b) inside, but beta_3^ > g1^
    ea = np.pi * SIG ** 2 * C2 / np.sqrt(2)
    assert abs(area(a["outer"]) / ea - 0.385) < 0.005 and abs(area(b["outer"]) / ea - 0.65) < 0.005
    assert abs(wedge_fraction(a["g"]) - 0.62) < 0.005 and abs(wedge_fraction(b["g"]) - 0.13) < 0.005
    for r in regs:                                                    # outer and inner areas within 2%
        assert area(r["inner"]) <= area(r["outer"]) < 1.02 * area(r["inner"])
    # (b) the left edge of the region lies right of the ellipse's: the face |z_eq| <= 1 cuts it
    assert b["outer"][:, 0].min() > b["g"][0] - SIG * np.sqrt(C2) + 0.005
    assert abs(b["outer"][:, 0].min() - 0.1872) < 2e-4 and abs(b["g"][0] - SIG * np.sqrt(C2) - 0.1808) < 1e-4
    assert abs(a["rho"] / SIG ** 2 - 1.12) < 0.01 and abs(b["zh"][11] - 0.1994) < 1e-4 and abs(b["g"][0] - 0.1955) < 1e-4
    assert abs(cov[0] - 0.95) < 0.005 and abs(cov[1] - 0.967) < 0.005


def panel(r, origin, title, lim):
    (x0, x1), (y0, y1) = lim
    p = N.Plot((x0, x1), (y0, y1), 6.4, 6.4 * (y1 - y0) / (x1 - x0), origin=origin)
    b = ["\\begin{scope}",
         "\\clip %s rectangle %s;" % (p.pt(x0, y0), p.pt(x1, y1)),
         "\\fill[ndSky!45] %s -- %s -- %s -- %s -- cycle;" % (p.pt(x0, x0 / 2), p.pt(x1, x1 / 2), p.pt(x1, y1), p.pt(x0, y1)),
         "\\fill[ndRose!70] %s -- %s -- %s -- %s -- cycle;" % (p.pt(x0, y0), p.pt(x1, y0), p.pt(x1, x1 / 2), p.pt(x0, x0 / 2)),
         "\\draw[ink] %s -- %s;" % (p.pt(x0, x0 / 2), p.pt(x1, x1 / 2))]
    b.append("\\fill[ndBlue, fill opacity=0.28] %s -- cycle;" % " -- ".join(p.pt(*q) for q in r["outer"]))
    b.append("\\draw[line width=0.9pt, ndBlue] %s -- cycle;" % " -- ".join(p.pt(*q) for q in r["outer"]))
    b.append(N.polyline([p.P(*q) for q in ellipse(r["g"])], "line width=0.9pt, ndRed, dash pattern=on 3pt off 2pt"))
    b.append("\\end{scope}")
    b += p.axes(xlabel="{\\scriptsize $1/T_1$}", ylabel="{\\scriptsize $1/T_2$}",
                xticks=[(v, "$%.2f$" % v) for v in np.arange(np.ceil(x0 * 100) / 100, x1, 0.01)],
                yticks=[(v, "$%.2f$" % v) for v in np.arange(np.ceil(y0 * 100) / 100, y1, 0.01)])
    b.append("\\draw[->, >=stealth, line width=0.6pt, ndInk!70, shorten >=1.5pt] %s -- %s;" % (p.pt(*r["g"]), p.pt(*r["near"])))
    b.append("\\fill[ndRed] %s circle (1.6pt);" % p.pt(*r["g"]))
    b.append("\\fill[ndBlue] %s circle (1.6pt);" % p.pt(*r["near"]))
    b.append("\\node[dot, fill=ndGold, minimum size=5pt] at %s {};" % p.pt(G1, GAM))
    b.append("\\node[font=\\small, anchor=north] at %s {%s};" % (N.pt(p.P((x0 + x1) / 2, y0) + np.array([0, -0.55])), title))
    return p, b


def main():
    regs = [region(s) for s in SEEDS]
    cov = coverage()
    checks(regs, cov)
    a, bb = regs
    b = []
    la = ((0.183, 0.225), (0.085, 0.114))
    pa, body = panel(a, (0.9, 0.8), "(a) the estimate outside the wedge", la)
    b += body
    b.append("\\node[note, anchor=south east, text=ndRed] at %s {$T_2>2T_1$};" % pa.pt(0.2245, 0.0858))
    b.append("\\node[note, anchor=north west] at %s {$T_2\\le2T_1$};" % pa.pt(0.1835, 0.1135))
    b.append("\\node[note, anchor=south west, text=ndRed] (n) at %s {naive $95\\%%$ ellipse};" % pa.pt(0.1835, 0.0855))
    b.append("\\draw[pointer] (n.north) to[bend left=20] %s;" % pa.pt(0.1948, 0.0928))
    b.append("\\node[note, anchor=east, text=ndBlue] (c) at %s {certified\\\\ physical\\\\ region};" % pa.pt(0.1905, 0.1065))
    b.append("\\draw[pointer] (c.east) to[bend left=20] %s;" % pa.pt(0.1975, 0.104))
    lb = ((0.172, 0.214), (0.0925, 0.1235))
    pb, body = panel(bb, (9.4, 0.8), "(b) the estimate inside the wedge", lb)
    b += body
    b.append("\\node[note, anchor=south east, text=ndRed] at %s {$T_2>2T_1$};" % pb.pt(0.2135, 0.0933))
    xl = bb["outer"][:, 0].min()
    b.append("\\node[note, anchor=north west] (z) at %s {cut by $\\abs{\\beta_3}\\le1/T_1$:\\\\ no relaxation outside the ball};" % pb.pt(0.1725, 0.1233))
    b.append("\\draw[pointer] (z.south) to[bend right=30] %s;" % pb.pt(xl + 0.0003, bb["outer"][np.argmin(bb["outer"][:, 0]), 1]))
    b.append("\\node[note, anchor=south east] (t) at %s {truth};" % pb.pt(0.2135, 0.0985))
    b.append("\\draw[pointer] (t.west) to[bend left=15] %s;" % pb.pt(G1 + 0.0004, GAM - 0.0002))
    N.write("ch11_region", b)
    return regs, cov


if __name__ == "__main__":
    regs, cov = main()
    ea = np.pi * SIG ** 2 * C2 / np.sqrt(2)
    for s, r in zip(SEEDS, regs):
        print("seed %d: estimate (%.4f, %.4f), beta_3^ %.4f, certified rho_min/sigma^2 %.2f, nearest physical (%.4f, %.4f)"
              % (s, r["g"][0], r["g"][1], r["zh"][11], r["rho"] / SIG ** 2, *r["near"]))
        print("   area: region / ellipse %.2f; ellipse outside the wedge %.2f; 1/T1 in [%.4f, %.4f] (ellipse [%.4f, %.4f])"
              % (area(r["outer"]) / ea, wedge_fraction(r["g"]), r["outer"][:, 0].min(), r["outer"][:, 0].max(),
                 r["g"][0] - SIG * np.sqrt(C2), r["g"][0] + SIG * np.sqrt(C2)))
    print("coverage over 300 data sets: naive %.3f, physical %.3f" % cov)
    print("written figures/ch11_region.tex")
