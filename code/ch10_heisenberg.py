"""ch10_heisenberg.py -- hard and soft constraints on the Heisenberg group.

R^3 with the frame X = d/dx - (y/2) d/dz, Y = d/dy + (x/2) d/dz, Z = d/dz,
[X, Y] = Z, the left-invariant Riemannian metric g making X, Y, Z orthonormal,
the distribution D = span{X, Y}, and the penalised metrics g_q = diag(1, 1, q)
in this frame.  A curve is horizontal if and only if
z' = (x y' - y x')/2, so that z is the signed area swept out by its shadow
(x, y) in the plane.  We join the origin to (0, 0, A), A = 1.

The geodesics of g_q.  With the momenta P_X, P_Y, P_Z = lambda (constant) of
the Hamiltonian (P_X^2 + P_Y^2 + P_Z^2/q)/2, the vector (P_X, P_Y) turns at the
rate lambda, so that the shadow of every geodesic is a circle (or a point),
and z = (area swept) + (lambda/q) t.  To return the shadow to the origin at
t = 1 needs lambda = 2 pi k; with k = 1 the radius is r_q, r_q^2 = (A - 2 pi/q)/pi,
and the length sqrt(4 pi A - 4 pi^2/q), real only for q > 2 pi/A.  The vertical
segment is a geodesic for every q, of length sqrt(q) A.  Hence the shortest
geodesic is the vertical segment for q <= 2 pi/A, and for q > 2 pi/A a circle
of helices branches off it (every rotation about the z-axis gives another),
with shadows of radius r_q = 0.262, 0.389, 0.488, 0.546 for q = 8, 12, 25, 100,
tending to the circle of area A, of radius 1/sqrt(pi) = 0.564; the lengths
2.763, 3.046, 3.315, 3.489 increase to the sub-Riemannian distance
2 sqrt(pi A) = 3.545.  This is Dido's problem: the shortest horizontal curve
encloses the given area with the least perimeter, and its shadow is a circle.
A discretised minimisation of the g_q-energy from random starts finds nothing
shorter.

The nonholonomic motion.  The Lagrange-d'Alembert equations for a free
particle (kinetic energy of g) under the same constraint, x'' = mu y/2,
y'' = -mu x/2, 0 = mu, force mu = 0: the shadow moves on straight lines and
z' = (x y' - y x')/2 stays constant.  From the origin the particle stays in
the plane z = 0 and never reaches (0, 0, A), although every point is joined to
the origin by horizontal curves.

(a) The view in space: the planes of D at points of the plane z = 0, the
    vertical segment (gold), the penalised geodesics for q = 8, 12, 25 and 100
    (blue), the sub-Riemannian geodesic (red) and nonholonomic paths (green).
(b) Their shadows in the plane.
(c) The length of the shortest g_q-geodesic against q.
Writes figures/ch10_heisenberg.tex.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import needham as N

A = 1.0
QS = (8, 12, 25, 100)
QCOL = {8: "ndBlue!35", 12: "ndBlue!55", 25: "ndBlue!75", 100: "ndBlue"}


def radius(q):
    return np.sqrt((A - 2 * np.pi / q) / np.pi) if np.isfinite(q) else np.sqrt(A / np.pi)


def helix(q, th0=0.0, n=241):
    """The shortest g_q-geodesic for q > 2 pi/A (q = inf: the sub-Riemannian one), from the origin to (0,0,A)."""
    t = np.linspace(0, 1, n)
    lam, r = 2 * np.pi, radius(q)
    w = -1j * r * (np.exp(1j * (lam * t + th0)) - np.exp(1j * th0))
    z = 0.5 * r ** 2 * (lam * t - np.sin(lam * t)) + (lam * t / q if np.isfinite(q) else 0.0)
    return np.stack([w.real, w.imag, z], 1)


def length(q):
    if np.isfinite(q) and q <= 2 * np.pi / A:
        return np.sqrt(q) * A
    return np.sqrt(4 * np.pi * A - (4 * np.pi ** 2 / q if np.isfinite(q) else 0.0))


def rhs(t, s, q):
    x, y, z, px, py, lam = s
    return [px, py, -y / 2 * px + x / 2 * py + lam / q, -lam * py, lam * px, 0.0]


def discrete_min_length(q, n=80, starts=8, rng=np.random.default_rng(0)):
    def energy(v):
        P = np.vstack([[0, 0, 0], v.reshape(n - 1, 3), [0, 0, A]])
        d = np.diff(P, axis=0) * n
        m = (P[1:] + P[:-1]) / 2
        c = d[:, 2] - (m[:, 0] * d[:, 1] - m[:, 1] * d[:, 0]) / 2
        return 0.5 * np.sum(d[:, 0] ** 2 + d[:, 1] ** 2 + q * c * c) / n
    best = np.inf
    t = np.linspace(0, 1, n + 1)[1:-1]
    for _ in range(starts):
        r0, th = rng.uniform(0, 0.6), rng.uniform(0, 2 * np.pi)
        v0 = np.stack([r0 * (np.cos(2 * np.pi * t + th) - np.cos(th)), r0 * (np.sin(2 * np.pi * t + th) - np.sin(th)), A * t], 1)
        v0 += rng.normal(0, 0.05, v0.shape)
        r = minimize(energy, v0.ravel(), method="L-BFGS-B", options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10})
        best = min(best, r.fun)
    return np.sqrt(2 * best)


def nonholonomic(p0, v0, T=1.0):
    """Lagrange-d'Alembert motion: x'' = mu y/2, y'' = -mu x/2, z'' = mu, with mu fixed by the constraint."""
    def f(t, s):
        x, y, z, vx, vy, vz = s
        mu = 0.0 * (1 + (x * x + y * y) / 4)        # the constraint forces mu (1 + (x^2+y^2)/4) = 0
        return [vx, vy, vz, mu * y / 2, -mu * x / 2, mu]
    vz = (p0[0] * v0[1] - p0[1] * v0[0]) / 2
    return solve_ivp(f, (0, T), [*p0, v0[0], v0[1], vz], rtol=1e-11, atol=1e-12, dense_output=True)


def checks():
    # the closed form agrees with Hamilton's equations, ends at (0,0,A), is a geodesic of the stated length
    for q in QS:
        lam, r = 2 * np.pi, radius(q)
        sol = solve_ivp(rhs, (0, 1), [0, 0, 0, lam * r, 0, lam], args=(q,), rtol=1e-11, atol=1e-12, t_eval=np.linspace(0, 1, 241))
        assert np.allclose(sol.y[:3].T, helix(q), atol=1e-8)
        assert np.allclose(helix(q)[-1], [0, 0, A], atol=1e-12)
        assert abs(np.hypot(lam * r, lam / np.sqrt(q)) - length(q)) < 1e-12        # constant speed sqrt(v^2 + lam^2/q)
    assert np.allclose([radius(q) for q in QS], [0.262, 0.389, 0.488, 0.546], atol=1e-3)
    assert np.allclose([length(q) for q in QS], [2.763, 3.046, 3.315, 3.489], atol=1e-3)
    assert abs(length(np.inf) - 2 * np.sqrt(np.pi * A)) < 1e-12 and abs(radius(np.inf) - 1 / np.sqrt(np.pi)) < 1e-12
    # the branches meet at q = 2 pi/A, and the helix is shorter beyond it
    q0 = 2 * np.pi / A
    assert abs(np.sqrt(4 * np.pi * A - 4 * np.pi ** 2 / q0) - np.sqrt(q0) * A) < 1e-12
    for q in (7, 12, 50):
        assert length(q) < np.sqrt(q) * A
    # the sub-Riemannian limit is horizontal, its shadow encloses the area A
    h = helix(np.inf, n=4001)
    dz = np.diff(h[:, 2]); m = (h[1:] + h[:-1]) / 2; d = np.diff(h, axis=0)
    assert np.allclose(dz, (m[:, 0] * d[:, 1] - m[:, 1] * d[:, 0]) / 2, atol=1e-7)
    # a discretised minimisation of the energy finds nothing shorter
    for q in (4, 12):
        Ld = discrete_min_length(q)
        assert length(q) <= Ld < 1.02 * length(q), (q, Ld, length(q))
    # the nonholonomic particle: straight shadows, constant z', and from the origin z stays 0
    for p0, v0 in [((0.3, -0.2, 0.1), (0.5, 0.8)), ((0, 0, 0), (0.6, 0.4))]:
        sol = nonholonomic(np.array(p0), np.array(v0))
        ts = np.linspace(0, 1, 51)
        s = sol.sol(ts)
        assert np.allclose(s[0], p0[0] + v0[0] * ts) and np.allclose(s[1], p0[1] + v0[1] * ts)
        assert np.allclose(s[2], p0[2] + (p0[0] * v0[1] - p0[1] * v0[0]) / 2 * ts)
    assert np.allclose(nonholonomic(np.zeros(3), np.array([0.6, 0.4])).sol(np.linspace(0, 1, 11))[2], 0)


def main():
    checks()
    b = []
    # ---- (a) the view in space
    V = N.View(az=-62.0, el=13.0, R=3.0, shift=(2.7, 0.9))
    o = np.zeros(3)
    for e, lab, off in [((0.95, 0, 0), "x", (0.0, -0.15)), ((0, 0.95, 0), "y", (0.14, 0.0)), ((0, 0, 1.25), "z", (0.0, 0.12))]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (V.pt(o), V.pt(e)))
        b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%s$};" % (N.pt(V.P(e) + np.array(off)), lab))
    sq = [(-1.1, -1.1, 0), (1.1, -1.1, 0), (1.1, 1.1, 0), (-1.1, 1.1, 0)]
    b.append("\\fill[ndPaper] %s -- cycle;" % " -- ".join(V.pt(p) for p in sq))
    b.append("\\draw[faint] %s -- cycle;" % " -- ".join(V.pt(p) for p in sq))
    # planes of D at points of the plane z = 0
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        p = np.array([0.92 * np.cos(ang), 0.92 * np.sin(ang), 0.0])
        Xv = np.array([1, 0, -p[1] / 2]); Yv = np.array([0, 1, p[0] / 2])
        Xv, Yv = 0.13 * Xv / np.linalg.norm(Xv), 0.13 * Yv / np.linalg.norm(Yv)
        quad = [p - Xv - Yv, p + Xv - Yv, p + Xv + Yv, p - Xv + Yv]
        b.append("\\filldraw[fill=ndSky, fill opacity=0.9, draw=ndBlue!60, line width=0.3pt] %s -- cycle;" % " -- ".join(V.pt(c) for c in quad))
    # nonholonomic paths from the origin: straight lines in the plane z = 0
    for ang in (1.9, 3.6):
        b.append("\\draw[line width=1.0pt, ndGreen, ->, >=stealth] %s -- %s;" % (V.pt(o), V.pt((0.62 * np.cos(ang), 0.62 * np.sin(ang), 0))))
    b.append("\\draw[line width=1.1pt, ndGold, dash pattern=on 3pt off 2pt] %s -- %s;" % (V.pt(o), V.pt((0, 0, A))))
    shadow = helix(np.inf) * np.array([1, 1, 0])
    b.append(N.polyline([V.P(p) for p in shadow], "line width=0.7pt, ndRed!70, dash pattern=on 2pt off 1.5pt"))
    for q in (12, 100):
        b.append(N.polyline([V.P(p) for p in helix(q)], "line width=0.9pt, %s" % QCOL[q]))
    b.append(N.polyline([V.P(p) for p in helix(np.inf)], "line width=1.5pt, ndRed"))
    b.append("\\node[dot] at %s {};" % V.pt(o))
    b.append("\\node[dot, fill=ndRed] at %s {};" % V.pt((0, 0, A)))
    b.append("\\node[font=\\scriptsize, anchor=north] at %s {$0$};" % N.pt(V.P(o) + np.array([0.02, -0.06])))
    b.append("\\node[font=\\scriptsize, anchor=south west] at %s {$(0,0,A)$};" % N.pt(V.P((0, 0, A)) + np.array([0.08, 0.04])))
    hp = helix(np.inf)[60]
    b.append("\\node[note, anchor=west, text=ndRed] (sr) at %s {sub-Riemannian:\\\\ the lift of a circle;\\\\ blue: $q=12$, $100$};" % N.pt(V.P(o) + np.array([1.95, 2.0])))
    b.append("\\draw[pointer] (sr.west) to[bend right=15] %s;" % N.pt(V.P(hp) + np.array([0.06, 0.02])))
    b.append("\\node[note, anchor=east, text=ndGold!55!ndInk] (rv) at %s {vertical: length $\\sqrt q\\,A$,\\\\ shortest for $q\\le2\\pi/A$};" % N.pt(V.P(o) + np.array([-0.85, 2.55])))
    b.append("\\draw[pointer] (rv.east) to[bend left=10] %s;" % N.pt(V.P((0, 0, 0.82 * A)) + np.array([-0.05, 0.0])))
    b.append("\\node[note, anchor=east, text=ndGreen] (nh) at %s {nonholonomic:\\\\ straight lines, $z=0$};" % N.pt(V.P(o) + np.array([-1.35, -0.75])))
    b.append("\\draw[pointer] (nh.east) to[bend right=10] %s;" % N.pt(V.P((0.45 * np.cos(3.6), 0.45 * np.sin(3.6), 0)) + np.array([0.0, -0.05])))
    b.append("\\node[note, anchor=north west] (pl) at %s {planes of $D$\\\\ twist about $z$};" % N.pt(V.P(o) + np.array([2.55, -1.2])))
    b.append("\\draw[pointer] (pl.north west) to[bend left=10] %s;" % N.pt(V.P((0.92 * np.cos(5.5), 0.92 * np.sin(5.5), 0)) + np.array([0.08, -0.02])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) geodesics from $0$ to $(0,0,A)$};" % N.pt(V.P(o) + np.array([0.0, -1.6])))

    # ---- (b) the shadows in the plane
    c = np.array([8.7, 1.15])
    S = 2.2
    P = lambda v: c + S * np.asarray(v[:2])
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$x$};" % (N.pt(c + [-0.4 * S, 0]), N.pt(c + [1.25 * S, 0])))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$y$};" % (N.pt(c + [0, -0.5 * S]), N.pt(c + [0, 1.35 * S])))
    lim = helix(np.inf)
    b.append("\\fill[ndRose!60] %s -- cycle;" % " -- ".join(N.pt(P(p)) for p in lim))
    for q in QS:
        b.append(N.polyline([P(p) for p in helix(q)], "line width=0.8pt, %s" % QCOL[q]))
    b.append(N.polyline([P(p) for p in lim], "line width=1.5pt, ndRed"))
    for ang in (1.9, 3.6):
        b.append("\\draw[line width=1.0pt, ndGreen, ->, >=stealth] %s -- %s;" % (N.pt(P(o)), N.pt(P((0.45 * np.cos(ang), 0.45 * np.sin(ang))))))
    b.append("\\node[dot] at %s {};" % N.pt(P(o)))
    b.append("\\node[note, text=ndRed!80!ndInk] at %s {area $A$};" % N.pt(P((0.0, 0.62))))
    b.append("\\node[note, anchor=west, text=ndBlue] (qq) at %s {$q=8,12,25,100$:\\\\ radius $\\sqrt{(A-2\\pi/q)/\\pi}$};" % N.pt(P((0.62, 0.95))))
    b.append("\\draw[pointer] (qq.south west) to[bend right=15] %s;" % N.pt(P(helix(25)[95]) + np.array([0.03, 0.03])))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) shadows in the plane};" % N.pt(c + [0.45 * S, -0.62 * S]))

    # ---- (c) the length against q
    pl = N.Plot((0, 105), (0, 4.4), 4.1, 3.4, origin=(13.6, 0.25))
    b += pl.axes(xlabel="{\\scriptsize $q$}", ylabel="{\\scriptsize length}", xticks=[(2 * np.pi, "$2\\pi$"), (50, "$50$"), (100, "$100$")],
                 yticks=[(1, "$1$"), (2 * np.sqrt(np.pi), "$2\\sqrt\\pi$")])
    qq = np.linspace(1, 19.4, 200)
    b.append(pl.curve(qq, np.sqrt(qq) * A, "line width=0.9pt, ndGold, dash pattern=on 3pt off 2pt"))
    qs = np.concatenate([np.linspace(1, 2 * np.pi, 60), np.linspace(2 * np.pi, 105, 400)])
    b.append(pl.curve(qs, [length(q) for q in qs], "line width=1.4pt, ndBlue"))
    b.append("\\draw[line width=0.7pt, ndRed, dash pattern=on 2pt off 1.5pt] %s -- %s;" % (pl.pt(4, length(np.inf)), pl.pt(105, length(np.inf))))
    b.append("\\node[dot, fill=ndInk, minimum size=3pt] at %s {};" % pl.pt(2 * np.pi, np.sqrt(2 * np.pi)))
    for q in QS:
        b.append("\\node[circle, fill=%s, inner sep=0pt, minimum size=3pt] at %s {};" % (QCOL[q], pl.pt(q, length(q))))
    b.append("\\node[note, anchor=north west] (bf) at %s {the circle of helices\\\\ branches off at $q=2\\pi/A$};" % pl.pt(20, 2.0))
    b.append("\\draw[pointer] (bf.north west) to[bend left=15] %s;" % pl.pt(2 * np.pi + 1, np.sqrt(2 * np.pi) - 0.08))
    b.append("\\node[note, anchor=south, text=ndRed] at %s {$2\\sqrt{\\pi A}=3.545$};" % pl.pt(70, length(np.inf) + 0.06))
    b.append("\\node[note, anchor=west, text=ndGold!55!ndInk] at %s {$\\sqrt q\\,A$};" % pl.pt(13, 3.9))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the shortest $g_q$-geodesic};" % pl.pt(52, -0.75))
    N.write("ch10_heisenberg", b)


if __name__ == "__main__":
    main()
    for q in QS:
        print("q=%d: radius %.3f, length %.3f (vertical %.3f)" % (q, radius(q), length(q), np.sqrt(q) * A))
    print("limit: radius %.3f, length %.3f" % (radius(np.inf), length(np.inf)))
    print("written figures/ch10_heisenberg.tex")
