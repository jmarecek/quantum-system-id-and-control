"""ch12_shooting.py -- the landscape of the shooting function for the time-optimal qubit.

The example of the chapter: the driftless qubit H = (u_x sigma_x + u_y sigma_y)/2,
u_x^2 + u_y^2 <= 1, Bloch vector q from (1,0,0) to (0,1,0) in least time.  The
state and the costate p obey the same equations, qdot = Omega x q and
pdot = Omega x p with Omega = (u_x, u_y, 0); the maximum condition gives
(u_x, u_y) = (M_x, M_y)/h with M = q x p and h = |(M_x, M_y)|.  Along an
extremal M_z = p_y(0) is conserved and (M_x, M_y) turns at the rate
c = p_y(0)/|p_z(0)|, so in a frame turning with the control the Bloch vector
rotates about the fixed axis (0, -sign p_z(0), c); this closed form is checked
against a direct integration of the state--costate system.  p_x(0) plays no
part, and H_P = 0 with p_0 = -1 fixes |p_z(0)| = 1.

  (a) the miss |q(T*) - q_target| at T* = pi sqrt(3)/2 over the unit sphere of
      initial costates p(0), in bands; it is constant along the meridians through
      +-e_x and vanishes on the two meridians through (., 1/sqrt3, +-1);
  (b) the miss over the chart (p_y(0), T) with p_z(0) = 1, i.e. the shooting
      function of the free-time problem: its zeros are (1/sqrt3, pi sqrt3/2), the
      optimum, and (3/sqrt7, pi sqrt7/2), (5/sqrt11, pi sqrt11/2), extremals that
      reach the target later; two damped Gauss--Newton runs are drawn, one
      converging to the optimum, one to a later extremal.
Writes figures/ch12_shooting.tex.
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

Q0, TARGET = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
T_STAR = np.pi * np.sqrt(3) / 2
LEVELS = [0.1, 0.25, 0.5, 1.0, 1.5]
BANDS = ["ndGold!90", "ndBlue!50", "ndBlue!30", "ndSky!90", "ndSky!35", "ndPaper"]


def flow(p, t):
    """Bloch vector at time t on the extremal with initial costate p (p_z != 0)."""
    c, s = p[1] / abs(p[2]), np.sign(p[2])
    q = Rotation.from_rotvec(np.array([0.0, -s, c]) * t).apply(Q0)
    return Rotation.from_rotvec([0.0, 0.0, -c * t]).apply(q)


def pmp_rhs(t, y):
    """State and costate of the maximum principle, the control eliminated."""
    q, p = y[:3], y[3:]
    M = np.cross(q, p)
    om = np.array([M[0], M[1], 0.0]) / np.hypot(M[0], M[1])
    return np.concatenate([np.cross(om, q), np.cross(om, p)])


def miss(py, T):
    return np.linalg.norm(flow(np.array([0.0, py, 1.0]), T) - TARGET)


def newton(z, steps=40):
    """Damped Gauss--Newton on (p_y(0), T) for q(T) = q_target; returns the path."""
    res = lambda z: flow(np.array([0.0, z[0], 1.0]), z[1]) - TARGET
    path = [np.array(z, float)]
    for _ in range(steps):
        z, r = path[-1], res(path[-1])
        if np.linalg.norm(r) < 1e-15:
            break
        J = np.column_stack([(res(z + e) - res(z - e)) / 2e-6 for e in np.eye(2) * 1e-6])
        d, lam = -np.linalg.lstsq(J, r, rcond=None)[0], 1.0
        while np.linalg.norm(res(z + lam * d)) >= np.linalg.norm(r) and lam > 1e-4:
            lam /= 2
        path.append(z + lam * d)
    return np.array(path)


def band_rows(xs, ys, F, P):
    """Run-length rectangles of the colour bands of F (rows follow ys); P maps (x, y) to the page."""
    out, dx, dy = [], xs[1] - xs[0], ys[1] - ys[0]
    idx = np.searchsorted(LEVELS, np.nan_to_num(F, nan=9.0))
    for i, y in enumerate(ys):
        j = 0
        while j < len(xs):
            k = j
            while k + 1 < len(xs) and idx[i, k + 1] == idx[i, j]:
                k += 1
            if idx[i, j] < len(BANDS) - 1:                       # the palest band is the paper underneath
                a, b = P(max(xs[j] - dx / 2, xs[0]), max(y - dy / 2, ys[0])), P(min(xs[k] + dx / 2, xs[-1]), min(y + dy / 2, ys[-1]))
                out.append("\\fill[%s] %s rectangle %s;" % (BANDS[idx[i, j]], N.pt(a), N.pt(b)))
            j = k + 1
    return out


def main():
    # the closed form against a direct integration of the state--costate system
    for p in ([0.3, 1 / np.sqrt(3), 1.0], [-0.2, 0.9, -1.0], [0.0, -0.4, 0.7]):
        sol = solve_ivp(pmp_rhs, (0, T_STAR), np.concatenate([Q0, p]), method="DOP853", rtol=1e-12, atol=1e-13)
        assert np.abs(sol.y[:3, -1] - flow(np.array(p), T_STAR)).max() < 1e-10
    # the optimal costates of the closed form: p(0) = (anything, 1/sqrt3, +-1)
    for px in (-0.7, 0.0, 2.0):
        for s in (1, -1):
            assert np.linalg.norm(flow(np.array([px, 1 / np.sqrt(3), s]), T_STAR) - TARGET) < 1e-14
    b = []

    # ---- (a) the sphere of initial costates, miss at T*
    V = N.View(az=32, el=22, R=2.0, shift=(2.3, 2.55))
    dmiss = lambda phi: np.linalg.norm(flow(np.array([0.0, np.sin(phi), np.cos(phi)]), T_STAR) - TARGET)
    n = 150
    us = np.linspace(-1, 1, n)
    F = np.full((n, n), np.nan)
    for i, v in enumerate(us):
        for j, u in enumerate(us):
            w = 1 - u * u - v * v
            if w > 0:
                x = u * V.e1 + v * V.e2 + np.sqrt(w) * V.d
                if abs(x[2]) > 1e-3:
                    F[i, j] = dmiss(np.arctan2(x[1], x[2]))
    b.append("\\begin{scope}\\clip %s circle (%.3f);" % (V.pt([0, 0, 0]), V.R))
    b.append("\\fill[ndPaper] %s circle (%.3f);" % (V.pt([0, 0, 0]), V.R))
    b += band_rows(us, us, F, lambda u, v: V.shift + V.R * np.array([u, v]))
    b.append("\\end{scope}")
    # the meridians at the band edges, found on a fine grid of the angle about e_x
    phis = np.linspace(-np.pi, np.pi, 7201)
    dm = np.array([dmiss(f) if abs(np.cos(f)) > 1e-6 else np.nan for f in phis])
    merid = lambda f: np.array([[np.cos(t), np.sin(t) * np.sin(f), np.sin(t) * np.cos(f)] for t in np.linspace(0, np.pi, 121)])
    for lev in LEVELS:
        for k in np.where(np.diff(np.sign(dm - lev)) != 0)[0]:
            if np.isfinite(dm[k]) and np.isfinite(dm[k + 1]):
                b += V.curve(merid(phis[k]), "line width=0.35pt, ndInk!45")
    dz = np.where(np.isfinite(dm), dm, 9.0)
    zeros = phis[1:-1][(dz[1:-1] < dz[:-2]) & (dz[1:-1] < dz[2:]) & (dz[1:-1] < 1e-2)]
    assert np.allclose(sorted(np.degrees(zeros)), [30.0, 150.0], atol=0.06)         # p_y : p_z = 1/sqrt3 : +-1
    for f in (np.radians(30), np.radians(150)):
        b += V.curve(merid(f), "line width=1.3pt, ndRed", "line width=0.6pt, ndRed, densely dotted")
    b.append("\\draw[ink] %s circle (%.3f);" % (V.pt([0, 0, 0]), V.R))
    for e, lab, anc in (([1, 0, 0], "p_x", "south east"), ([0, 1, 0], "p_y", "west"), ([0, 0, 1], "p_z", "south")):
        e = np.array(e, float)
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[anchor=%s, font=\\scriptsize, text=ndInk] {$%s$};"
                 % (V.pt(e), V.pt(1.32 * e), anc, lab))
    b.append("\\node[dot, minimum size=2.6pt] at %s {};" % V.pt([1, 0, 0]))
    top = V.P([0, 0, 0]) + np.array([0, V.R])
    b.append("\\node[note, anchor=south east, text=ndRed, align=right] (r) at (%.3f,%.3f) {red: the optimal costates\\\\ $p(0)=(\\,\\cdot\\,,1/\\sqrt3,\\pm1)$};"
             % (top[0] - 0.35, top[1] + 0.05))
    pr = V.P(np.array([np.cos(1.05), np.sin(1.05) * 0.5, np.sin(1.05) * np.cos(np.radians(30))]))
    b.append("\\draw[pointer] (r.south) to[bend right=20] %s;" % N.pt(pr))
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {$p_x(0)$ plays no part: the miss is\\\\ constant along each meridian through $\\pm e_x$};"
             % (V.P([0, 0, 0])[0], V.P([0, 0, 0])[1] - V.R - 0.12))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) the miss at $T^\\star$ over the sphere of $p(0)$};"
             % (V.P([0, 0, 0])[0], -0.95))

    # ---- (b) the free-time shooting function on (p_y(0), T)
    pys, Ts = np.linspace(-0.6, 2.0, 209), np.linspace(0.0, 6.0, 241)
    D = np.array([[miss(py, T) for py in pys] for T in Ts])
    Pl = N.Plot((-0.6, 2.0), (0.0, 6.0), 4.6, 4.6, origin=(6.3, 0.35))
    b.append("\\fill[ndPaper] %s rectangle %s;" % (Pl.pt(-0.6, 0), Pl.pt(2.0, 6.0)))
    b += band_rows(pys, Ts, D, Pl.P)
    from skimage import measure
    for lev in LEVELS:
        for path in measure.find_contours(D, lev):
            pts = [Pl.P(np.interp(cj, np.arange(len(pys)), pys), np.interp(ci, np.arange(len(Ts)), Ts)) for ci, cj in path]
            b.append(N.polyline(pts, "line width=0.35pt, ndInk!45"))
    b.append("\\draw[ink, line width=0.5pt] %s rectangle %s;" % (Pl.pt(-0.6, 0), Pl.pt(2.0, 6.0)))
    for v in (0, 1, 2):
        b.append("\\draw[faint] %s -- ++(0,-0.08) node[below, font=\\scriptsize, text=ndInk] {$%d$};" % (Pl.pt(v, 0), v))
    for v in (0, 2, 4, 6):
        b.append("\\draw[faint] %s -- ++(-0.08,0) node[left, font=\\scriptsize, text=ndInk] {$%d$};" % (Pl.pt(-0.6, v), v))
    b.append("\\node[font=\\small] at %s {$p_y(0)$};" % N.pt(Pl.P(0.7, 0) + np.array([0, -0.55])))
    b.append("\\node[font=\\small] at %s {$T$};" % N.pt(Pl.P(-0.6, 3.0) + np.array([-0.5, 0])))
    # the zeros: c_k = (2k-1)/sqrt(4k-1), T_k = pi sqrt(4k-1)/2; Newton from two guesses
    exact = [((2 * k - 1) / np.sqrt(4 * k - 1), np.pi * np.sqrt(4 * k - 1) / 2) for k in (1, 2, 3)]
    for z in exact:
        assert miss(*z) < 1e-14
    near = [(pys[j], Ts[i]) for i in range(1, len(Ts) - 1) for j in range(1, len(pys) - 1)
            if D[i, j] == D[i - 1:i + 2, j - 1:j + 2].min() and D[i, j] < 0.1]
    assert len(near) == 3 and all(min(np.hypot(a - c, t - T) for c, T in exact) < 0.05 for a, t in near)
    runs = [(newton([0.15, 1.6]), exact[0]), (newton([1.3, 4.0]), exact[1])]
    for path, z in runs:
        assert np.abs(path[-1] - np.array(z)).max() < 1e-13 and miss(*path[-1]) < 1e-14
    assert abs(T_STAR - min(t for _, t in exact)) < 1e-15                      # the earliest zero is the optimum
    for path, _ in runs:
        b.append(N.polyline([Pl.P(*z) for z in path], "line width=0.7pt, ndInk"))
        b += ["\\fill[ndInk] %s circle (1.0pt);" % Pl.pt(*z) for z in path[:-1]]
        b.append("\\draw[ndInk, fill=white, line width=0.5pt] %s circle (1.5pt);" % Pl.pt(*path[0]))
    a = np.pi / 2 + np.arange(10) * np.pi / 5
    rr = np.where(np.arange(10) % 2 == 0, 0.17, 0.07)
    c0 = Pl.P(*exact[0])
    b.append("\\fill[ndGold, draw=ndInk, line width=0.3pt] %s -- cycle;"
             % " -- ".join(N.pt(c0 + q * np.array([np.cos(t), np.sin(t)])) for q, t in zip(rr, a)))
    for z in exact[1:]:
        b.append("\\draw[ndRed, fill=white, line width=0.8pt] %s circle (2.0pt);" % Pl.pt(*z))
    x0 = Pl.P(2.0, 0)[0] + 0.25
    b.append("\\node[note, anchor=west, align=left] (s) at (%.3f,%.3f) {star: the optimum,\\\\ $p_y(0)=1/\\sqrt3$,\\\\ $T^\\star=\\pi\\sqrt3/2$};"
             % (x0, Pl.P(0, 2.2)[1]))
    b.append("\\draw[pointer] (s.west) to[bend left=12] %s;" % N.pt(c0 + np.array([0.2, 0.0])))
    b.append("\\node[note, anchor=west, align=left, text=ndRed] (l) at (%.3f,%.3f) {open: later extremals,\\\\ $(3/\\sqrt7,\\pi\\sqrt7/2)$,\\\\ $(5/\\sqrt{11},\\pi\\sqrt{11}/2)$};"
             % (x0, Pl.P(0, 4.9)[1]))
    b.append("\\draw[pointer] (l.west) to[bend right=12] %s;" % N.pt(Pl.P(*exact[2]) + np.array([0.08, 0.04])))
    b.append("\\node[note, anchor=north west, align=left] at (%.3f,%.3f) {black: Newton from\\\\ two guesses, %d and %d\\\\ steps to $10^{-15}$};"
             % (x0, Pl.P(0, 1.0)[1] + 0.25, len(runs[0][0]) - 1, len(runs[1][0]) - 1))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) the shooting function, $p_z(0)=1$};" % (Pl.P(0.7, 0)[0] + 0.9, -0.95))
    # the key of the bands, under both panels
    edges = ["0"] + ["%g" % l for l in LEVELS] + ["2"]
    xk, yk = 0.6, -1.75
    b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {miss $\\abs{q(T)-q_{\\mathrm{target}}}$:};" % (xk + 1.65, yk + 0.11))
    for k, col in enumerate(BANDS):
        x = xk + 1.8 + 1.75 * k
        b.append("\\fill[%s, draw=ndInk!50, line width=0.3pt] (%.3f,%.3f) rectangle ++(0.36,0.22);" % (col, x, yk))
        b.append("\\node[font=\\scriptsize, anchor=west] at (%.3f,%.3f) {$%s$--$%s$};" % (x + 0.4, yk + 0.11, edges[k], edges[k + 1]))
    N.write("ch12_shooting", b)
    print("written figures/ch12_shooting.tex; Newton steps %d and %d; zeros (p_y, T):"
          % (len(runs[0][0]) - 1, len(runs[1][0]) - 1), [tuple(np.round(z, 4)) for z in exact])


if __name__ == "__main__":
    main()
