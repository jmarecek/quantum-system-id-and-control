"""ch09_brachistochrone.py -- Johann Bernoulli's problem of 1696.

A bead slides without friction, from rest, along a curve from A = (0, 0) to
B = (pi, -2), with g = 1; its speed at the height y is sqrt(2 g |y|), so that
the time of descent is T = int ds / sqrt(2 g |y|).  Four curves:
    the cycloid x = theta - sin theta, y = -(1 - cos theta), theta in [0, pi],
        traced by a point of a circle of radius 1 rolling beneath the line
        y = 0, for which dt = d theta and T = pi = 3.142 exactly;
    the arc of the circle through A and B with a vertical tangent at A
        (centre (c, 0), c = (pi^2 + 4)/(2 pi) = 2.207), T = 3.221;
    the broken line through (0, -2), a fall followed by a horizontal run,
        T = 2 + pi/2 = 3.571;
    the straight line, T = |AB|/sqrt(g) = sqrt(pi^2 + 4) = 3.724.
The cycloid is the quickest although it is longer than the line (4 against
3.724); it reaches B, the lowest point of its arch, horizontally.  The
circular arc, which dips below B and which Galileo had thought the quickest
in 1638, comes second.  Beads released together are drawn every
half unit of time, and their positions at t = pi, when the bead on the
cycloid arrives at B, are marked.
Writes figures/ch09_brachistochrone.tex.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import needham as N

G = 1.0
B = np.array([np.pi, -2.0])
C = (np.pi ** 2 + 4) / (2 * np.pi)                     # centre (C, 0) and radius C of the circular arc
PHI_B = np.arctan2(B[1], B[0] - C) + 2 * np.pi         # the arc runs from phi = pi (at A) to PHI_B
L_LINE = np.hypot(*B)


def cycloid(theta):
    return np.stack([theta - np.sin(theta), -(1 - np.cos(theta))], -1)


def circle(phi):
    return np.stack([C + C * np.cos(phi), C * np.sin(phi)], -1)


def circle_time(phi):
    """Time to reach the angle phi on the arc, from rest at A (phi = pi)."""
    return quad(lambda p: C / np.sqrt(2 * G * abs(C * np.sin(p))), np.pi, phi, limit=200)[0]


T_CYC = np.pi
T_CIRC = circle_time(PHI_B)
T_BROKEN = np.sqrt(2 * 2 / G) + np.pi / np.sqrt(2 * G * 2)
T_LINE = L_LINE / np.sqrt(G)


def position(curve, t):
    """Where the bead on each curve is at the time t (it stops at B)."""
    if curve == "cycloid":
        return cycloid(min(t, np.pi))
    if curve == "line":
        s = min(0.5 * (G * 2 / L_LINE) * t ** 2, L_LINE)
        return s * B / L_LINE
    if curve == "broken":
        if t <= 2:
            return np.array([0.0, -0.5 * G * t ** 2])
        return np.array([min(2 * (t - 2), np.pi), -2.0])
    if curve == "circle":
        if t >= T_CIRC:
            return B
        if t <= 0:
            return np.zeros(2)
        return circle(brentq(lambda p: circle_time(p) - t, np.pi, PHI_B))
    raise ValueError(curve)


def descent_time(points):
    """T = int ds / sqrt(2 g |y|) along a polygon, by the midpoint rule on each segment (start excluded)."""
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mid = 0.5 * (points[1:] + points[:-1])
    keep = d > 0
    return np.sum(d[keep] / np.sqrt(2 * G * np.abs(mid[keep, 1])))


def checks():
    assert np.allclose(cycloid(np.pi), B) and np.allclose(circle(np.pi), [0, 0]) and np.allclose(circle(PHI_B), B)
    # the times, by quadrature along fine polygons (refined towards A, where the speed vanishes)
    s = np.linspace(0, 1, 400001) ** 2
    assert abs(descent_time(cycloid(np.pi * s)) - np.pi) < 2e-3
    assert abs(descent_time(np.outer(s, B)) - T_LINE) < 2e-3
    assert abs(descent_time(circle(np.pi + (PHI_B - np.pi) * s)) - T_CIRC) < 2e-3
    assert abs(T_CIRC - 3.2213) < 1e-3 and abs(T_LINE - 3.7242) < 1e-3 and abs(T_BROKEN - 3.5708) < 1e-3
    assert T_CYC < T_CIRC < T_BROKEN < T_LINE
    # the cycloid is longer than the line, and B is the lowest point of its arch
    th = np.linspace(0, np.pi, 100001)
    assert abs(np.sum(np.linalg.norm(np.diff(cycloid(th), axis=0), axis=1)) - 4.0) < 1e-6 and 4.0 > L_LINE
    assert np.argmin(cycloid(th)[:, 1]) == len(th) - 1
    # dt = d theta on the cycloid: the bead at the time t is at theta = t
    for t in (0.5, 1.7, 3.0):
        assert abs(descent_time(cycloid(t * np.linspace(0, 1, 200001) ** 2)) - t) < 2e-3
    # the circle passes below B and rises to it
    phis = np.linspace(np.pi, PHI_B, 10001)
    assert circle(phis)[:, 1].min() < B[1]


def main():
    checks()
    b = []
    S = 2.45
    o = np.array([0.6, 5.9])
    P = lambda v: o + S * np.array([v[0], v[1]])
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$x$};" % (N.pt(P([-0.15, 0])), N.pt(P([3.55, 0]))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[below, font=\\scriptsize, text=ndInk] {$y$};" % (N.pt(P([0, 0.15])), N.pt(P([0, -2.35]))))
    # the generating circle, rolling beneath y = 0
    th0 = 2.1
    cc = np.array([th0, -1.0])
    b.append("\\draw[faint, dash pattern=on 2pt off 1.5pt] %s circle (%.3f);" % (N.pt(P(cc)), S * 1.0))
    q = cycloid(th0)
    b.append("\\draw[faint] %s -- %s;" % (N.pt(P(cc)), N.pt(P(q))))
    b.append("\\fill[ndInk!45] %s circle (1.2pt);" % N.pt(P(cc)))
    # the curves
    sty = {"cycloid": "line width=1.6pt, ndRed", "circle": "line width=1.1pt, ndBlue",
           "broken": "line width=1.0pt, ndGreen", "line": "line width=1.0pt, ndGold!85!ndInk"}
    col = {"cycloid": "ndRed", "circle": "ndBlue", "broken": "ndGreen", "line": "ndGold!85!ndInk"}
    b.append(N.polyline([P(p) for p in np.outer(np.linspace(0, 1, 2), B)], sty["line"]))
    b.append(N.polyline([P([0, 0]), P([0, -2]), P(B)], sty["broken"]))
    b.append(N.polyline([P(p) for p in circle(np.linspace(np.pi, PHI_B, 241))], sty["circle"]))
    b.append(N.polyline([P(p) for p in cycloid(np.linspace(0, np.pi, 241))], sty["cycloid"]))
    # beads every half unit of time, and all four at t = pi
    for name in ("line", "broken", "circle", "cycloid"):
        for t in np.arange(0.5, 3.01, 0.5):
            b.append("\\fill[%s] %s circle (1.3pt);" % (col[name], N.pt(P(position(name, t)))))
        b.append("\\filldraw[fill=white, draw=%s, line width=1.0pt] %s circle (2.6pt);" % (col[name], N.pt(P(position(name, np.pi)))))
    b.append("\\node[dot] at %s {};" % N.pt(P([0, 0])))
    b.append("\\node[dot] at %s {};" % N.pt(P(B)))
    b.append("\\node[font=\\small, anchor=south east] at %s {$A$};" % N.pt(P([0, 0]) + [-0.05, 0.05]))
    b.append("\\node[font=\\small, anchor=west] at %s {$B$};" % N.pt(P(B) + [0.18, 0.05]))
    # the times of descent, as a key beside B, in the order of arrival
    kx, ky = P(B)[0] + 0.75, P([0, -0.45])[1]
    b.append("\\node[font=\\small, anchor=west] at (%.3f,%.3f) {time of descent};" % (kx - 0.05, ky + 0.45))
    for k, (name, txt) in enumerate([("cycloid", "cycloid, $T=\\pi=3.14$"), ("circle", "circular arc, $3.22$"),
                                     ("broken", "fall, then run, $3.57$"), ("line", "straight line, $3.72$")]):
        y = ky - 0.48 * k
        b.append("\\draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);" % (sty[name], kx, y, kx + 0.55, y))
        b.append("\\node[font=\\small, anchor=west, text=%s] at (%.3f,%.3f) {%s};" % (col[name], kx + 0.65, y, txt))
    b.append("\\node[note, anchor=north west] at (%.3f,%.3f) {The cycloid is the quickest,\\\\ though longer than the line\\\\ ($4$ against $3.72$).\\\\ Galileo had thought the\\\\ circular arc quickest (1638).};" % (kx - 0.05, ky - 2.05))
    b.append("\\node[note, anchor=north] at %s {small dots: every $\\tfrac12$ unit of time; rings: all four beads at $t=\\pi$,\\\\ when the bead on the cycloid reaches $B$};" % N.pt(P([1.6, -2.42])))
    b.append("\\node[note, anchor=south west, text=ndInk!70] (rc) at %s {the cycloid is traced by a point\\\\ of a circle rolling beneath $AB$};" % N.pt(P([2.15, 0.12])))
    b.append("\\draw[pointer] (rc.south west) to[bend right=20] %s;" % N.pt(P(cc + np.array([-0.55, 0.62]))))
    N.write("ch09_brachistochrone", b)


if __name__ == "__main__":
    main()
    print("T: cycloid %.4f, circle %.4f, broken %.4f, line %.4f" % (T_CYC, T_CIRC, T_BROKEN, T_LINE))
    print("written figures/ch09_brachistochrone.tex")
