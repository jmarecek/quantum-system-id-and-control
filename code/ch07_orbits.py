"""ch07_orbits.py -- unitary orbits and the spectra that label them.

(a) A qubit.  Conjugation rho -> U rho U^dagger rotates the Bloch vector, so
    the orbits of the unitary group are the spheres |r| = const: the pure
    states on the unit sphere, and inside it spheres of mixed states of equal
    purity.  Unitary control moves a state along its sphere; only noise or
    measurement moves it across the spheres.
(b) A qutrit.  The orbit of rho is fixed by its spectrum lambda, and the
    diagonal states form the probability simplex, drawn in the coherence
    coordinates (x3, x8) of Chapter 2.  By the Schur--Horn theorem the
    diagonals of the states U rho U^dagger (the populations that unitary
    control can produce from rho) fill the convex hull of the six permutations
    of lambda = (0.6, 0.3, 0.1), a hexagon.  The point mu = (0.8, 0.1, 0.1) is
    outside it: it is not majorised by lambda, being purer.  The ordered
    spectra lambda1 >= lambda2 >= lambda3 form one sixth of the simplex.
The script checks Schur--Horn numerically with Haar-random unitaries.
Writes figures/ch07_orbits.tex.
"""
import itertools
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import needham as N

LAM = np.array([0.6, 0.3, 0.1])
MU = np.array([0.8, 0.1, 0.1])
S2, S6 = np.sqrt(2.0), np.sqrt(6.0)


def xy(p):
    """Coherence coordinates (x3, x8) of the diagonal state with populations p."""
    p = np.asarray(p, dtype=float)
    return np.array([(p[0] - p[1]) / S2, (p[0] + p[1] - 2 * p[2]) / S6])


def majorised(a, b):
    """True if a is majorised by b (both probability vectors)."""
    a, b = np.sort(a)[::-1], np.sort(b)[::-1]
    return np.all(np.cumsum(a)[:-1] <= np.cumsum(b)[:-1] + 1e-12)


def haar(rng, n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def check():
    rng = np.random.default_rng(3)
    perms = np.array([LAM[list(p)] for p in itertools.permutations(range(3))])
    hexagon = Delaunay(np.array([xy(p) for p in perms]))
    D = np.array([np.real(np.diag(U @ np.diag(LAM) @ U.conj().T)) for U in (haar(rng, 3) for _ in range(40000))])
    assert np.all([majorised(d, LAM) for d in D[:2000]])                  # Schur: the diagonal is majorised
    assert np.all(hexagon.find_simplex(np.array([xy(d) for d in D]), tol=1e-9) >= 0)   # inside the hexagon
    for p in itertools.permutations(range(3)):                             # permutation matrices give the vertices
        P = np.eye(3)[list(p)]
        assert np.allclose(np.diag(P @ np.diag(LAM) @ P.T), LAM[list(p)])
    top = D[:, 0].max()
    assert 0.58 < top <= 0.6 + 1e-12                                      # the largest population is lambda_max
    hull = ConvexHull(np.array([xy(d) for d in D]))
    hexa = ConvexHull(np.array([xy(p) for p in perms]))
    assert hull.volume > 0.95 * hexa.volume                                 # the samples fill the hexagon (Horn)
    assert not majorised(MU, LAM) and hexagon.find_simplex(xy(MU)) < 0
    print("Schur-Horn: 40000 Haar samples inside the hexagon; largest population %.4f (lambda_max = 0.6); "
          "sampled hull %.1f%% of the hexagon; mu = (0.8, 0.1, 0.1) not majorised by lambda" % (top, 100 * hull.volume / hexa.volume))
    return perms


def qubit_panel(b):
    az, el, R0 = 30.0, 18.0, 2.0
    radii = (1.0, 0.6, 0.3)
    for k, rad in enumerate(radii):
        V = N.View(az=az, el=el, R=R0 * rad)
        if k == 0:
            b += V.sphere()
        else:
            b.append("\\fill[ndSky!%d!ndBlue, fill opacity=0.35] %s circle (%.3f);" % (80 - 15 * k, V.pt([0, 0, 0]), V.R))
            b.append("\\draw[ink, line width=0.6pt] %s circle (%.3f);" % (V.pt([0, 0, 0]), V.R))
        b += V.great_circle([0, 0, 1], front="faint", back="faint, densely dotted")
    V = N.View(az=az, el=el, R=R0)
    # a state on the silhouette of the middle sphere, and a rotation along it (about the line of sight)
    rad = radii[1]
    beta = np.radians(35.0)
    r0 = rad * (np.cos(beta) * V.e1 + np.sin(beta) * V.e2)
    axis = V.d

    def rot(v, a):
        return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * (axis @ v) * (1 - np.cos(a))
    arc = np.array([rot(r0, a) for a in np.linspace(0.06, 1.25, 60)])
    assert np.allclose(np.linalg.norm(arc, axis=1), rad)                    # a rotation keeps |r|
    b.append(N.polyline([V.P(p) for p in arc], "->, >=stealth, line width=1.3pt, ndBlue"))
    inward = 0.36 * r0
    b.append("\\draw[->, >=stealth, line width=1.2pt, ndRed] %s -- %s;" % (V.pt(r0), V.pt(inward)))
    b.append("\\node[dot] at %s {};" % V.pt(r0))
    b.append("\\node[dot, minimum size=2.8pt] at %s {};" % V.pt([0, 0, 0]))
    b.append("\\node[font=\\small, anchor=north] at %s {$\\tfrac12\\Id$};" % N.pt(V.P([0, 0, 0]) + np.array([-0.05, -0.04])))
    b.append("\\node[font=\\small, anchor=south] at %s {$\\ket0$};" % V.pt([0, 0, 1.0]))
    b.append("\\node[dot, minimum size=2.6pt] at %s {};" % V.pt([0, 0, 1.0]))
    c = V.P([0, 0, 0])
    b.append("\\node[note, anchor=west, text=ndBlue] (u) at %s {unitary control:\\\\ along the sphere};" % N.pt(c + np.array([R0 + 0.2, 0.95])))
    b.append("\\draw[pointer] (u.west) to[bend right=15] %s;" % N.pt(V.P(arc[30])))
    b.append("\\node[note, anchor=west, text=ndRed] (n) at %s {noise or\\\\ measurement:\\\\ across the spheres};"
             % N.pt(c + np.array([R0 + 0.2, -0.75])))
    b.append("\\draw[pointer] (n.west) to[bend left=15] %s;" % N.pt(0.5 * (V.P(r0) + V.P(inward)) + np.array([0.03, -0.03])))
    b.append("\\node[note, anchor=east] (p) at %s {$\\abs{\\Bloch}=1$:\\\\ pure states};" % N.pt(c + np.array([-R0 - 0.15, 1.35])))
    b.append("\\draw[pointer] (p.east) to[bend left=10] %s;" % N.pt(c + R0 * np.array([np.cos(2.35), np.sin(2.35)])))
    b.append("\\node[note, anchor=east] (m) at %s {$\\abs{\\Bloch}<1$:\\\\ mixed states\\\\ of equal purity};" % N.pt(c + np.array([-R0 - 0.15, -1.0])))
    b.append("\\draw[pointer] (m.east) to[bend right=10] %s;" % N.pt(c + 0.6 * R0 * np.array([np.cos(3.75), np.sin(3.75)])))
    b.append("\\node[font=\\small, anchor=north, align=center] at %s {(a) a qubit: the orbits are\\\\ the spheres $\\abs{\\Bloch}=\\mathrm{const}$};"
             % N.pt(c + np.array([0.0, -R0 - 0.35])))


def qutrit_panel(b, perms, ox):
    s = 2.9                                                                 # cm per unit of x
    P = lambda p: np.array([ox, 0.55]) + s * xy(p)
    E = np.eye(3)
    corners = [P(e) for e in E]
    b.append("\\fill[ndPaper] %s -- %s -- %s -- cycle;" % tuple(N.pt(c) for c in corners))
    # the Weyl chamber lambda1 >= lambda2 >= lambda3: corner |0>, midpoint of |0>|1>, centre
    ch = [P(E[0]), P([0.5, 0.5, 0]), P([1 / 3, 1 / 3, 1 / 3])]
    b.append("\\fill[ndGold, fill opacity=0.2] %s -- %s -- %s -- cycle;" % tuple(N.pt(c) for c in ch))
    b.append("\\draw[ndGold!80!ndInk, line width=0.5pt, densely dashed] %s -- %s -- %s;" % tuple(N.pt(c) for c in ch))
    # the permutohedron
    pts = np.array([P(p) for p in perms])
    hull = ConvexHull(pts)
    poly = [pts[i] for i in hull.vertices]
    b.append("\\fill[ndSky, fill opacity=0.85] %s -- cycle;" % " -- ".join(N.pt(q) for q in poly))
    b.append("\\draw[ndBlue, line width=0.9pt] %s -- cycle;" % " -- ".join(N.pt(q) for q in poly))
    b.append("\\draw[ink] %s -- %s -- %s -- cycle;" % tuple(N.pt(c) for c in corners))
    for p in perms:
        b.append("\\node[dot, fill=ndBlue, minimum size=3pt] at %s {};" % N.pt(P(p)))
    b.append("\\node[dot, fill=ndBlue, minimum size=4.4pt] at %s {};" % N.pt(P(LAM)))
    b.append("\\node[font=\\small, anchor=north east, text=ndBlue] at %s {$\\lambda(\\rho)$};" % N.pt(P(LAM) + np.array([-0.04, -0.04])))
    c0 = P([1 / 3, 1 / 3, 1 / 3])
    b.append("\\node[dot, minimum size=2.8pt] at %s {};" % N.pt(c0))
    b.append("\\node[font=\\small, anchor=north] at %s {$\\tfrac13\\Id$};" % N.pt(c0 + np.array([0.0, -0.05])))
    for e, lab, anc in [(E[0], "\\ket0\\bra0", "west"), (E[1], "\\ket1\\bra1", "east"), (E[2], "\\ket2\\bra2", "north")]:
        b.append("\\node[dot, minimum size=3pt] at %s {};" % N.pt(P(e)))
        off = {"west": (0.08, 0), "east": (-0.08, 0), "north": (0, -0.08)}[anc]
        b.append("\\node[font=\\small, anchor=%s] at %s {$%s$};" % (anc, N.pt(P(e) + np.array(off)), lab))
    # mu, outside the hexagon
    b.append("\\node[dot, fill=ndRed, minimum size=4pt] at %s {};" % N.pt(P(MU)))
    b.append("\\node[note, anchor=west, text=ndRed] (m) at %s {$\\mu=(0.8,0.1,0.1)$:\\\\ not reachable,\\\\ purer than $\\rho$};"
             % N.pt(P(E[0]) + np.array([-0.35, -1.05])))
    b.append("\\draw[pointer] (m.north west) to[bend left=12] %s;" % N.pt(P(MU) + np.array([0.02, -0.06])))
    # "more mixed": towards the centre
    a0 = P([0.47, 0.03, 0.5])
    b.append("\\draw[->, >=stealth, line width=0.9pt, ndInk!70] %s -- %s;" % (N.pt(a0), N.pt(c0 + 0.32 * (a0 - c0))))
    b.append("\\node[note, anchor=north west] at %s {more mixed};" % N.pt(a0 + np.array([-0.05, -0.02])))
    # notes on the hexagon and the chamber
    b.append("\\node[note, anchor=east, text=ndBlue] (h) at %s {populations reachable\\\\ by unitary control:\\\\ the hexagon (Schur--Horn)};"
             % N.pt(P(E[1]) + np.array([0.75, -1.75])))
    b.append("\\draw[pointer] (h.east) to[bend right=12] %s;" % N.pt(P([0.2, 0.5, 0.3])))
    b.append("\\node[note, anchor=south, text=ndGold!70!ndInk] (w) at %s {ordered spectra\\\\ $\\lambda_1\\ge\\lambda_2\\ge\\lambda_3$};"
             % N.pt(P([0.65, 0.35, 0]) + np.array([0.1, 0.35])))
    b.append("\\draw[pointer] (w.south) to[bend left=10] %s;" % N.pt(P([0.64, 0.34, 0.02])))
    b.append("\\node[font=\\small, anchor=north, align=center] at %s {(b) a qutrit: the populations of\\\\ $U\\rho U^\\dagger$, for $\\lambda(\\rho)=(0.6,0.3,0.1)$};"
             % N.pt(np.array([ox, P(E[2])[1] - 0.45])))


def main():
    perms = check()
    b = []
    qubit_panel(b)
    qutrit_panel(b, perms, 10.0)
    N.write("ch07_orbits", b)
    print("written figures/ch07_orbits.tex")


if __name__ == "__main__":
    main()
