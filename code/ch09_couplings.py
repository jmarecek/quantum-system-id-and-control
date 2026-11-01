"""ch09_couplings.py -- the least times of two-qubit gates for three couplings.

Two qubits with free local gates (Khaneja, Brockett and Glaser 2001) and one of
three couplings of strength J:
    Ising        H = (J/2) ZZ               drift alone: c = -Jt (0, 0, 1)
    XY           H = (J/2) (XX + YY)        drift alone: c = -Jt (1, 1, 0)
    Heisenberg   H = (J/2) (XX + YY + ZZ)   drift alone: c = -Jt (1, 1, 1),
in the coordinates c of the Weyl chamber (ch07_weyl.py).  The classes reached
by the time t are those with a representative in t P, where P is the convex
hull of the orbit of the drift's direction under the Weyl group (permutations
and changes of sign of two coordinates): for Ising the octahedron
sum |c_i| <= 1; for XY the cuboctahedron max |c_i| <= 1, sum |c_i| <= 2; for
Heisenberg the tetrahedron with the vertices -(1,1,1), (1,1,-1), (1,-1,1),
(-1,1,1).  Since each c_i matters only modulo pi, the least time to the class
c is J t* = min over n in Z^3 of the gauge of c + pi n with respect to P.

The least times, in units of 1/J (verified below):
                CNOT   sqrt(iSWAP)  iSWAP   sqrt(SWAP) (P)  sqrt(SWAP)^dag (N)  SWAP
    Ising       pi/2   pi/2         pi      3pi/4           3pi/4               3pi/2
    XY          pi/2   pi/4         pi/2    3pi/8           3pi/8               3pi/4
    Heisenberg  pi/2   pi/2         pi      3pi/4           pi/4                pi/2
The two square roots of SWAP are named as in Chapter 7 (ch07_weyl.GATES):
sqrt(SWAP) at P = [pi/4, pi/4, pi/4], and its inverse at N = [3pi/4, pi/4, pi/4].
The usual textbook matrix, with (1 +- i)/2 in the middle block, is the one at
N; it equals exp(-i (pi/4) SWAP) up to a phase, and the Heisenberg coupling
with this sign of J makes it directly, in pi/(4J), while the root at P needs
3pi/(4J).  For the Ising and XY couplings the two roots take the same time.

(a)-(c) The chamber for each coupling, with the region J t* <= pi/2 shaded:
    two tetrahedra meeting at CNOT for Ising; for XY a larger region that
    contains iSWAP; for Heisenberg a region that reaches SWAP.
(d) The least times of six gates for the three couplings.
Checked by sampling, as in ch09_kbg.py: products of random local gates and
stretches of the drift, of total J t = pi/2, never lie beyond the computed
region, and come within 3% of its edge.
Writes figures/ch09_couplings.tex.
"""
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import needham as N
import ch07_weyl as W
import ch09_kbg as K

PI = np.pi
COUPLINGS = [("Ising", 0.5 * W.ZZ, (0, 0, -1), "ndBlue"),
             ("XY", 0.5 * (W.XX + W.YY), (-1, -1, 0), "ndGreen"),
             ("Heisenberg", 0.5 * (W.XX + W.YY + W.ZZ), (-1, -1, -1), "ndGold")]
SQSWAP = np.array([[1, 0, 0, 0], [0, (1 + 1j) / 2, (1 - 1j) / 2, 0], [0, (1 - 1j) / 2, (1 + 1j) / 2, 0], [0, 0, 0, 1]])
SQISWAP = np.array([[1, 0, 0, 0], [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0], [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 0, 1]])
GATES = [("CNOT", W.CNOT), ("$\\sqrt{\\mathrm{iSWAP}}$", SQISWAP), ("iSWAP", W.ISWAP),
         ("$\\sqrt{\\mathrm{SWAP}}$ at $P$", SQSWAP.conj().T), ("$\\sqrt{\\mathrm{SWAP}}^{\\,\\dagger}$ at $N$", SQSWAP),
         ("SWAP", W.SWAP)]
EXPECTED = {"Ising": [1 / 2, 1 / 2, 1, 3 / 4, 3 / 4, 3 / 2], "XY": [1 / 2, 1 / 4, 1 / 2, 3 / 8, 3 / 8, 3 / 4],
            "Heisenberg": [1 / 2, 1 / 2, 1, 3 / 4, 1 / 4, 1 / 2]}
CHAMBER_A = np.array([[-1, 1, 0], [0, -1, 1], [0, 0, -1], [1, 1, 0]], float)
CHAMBER_B = np.array([0, 0, 0, PI])


def orbit(a):
    pts = set()
    for perm in itertools.permutations(range(3)):
        for fl in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)):
            pts.add(tuple(np.round(np.array([a[i] for i in perm], float) * fl, 12)))
    return np.array(sorted(pts))


def facets(V):
    """Rows a with P = {x : a.x <= 1}."""
    out = []
    for eq in ConvexHull(V).equations:
        a = eq[:3] / (-eq[3])
        if not any(np.allclose(a, q) for q in out):
            out.append(a)
    return np.array(out)


def least_time(c, A, R=2):
    """J t* = min over translates of the gauge of P."""
    return min((A @ (np.asarray(c) + PI * np.array(n))).max() for n in itertools.product(range(-R, R + 1), repeat=3))


def region(A, s):
    """The pieces of {c in the chamber : J t*(c) <= s}, one convex polytope per translate."""
    pieces = []
    for n in itertools.product(range(-1, 2), repeat=3):
        G = np.vstack([CHAMBER_A, A])
        h = np.concatenate([CHAMBER_B, s - A @ (PI * np.array(n))])
        # Chebyshev centre: is the piece full-dimensional?
        nrm = np.linalg.norm(G, axis=1)
        res = linprog([0, 0, 0, -1], A_ub=np.hstack([G, nrm[:, None]]), b_ub=h, bounds=[(None, None)] * 3 + [(0, None)],
                      method="revised simplex")
        if not res.success or res.x[3] < 1e-6:
            continue
        verts = []
        for i, j, k in itertools.combinations(range(len(G)), 3):
            M = G[[i, j, k]]
            if abs(np.linalg.det(M)) < 1e-10:
                continue
            x = np.linalg.solve(M, h[[i, j, k]])
            if np.all(G @ x <= h + 1e-9) and not any(np.allclose(x, v) for v in verts):
                verts.append(x)
        pieces.append(np.array(verts))
    return pieces


def sample(rng, H, total, n=600, segs=4):
    out = []
    for _ in range(n):
        tau = rng.dirichlet(np.ones(segs)) * total
        U = W.random_local(rng)
        for t in tau:
            U = W.random_local(rng) @ expm(-1j * H * t) @ U
        out.append(K.cartan(U))
    return np.array(out)


def checks():
    assert np.allclose(K.cartan(SQSWAP), [3 * PI / 4, PI / 4, PI / 4])             # the usual matrix is at N
    assert np.allclose(K.cartan(SQSWAP.conj().T), [PI / 4, PI / 4, PI / 4])     # its inverse at P
    for name, H, a, col in COUPLINGS:
        A = facets(orbit(a))
        # the drift alone moves c along the direction a
        for t in (0.2, 0.5):
            assert abs(least_time(K.cartan(expm(-1j * H * t)), A) - t) < 1e-9
        assert np.allclose([least_time(K.cartan(U), A) / PI for _, U in GATES], EXPECTED[name])
        rng = np.random.default_rng(len(name))
        f = np.array([least_time(p, A) for p in sample(rng, H, PI / 2)])
        assert f.max() <= PI / 2 + 1e-7 and f.max() > 0.97 * PI / 2, (name, f.max())
    # the shapes: octahedron, cuboctahedron, tetrahedron
    assert [len(orbit(a)) for _, _, a, _ in COUPLINGS] == [6, 12, 4]
    assert [len(facets(orbit(a))) for _, _, a, _ in COUPLINGS] == [8, 14, 4]


def main():
    checks()
    V0 = {k: W.unit(v) for k, v in W.VERTICES.items()}
    ch = [V0["O"], V0["A1"], V0["A2"], V0["A3"]]
    b = []
    xs = [1.6, 5.5, 9.4]
    marks = {"L": "CNOT", "Q": "sqrt iSWAP", "A2": "iSWAP", "A3": "SWAP", "P": "sqrt SWAP", "N": "sqrt SWAP dag"}
    for (name, H, a, col), x0 in zip(COUPLINGS, xs):
        A = facets(orbit(a))
        V = N.View(az=-75.0, el=22.0, R=3.6, shift=(x0, 4.0))
        b += W.draw_polytope(V, ch, fill="ndPaper")
        for piece in region(A, PI / 2):
            b += W.draw_polytope(V, [W.unit(p) for p in piece], front="line width=0.4pt, %s!70!ndInk" % col,
                                 back=None, fill="%s!45, fill opacity=0.85" % col)
        b += W.draw_polytope(V, ch)
        for k, lab in marks.items():
            c = W.VERTICES[k]
            reached = least_time(c, A) <= PI / 2 + 1e-9
            b.append("\\node[dot, minimum size=3pt, fill=%s] at %s {};" % ("ndRed" if reached else "ndInk!50", V.pt(V0[k])))
        b.append("\\node[font=\\small, anchor=north] at %s {(%s) %s};" % (N.pt(V.P(V0["L"]) + np.array([0, -0.45])), "abc"[xs.index(x0)], name))
    # gate labels once, on the first chamber
    V = N.View(az=-75.0, el=22.0, R=3.6, shift=(xs[0], 4.0))
    for k, (lab, d, anc) in {"L": ("CNOT", (0.0, -0.1), "north"), "A3": ("SWAP", (0.0, 0.1), "south"),
                            "N": ("$\\sqrt{\\mathrm{SWAP}}^{\\,\\dagger}$", (0.12, -0.08), "north west"),
                            "P": ("$\\sqrt{\\mathrm{SWAP}}$", (-0.12, 0.04), "south east")}.items():
        b.append("\\node[font=\\scriptsize, anchor=%s] at %s {%s};" % (anc, N.pt(V.P(V0[k]) + np.array(d)), lab))
    b.append("\\node[font=\\scriptsize, anchor=west] (is) at %s {iSWAP};" % N.pt(V.P(V0["A2"]) + np.array([0.95, 0.95])))
    b.append("\\draw[pointer] (is.west) to[bend right=15] %s;" % N.pt(V.P(V0["A2"]) + np.array([0.05, 0.03])))
    b.append("\\node[note, anchor=south west] at %s {shaded in (a)--(c): the classes\\\\ reached by $Jt=\\frac\\pi2$; dots: the six\\\\ gates of (d), red if reached};" % N.pt(np.array([10.55, 0.75])))

    # ---- (d) the least times
    pl = N.Plot((0, 1.6 * PI), (0, 6), 7.6, 3.6, origin=(2.0, -1.4))
    b += pl.axes(xlabel="{\\scriptsize $Jt^\\star$}", xticks=[(PI / 4, "$\\frac\\pi4$"), (PI / 2, "$\\frac\\pi2$"), (3 * PI / 4, "$\\frac{3\\pi}4$"),
                                                         (PI, "$\\pi$"), (3 * PI / 2, "$\\frac{3\\pi}2$")])
    for g, (gname, U) in enumerate(GATES):
        y = 5.5 - g
        b.append("\\node[font=\\scriptsize, anchor=east] at %s {%s};" % (pl.pt(-0.05, y), gname))
        for j, (name, H, a, col) in enumerate(COUPLINGS):
            tval = least_time(K.cartan(U), facets(orbit(a)))
            yy = y + 0.27 - 0.27 * j
            b.append("\\fill[%s] %s rectangle %s;" % (col, pl.pt(0, yy - 0.1), pl.pt(tval, yy + 0.1)))
    for j, (name, H, a, col) in enumerate(COUPLINGS):
        b.append("\\fill[%s] %s rectangle %s;" % (col, N.pt(np.array([10.6, 0.3 - 0.45 * j])), N.pt(np.array([10.95, 0.45 - 0.45 * j]))))
        b.append("\\node[font=\\scriptsize, anchor=west] at %s {%s};" % (N.pt(np.array([11.05, 0.375 - 0.45 * j])), name))
    b.append("\\node[font=\\small, anchor=north] at %s {(d) the least times of six gates};" % pl.pt(0.8 * PI, -0.7))
    N.write("ch09_couplings", b)


if __name__ == "__main__":
    main()
    for name, H, a, col in COUPLINGS:
        A = facets(orbit(a))
        print(name, ["%.3f pi" % (least_time(K.cartan(U), A) / PI) for _, U in GATES])
    print("written figures/ch09_couplings.tex")
