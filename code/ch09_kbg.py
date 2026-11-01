"""ch09_kbg.py -- time-optimal two-qubit gates with fast local control.

Two qubits with the Ising coupling H_d = (J/2) sigma_z (x) sigma_z of Chapter 2
and local controls so strong that local gates cost no time (Khaneja, Brockett
and Glaser 2001).  Up to local gates a gate is a point c of the Weyl chamber
of Chapter 7, U = k1 exp{(i/2)(c1 XX + c2 YY + c3 ZZ)} k2.  The drift alone
gives c = (Jt, 0, 0), and local gates turn the coupling into any of
+-XX, +-YY, +-ZZ, so that the classes reachable in the time t are those with a
representative in the octahedron |c1| + |c2| + |c3| <= Jt (the convex hull of
the Weyl orbit of (Jt, 0, 0), by Kostant's convexity theorem).  Since each c_i
may be shifted by pi, the least time to the class c is
    J t*(c) = dist(c1, pi Z) + dist(c2, pi Z) + dist(c3, pi Z),
which in the chamber is min(c1, pi - c1) + c2 + c3.  Hence CNOT (L) needs
pi/(2J), sqrt(iSWAP) (Q) also pi/(2J), the square roots of SWAP (P, N)
3 pi/(4J), iSWAP (A2) pi/J and SWAP (A3) 3 pi/(2J), the longest of all.

(a) The level sets J t* = pi/4, pi/2, pi and 5 pi/4 in the chamber: the
    fronts reached in those times, which start at the two local corners O and
    A1 and close on SWAP last.  The bold path O -> L -> A2 -> A3 is a
    time-optimal route to SWAP: three stretches of the coupling, each of
    duration pi/(2J), turned by local gates into ZZ, YY and XX.
(b) A check by sampling: 1500 sequences k0 e^{-i H_d tau_1} k1 ... e^{-i H_d tau_4} k4
    with random local k_j and random durations of total J sum tau = pi/2,
    reduced into the chamber.  They fill the region J t* <= pi/2, the two
    tetrahedra at O and A1 that meet at CNOT, and none lies beyond it.
Writes figures/ch09_kbg.tex.
"""
import itertools
import numpy as np
from scipy.linalg import expm
import needham as N
import ch07_weyl as W

PI = np.pi
J = 1.0
HD = 0.5 * J * W.ZZ


def reduce_to_chamber(c):
    """A point of the chamber pi - c2 >= c1 >= c2 >= c3 >= 0 equivalent to c (folded so that c1 <= pi/2 on the base)."""
    for perm in itertools.permutations(range(3)):
        for flips in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)):
            e = np.mod(np.array([c[i] for i in perm]) * np.array(flips), PI)
            e = np.where(np.isclose(e, PI, atol=1e-12), 0.0, e)
            if W.in_chamber(e, 1e-9):
                if abs(e[2]) < 1e-9 and e[0] > PI / 2:
                    e[0] = PI - e[0]
                return e
    return None


def cartan(U):
    """Chamber coordinates of a two-qubit gate, from the eigenvalues of m(U), checked against Makhlin's invariants."""
    U = U / np.linalg.det(U) ** 0.25
    UB = W.MAGIC.conj().T @ U @ W.MAGIC
    th = np.angle(np.linalg.eigvals(UB.T @ UB))
    G = W.makhlin(U)
    for perm in itertools.permutations(range(4)):
        t = th[list(perm)]
        for k in itertools.product((-1, 0, 1), repeat=4):
            tt = t + 2 * PI * np.array(k)
            if abs(tt.sum()) > 1e-6:
                continue
            e = reduce_to_chamber(np.array([(tt[0] + tt[1]) / 2, (tt[1] + tt[3]) / 2, (tt[0] + tt[3]) / 2]))
            if e is not None and np.allclose(W.makhlin_c(e), G, atol=1e-7):
                return e
    raise ValueError("no chamber point found")


def least_time(c):
    """J t* for a point of the chamber."""
    return min(c[0], PI - c[0]) + c[1] + c[2]


def level_polygon(s, half):
    """The part of the level set J t* = s in the half c1 <= pi/2 (half = 0) or c1 >= pi/2 (half = 1) of the chamber."""
    # the half-chamber as {x : A x <= b}
    A = [[-1, 1, 0], [0, -1, 1], [0, 0, -1], [1, 1, 0]]
    bb = [0, 0, 0, PI]
    A.append([1, 0, 0] if half == 0 else [-1, 0, 0])
    bb.append(PI / 2 if half == 0 else -PI / 2)
    A, bb = np.array(A, float), np.array(bb, float)
    nrm = np.array([1.0, 1, 1]) if half == 0 else np.array([-1.0, 1, 1])
    off = s if half == 0 else s - PI                                  # nrm . c = off
    pts = []
    for i, j in itertools.combinations(range(len(A)), 2):
        M = np.array([A[i], A[j], nrm])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, [bb[i], bb[j], off])
        if np.all(A @ x <= bb + 1e-9) and not any(np.allclose(x, p) for p in pts):
            pts.append(x)
    if len(pts) < 3:
        return []
    ctr = np.mean(pts, 0)
    e1 = pts[0] - ctr
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm / np.linalg.norm(nrm), e1)
    return sorted(pts, key=lambda p: np.arctan2((p - ctr) @ e2, (p - ctr) @ e1))


def sample(rng, total, n=1500, segs=4):
    out = []
    for _ in range(n):
        tau = rng.dirichlet(np.ones(segs)) * total / J
        U = W.random_local(rng)
        for t in tau:
            U = W.random_local(rng) @ expm(-1j * HD * t) @ U
        out.append(cartan(U))
    return np.array(out)


def checks(rng):
    # the extraction recovers known gates and random classes
    V0 = W.VERTICES
    for G, k in [(W.CNOT, "L"), (W.SWAP, "A3"), (W.ISWAP, "A2"), (W.CZ, "L")]:
        assert np.allclose(cartan(G), V0[k])
    for _ in range(30):
        c = reduce_to_chamber(rng.uniform(0, PI, 3))
        U = W.random_local(rng) @ W.nonlocal_gate(c) @ W.random_local(rng)
        assert np.allclose(W.makhlin_c(cartan(U)), W.makhlin_c(c), atol=1e-8)
    # the drift alone runs along the edge O L, reaching CNOT at J t = pi/2
    for t in (0.3, 0.9, PI / 2):
        assert np.allclose(cartan(expm(-1j * HD * t)), [t, 0, 0], atol=1e-9)
    # the times of the named gates
    times = {k: least_time(np.array(V0[k])) for k in ("L", "Q", "P", "N", "A2", "A3")}
    assert np.allclose([times[k] for k in ("L", "Q", "P", "N", "A2", "A3")],
                       [PI / 2, PI / 2, 3 * PI / 4, 3 * PI / 4, PI, 3 * PI / 2])
    # the route O -> L -> A2 -> A3 makes SWAP from three stretches of pi/(2J)
    H = PI / 2
    U = expm(-1j * HD * H)
    for P in (np.kron(W.X, W.I2) @ np.kron(W.I2, W.I2), None):
        pass
    Hy = 0.5 * J * W.YY
    Hx = 0.5 * J * W.XX
    S = expm(-1j * Hx * H) @ expm(-1j * Hy * H) @ expm(-1j * HD * H)
    assert np.allclose(cartan(S), V0["A3"])
    # the sampled gates never exceed the bound, and come close to it
    pts = sample(rng, PI / 2, n=400)
    f = np.array([least_time(p) for p in pts])
    assert f.max() <= PI / 2 + 1e-7 and f.max() > 0.97 * PI / 2
    return times


def main():
    rng = np.random.default_rng(11)
    checks(rng)
    pts = sample(np.random.default_rng(5), PI / 2)
    f = np.array([least_time(p) for p in pts])
    assert f.max() <= PI / 2 + 1e-7
    V0 = {k: W.unit(v) for k, v in W.VERTICES.items()}
    ch = [V0["O"], V0["A1"], V0["A2"], V0["A3"]]
    az, el = -75.0, 22.0
    b = []
    # ---- (a) the fronts of least time
    Va = N.View(az=az, el=el, R=6.0, shift=(1.0, 0.0))
    b += W.draw_polytope(Va, ch, fill="ndPaper")
    fronts = [(PI / 4, "ndSky", "\\frac{\\pi}{4}"), (PI / 2, "ndBlue!35", "\\frac{\\pi}{2}"),
              (PI, "ndGold!45", "\\pi"), (5 * PI / 4, "ndRose", "\\frac{5\\pi}{4}")]
    for s_, col, txt in fronts:
        for half in (0, 1):
            poly = level_polygon(s_, half)
            if poly:
                b.append("\\filldraw[fill=%s, fill opacity=0.8, draw=ndInk!70, line width=0.45pt] %s -- cycle;"
                         % (col, " -- ".join(Va.pt(W.unit(p)) for p in poly)))
        # label each front where it meets the edge O A3 (half 0) at c = (s/3, s/3, s/3)
        q = Va.P(W.unit(np.array([s_ / 3] * 3)))
        b.append("\\node[font=\\scriptsize, anchor=east, text=ndInk!80] at %s {$Jt=%s$};" % (N.pt(q + np.array([-0.1, 0.0])), txt))
    b += W.draw_polytope(Va, ch)
    route = [V0["O"], V0["L"], V0["A2"], V0["A3"]]
    b.append("\\draw[line width=1.8pt, ndRed, ->, >=stealth] %s;" % " -- ".join(Va.pt(p) for p in route))
    for k in ("O", "A1", "L", "P", "A2", "A3"):
        b.append("\\node[dot, minimum size=3.4pt] at %s {};" % Va.pt(V0[k]))
    lab = {"O": ("$O$", (-0.12, -0.05), "north east"), "A1": ("$A_1$", (0.12, 0.0), "west"),
           "L": ("CNOT, $\\frac{\\pi}{2J}$", (-0.05, -0.14), "north"),
           "A3": ("SWAP, $\\frac{3\\pi}{2J}$", (0.0, 0.14), "south"),
           "P": ("$\\sqrt{\\mathrm{SWAP}}$, $\\frac{3\\pi}{4J}$", (0.14, -0.02), "west")}
    for k, (txt, d, anc) in lab.items():
        b.append("\\node[font=\\scriptsize, anchor=%s, fill=ndPaper, inner sep=1pt] at %s {%s};" % (anc, N.pt(Va.P(V0[k]) + np.array(d)), txt))
    b.append("\\node[font=\\scriptsize, anchor=west] (lA2) at (7.0,0.85) {iSWAP, $\\frac{\\pi}{J}$};")
    b.append("\\draw[pointer] (lA2.west) to[bend left=10] %s;" % N.pt(Va.P(V0["A2"]) + np.array([0.06, 0.0])))
    b.append("\\node[note, anchor=west, text=ndRed] (rt) at %s {a fastest route to SWAP:\\\\ $ZZ$, then $YY$, then $XX$,\\\\ each for $\\frac{\\pi}{2J}$};" % N.pt(Va.P(V0["A3"]) + np.array([1.9, -0.4])))
    b.append("\\draw[pointer] (rt.west) to[bend right=15] %s;" % N.pt(Va.P((V0["A2"] + V0["A3"]) / 2) + np.array([0.06, 0.0])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the least time to each class of gates};" % N.pt(Va.P(V0["L"]) + np.array([0.0, -0.95])))

    # ---- (b) the sampled gates for J t = pi/2
    Vb = N.View(az=az, el=el, R=4.2, shift=(9.6, 0.4))
    b += W.draw_polytope(Vb, ch, fill="ndPaper")
    for half in (0, 1):
        poly = level_polygon(PI / 2, half)
        b.append("\\filldraw[fill=ndBlue!25, fill opacity=0.6, draw=ndBlue!70!ndInk, line width=0.5pt] %s -- cycle;"
                 % " -- ".join(Vb.pt(W.unit(p)) for p in poly))
    for p in pts:
        b.append("\\fill[ndInk!80] %s circle (0.45pt);" % Vb.pt(W.unit(p)))
    b += W.draw_polytope(Vb, ch)
    for k in ("O", "A1", "L", "A3"):
        b.append("\\node[dot, minimum size=2.8pt] at %s {};" % Vb.pt(V0[k]))
    b.append("\\node[font=\\scriptsize, anchor=north] at %s {CNOT};" % N.pt(Vb.P(V0["L"]) + np.array([0, -0.1])))
    b.append("\\node[note, anchor=south] at %s {1500 random sequences of\\\\ local gates and $\\ee^{-\\imath H_d\\tau}$,\\\\ $J\\sum\\tau=\\frac\\pi2$};" % N.pt(Vb.P(V0["A3"]) + np.array([0.0, 0.35])))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) all of them reach $Jt^\\star\\le\\frac\\pi2$};" % N.pt(Vb.P(V0["L"]) + np.array([0.0, -1.35])))
    N.write("ch09_kbg", b)
    return f


if __name__ == "__main__":
    f = main()
    print("sampled: max J t* = %.4f (bound pi/2 = %.4f)" % (f.max(), PI / 2))
    print("written figures/ch09_kbg.tex")
