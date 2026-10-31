"""ch07_weyl.py -- the Weyl chamber of two-qubit gates.

Every two-qubit gate is U = k1 exp{(i/2)(c1 XX + c2 YY + c3 ZZ)} k2 with local
k1, k2 (the Cartan, or KAK, decomposition), and its class under local gates is
fixed by the point c = (c1, c2, c3) of the Weyl chamber
    pi - c2 >= c1 >= c2 >= c3 >= 0,
the tetrahedron O A1 A2 A3 of Zhang, Vala, Sastry and Whaley, PRA 67, 042313
(2003), whose base c3 = 0 is folded along the line L A2.  Makhlin's local
invariants G1, G2 are computed from the matrix of a gate in the magic basis.
The perfect entanglers, the gates that turn some product state into a
maximally entangled one, are those for which the convex hull of the
eigenvalues of m(U) contains 0; in the chamber they fill the polyhedron
L M N P Q A2, half its volume.

This module holds the helpers used by ch07_entanglers.py and ch07_makhlin.py.
"""
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
XX, YY, ZZ = np.kron(X, X), np.kron(Y, Y), np.kron(Z, Z)
MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / np.sqrt(2)

PI = np.pi
VERTICES = {                                   # the chamber and the midpoints of its edges
    "O": (0, 0, 0), "A1": (PI, 0, 0), "A2": (PI / 2, PI / 2, 0), "A3": (PI / 2, PI / 2, PI / 2),
    "L": (PI / 2, 0, 0), "M": (3 * PI / 4, PI / 4, 0), "N": (3 * PI / 4, PI / 4, PI / 4),
    "P": (PI / 4, PI / 4, PI / 4), "Q": (PI / 4, PI / 4, 0),
}
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)


def nonlocal_gate(c):
    """exp{(i/2)(c1 XX + c2 YY + c3 ZZ)}, the middle factor of the KAK decomposition."""
    return expm(0.5j * (c[0] * XX + c[1] * YY + c[2] * ZZ))


def makhlin(U):
    """Makhlin's invariants (G1, G2) from the matrix of a gate, via m = U_B^T U_B in the magic basis."""
    UB = MAGIC.conj().T @ U @ MAGIC
    m = UB.T @ UB
    det = np.linalg.det(U)
    tr, tr2 = np.trace(m), np.trace(m @ m)
    return tr ** 2 / (16 * det), np.real_if_close((tr ** 2 - tr2) / (4 * det))


def makhlin_c(c):
    """The same invariants in the coordinates of the chamber."""
    c1, c2, c3 = c
    cc = np.cos(c1) ** 2 * np.cos(c2) ** 2 * np.cos(c3) ** 2
    ss = np.sin(c1) ** 2 * np.sin(c2) ** 2 * np.sin(c3) ** 2
    G1 = cc - ss + 0.25j * np.sin(2 * c1) * np.sin(2 * c2) * np.sin(2 * c3)
    G2 = 4 * cc - 4 * ss - np.cos(2 * c1) * np.cos(2 * c2) * np.cos(2 * c3)
    return G1, G2


def m_phases(c):
    """The eigenvalues of m for the gate with coordinates c, as points of the unit circle."""
    c1, c2, c3 = c
    return np.exp(1j * np.array([c1 - c2 + c3, c1 + c2 - c3, -(c1 + c2 + c3), -c1 + c2 + c3]))


def hull_contains_zero(z, tol=1e-12):
    """Do points of the unit circle surround 0?  Yes iff no angular gap exceeds pi."""
    a = np.sort(np.angle(z))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * PI]]))
    return gaps.max() <= PI + tol


def perfect_entangler(U):
    UB = MAGIC.conj().T @ U @ MAGIC
    return hull_contains_zero(np.linalg.eigvals(UB.T @ UB))


def in_chamber(c, tol=1e-12):
    c1, c2, c3 = c
    return c1 >= c2 - tol and c2 >= c3 - tol and c3 >= -tol and c1 + c2 <= PI + tol


def perfect_entangler_c(c, tol=1e-12):
    """The polyhedron L M N P Q A2: c1 + c2 >= pi/2, c1 - c2 <= pi/2, c2 + c3 <= pi/2."""
    c1, c2, c3 = c
    return c1 + c2 >= PI / 2 - tol and c1 - c2 <= PI / 2 + tol and c2 + c3 <= PI / 2 + tol


def haar(rng, n):
    Zm = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(Zm)
    d = np.diag(r)
    return q * (d / abs(d))


def random_local(rng):
    return np.kron(haar(rng, 2), haar(rng, 2))


def tet_volume(a, b, c, d):
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    return abs(np.linalg.det(np.array([b - a, c - a, d - a]))) / 6


def checks(rng=None):
    """The facts the figures rely on."""
    rng = np.random.default_rng(7) if rng is None else rng
    for _ in range(200):
        c = rng.uniform(0, PI, 3)
        G1, G2 = makhlin(nonlocal_gate(c))
        H1, H2 = makhlin_c(c)
        assert abs(G1 - H1) < 1e-10 and abs(G2 - H2) < 1e-10
        U = random_local(rng) @ nonlocal_gate(c) @ random_local(rng) * np.exp(1j * rng.uniform(0, 2 * PI))
        K1, K2 = makhlin(U)
        assert abs(K1 - H1) < 1e-9 and abs(K2 - H2) < 1e-9            # invariant under local gates and phase
        assert np.allclose(np.sort_complex(np.linalg.eigvals((MAGIC.conj().T @ nonlocal_gate(c) @ MAGIC).T
                                                             @ (MAGIC.conj().T @ nonlocal_gate(c) @ MAGIC))),
                           np.sort_complex(m_phases(c)))
    # named gates sit at the named points
    for U, v in [(CNOT, "L"), (CZ, "L"), (SWAP, "A3"), (ISWAP, "A2"), (np.eye(4), "O"), (np.eye(4), "A1")]:
        assert np.allclose(makhlin(U), makhlin_c(VERTICES[v]))
    # the square root of SWAP with the phase -i on the singlet lies at P, as in the paper; its inverse at N
    sqrt_swap = expm(0.25j * PI * (SWAP - np.eye(4)))
    assert np.allclose(sqrt_swap @ sqrt_swap, SWAP)
    assert np.allclose(makhlin(sqrt_swap), makhlin_c(VERTICES["P"]))
    assert np.allclose(makhlin(sqrt_swap.conj().T), makhlin_c(VERTICES["N"]))
    # the base is folded along L A2: [c1, c2, 0] and [pi - c1, c2, 0] are one class
    for _ in range(50):
        c1, c2 = rng.uniform(0, PI, 2)
        assert np.allclose(makhlin_c((c1, c2, 0)), makhlin_c((PI - c1, c2, 0)))
    # the perfect entanglers: hull criterion = polyhedron, half the volume of the chamber
    pts = rng.uniform(0, PI, (240000, 3))
    pts = pts[[in_chamber(p) for p in pts]]
    agree = [hull_contains_zero(m_phases(p)) == perfect_entangler_c(p) for p in pts]
    assert np.mean(agree) > 0.9995
    V = VERTICES
    v_ch = tet_volume(V["O"], V["A1"], V["A2"], V["A3"])
    v_out = [tet_volume(V["L"], V["Q"], V["P"], V["O"]), tet_volume(V["N"], V["P"], V["A2"], V["A3"]),
             tet_volume(V["L"], V["M"], V["N"], V["A1"])]
    assert np.isclose(v_ch, PI ** 3 / 24) and np.allclose(v_out, [PI ** 3 / 192, PI ** 3 / 96, PI ** 3 / 192])
    assert np.isclose(v_ch - sum(v_out), PI ** 3 / 48)
    frac = np.mean([perfect_entangler_c(p) for p in pts])
    assert abs(frac - 0.5) < 0.01
    return frac


# ---------------------------------------------------------------- drawing (coordinates in units of pi)

def unit(c):
    return np.asarray(c, dtype=float) / PI


def polytope_edges(verts, d):
    """Edges of the convex hull of `verts` (rows), each with a flag: True if on a face turned towards d."""
    from scipy.spatial import ConvexHull
    verts = np.asarray(verts, dtype=float)
    hull = ConvexHull(verts)
    planes = []                                                  # distinct facet planes (normal, offset)
    for eq in hull.equations:
        if not any(np.allclose(eq, q, atol=1e-9) for q in planes):
            planes.append(eq)
    out = []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            on = [q for q in planes if abs(q[:3] @ verts[i] + q[3]) < 1e-9 and abs(q[:3] @ verts[j] + q[3]) < 1e-9]
            if len(on) >= 2:                                     # an edge lies on two facet planes
                out.append((i, j, any(q[:3] @ d > 0 for q in on)))
    return out


def draw_polytope(V, verts, front="ink", back="dashedink", fill=None):
    verts = [np.asarray(v, dtype=float) for v in verts]
    b = []
    if fill:
        from scipy.spatial import ConvexHull
        P2 = np.array([V.P(v) for v in verts])
        h = ConvexHull(P2)
        b.append("\\fill[%s] %s -- cycle;" % (fill, " -- ".join(N_pt(P2[k]) for k in h.vertices)))
    for i, j, vis in polytope_edges(verts, V.d):
        style = front if vis else back
        if style:
            b.append("\\draw[%s] %s -- %s;" % (style, V.pt(verts[i]), V.pt(verts[j])))
    return b


def N_pt(p):
    return "(%.3f,%.3f)" % (p[0], p[1])


def axes(V, origin, length, labels=("c_1", "c_2", "c_3"), font="\\scriptsize"):
    o = np.asarray(origin, dtype=float)
    b = []
    for e, lab in zip(np.eye(3), labels):
        q = V.P(o + length * e)
        u = q - V.P(o)
        u = u / np.linalg.norm(u)
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (V.pt(o), N_pt(q)))
        b.append("\\node[font=%s, text=ndInk!80] at %s {$%s$};" % (font, N_pt(q + 0.2 * u), lab))
    return b


GATES = {                                                       # label, gate names, page offset (cm) for panel (d)
    "O": ("O", "\\Id"), "A1": ("A_1", "\\Id"), "A2": ("A_2", "\\mathrm{iSWAP}"),
    "A3": ("A_3", "\\mathrm{SWAP}"), "L": ("L", "\\mathrm{CNOT},\\ \\mathrm{CZ}"),
    "P": ("P", "\\sqrt{\\mathrm{SWAP}}"), "N": ("N", "\\sqrt{\\mathrm{SWAP}}^{\\,\\dagger}"),
    "Q": ("Q", "\\sqrt{\\mathrm{iSWAP}}"), "M": ("M", "\\equiv Q"),
}


def chamber_figure():
    import needham as N
    V0 = {k: unit(v) for k, v in VERTICES.items()}
    cube = [np.array(p, dtype=float) for p in np.ndindex(2, 2, 2)]
    centre = np.array([0.5, 0.5, 0.5])
    assert np.allclose(centre, V0["A3"])
    pyr = [np.array([0, 0, 0.]), np.array([1, 0, 0.]), np.array([1, 1, 0.]), np.array([0, 1, 0.]), centre]
    ch = [V0["O"], V0["A1"], V0["A2"], V0["A3"]]
    # the chamber has volume pi^3/24; the pyramid holds four of them and the cube six pyramids
    assert np.isclose(tet_volume(*ch), 1 / 24) and np.isclose(4 * tet_volume(*ch), 1 / 6)
    # the bottom pyramid is the part of the cube below the four planes c1 = c3, c1 + c3 = pi, c2 = c3, c2 + c3 = pi,
    # and the chamber the part of the pyramid with c2 <= c1 and c1 + c2 <= pi
    rng = np.random.default_rng(3)
    for p in rng.uniform(0, 1, (4000, 3)):
        in_pyr = abs(p[0] - 0.5) <= 0.5 - p[2] and abs(p[1] - 0.5) <= 0.5 - p[2]
        below = p[0] >= p[2] and p[0] + p[2] <= 1 and p[1] >= p[2] and p[1] + p[2] <= 1
        assert in_pyr == below
        assert in_chamber(PI * p) == (below and p[1] <= p[0] and p[0] + p[1] <= 1)
    # the controlled-U gates fill the edge O L: exp(i gamma n.sigma) controlled has coordinates [gamma, 0, 0]
    for g in np.linspace(0.05, PI / 2, 7):
        u = expm(1j * g * Z)
        CU = np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), u]])
        assert np.allclose(makhlin(CU), makhlin_c((g, 0, 0)))

    b = []
    az, el = -75.0, 22.0
    Rs, ytop = 1.5, 6.2
    xs = [0.0, 3.35, 6.7]                                       # origins of the small panels, top row
    # ---- (a) the cube, with the half-diagonals that bound the six pyramids
    Va = N.View(az=az, el=el, R=Rs, shift=(xs[0], ytop))
    b += draw_polytope(Va, pyr, front=None, back=None, fill="ndSky, fill opacity=0.8")
    for v in cube:
        b.append("\\draw[line width=0.35pt, ndInk!50] %s -- %s;" % (Va.pt(centre), Va.pt(v)))
    b += draw_polytope(Va, cube)
    b.append("\\node[dot, minimum size=2.6pt] at %s {};" % Va.pt(centre))
    for e, lab, (dx, dy) in zip(np.eye(3), ("c_1", "c_2", "c_3"), ((0.0, -0.17), (0.14, -0.1), (-0.17, 0.0))):
        b.append("\\node[font=\\scriptsize, text=ndInk!80] at %s {$%s$};" % (N_pt(Va.P(0.5 * e) + np.array([dx, dy])), lab))
    b.append("\\node[font=\\scriptsize, text=ndInk!80] at %s {$\\pi$};" % N_pt(Va.P((1, 0, 0)) + np.array([0.05, -0.17])))
    # ---- (b) the bottom pyramid, cut by c1 = c2 and c1 + c2 = pi into four tetrahedra
    Vb = N.View(az=az, el=el, R=Rs, shift=(xs[1], ytop))
    b += draw_polytope(Vb, ch, front=None, back=None, fill="ndSky, fill opacity=0.95")
    b += draw_polytope(Vb, pyr)
    for v in ([0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]):
        b.append("\\draw[line width=0.4pt, ndInk!60, densely dotted] %s -- %s;" % (Vb.pt(v), Vb.pt([0.5, 0.5, 0])))
    b.append("\\draw[line width=0.4pt, ndInk!60, densely dotted] %s -- %s;" % (Vb.pt([0.5, 0.5, 0]), Vb.pt(centre)))
    b.append("\\node[dot, minimum size=2.6pt] at %s {};" % Vb.pt(centre))
    # ---- (c) the chamber
    Vc = N.View(az=az, el=el, R=Rs, shift=(xs[2], ytop))
    b += draw_polytope(Vc, ch, fill="ndSky, fill opacity=0.95")
    for k, (dx, dy, anc) in {"O": (-0.05, 0.0, "east"), "A1": (0.05, -0.02, "west"), "A2": (0.04, 0.06, "south west"),
                             "A3": (0.0, 0.06, "south")}.items():
        b.append("\\node[dot, minimum size=2.6pt] at %s {};" % Vc.pt(V0[k]))
        b.append("\\node[font=\\scriptsize, anchor=%s] at %s {$%s$};" % (anc, N_pt(Vc.P(V0[k]) + np.array([dx, dy])), GATES[k][0]))
    # arrows between the small panels, with the mirrors that cut
    yarr = ytop + 0.5
    for x0, txt in [(xs[0] + 2.2, "$c_i=c_3$,\\\\ $c_i+c_3=\\pi$"), (xs[1] + 2.1, "$c_1=c_2$,\\\\ $c_1+c_2=\\pi$")]:
        b.append("\\draw[->, >=stealth, line width=0.6pt, ndInk!70] (%.2f,%.2f) -- (%.2f,%.2f);" % (x0, yarr, x0 + 0.85, yarr))
        b.append("\\node[note, anchor=south, font=\\scriptsize\\itshape] at (%.2f,%.2f) {%s};" % (x0 + 0.43, yarr + 0.12, txt))
    ycap = ytop - 0.45
    for x, txt in [(xs[0] + 0.95, "(a) the cube of $(c_1,c_2,c_3)$:\\\\ six pyramids"), (xs[1] + 0.95, "(b) one pyramid:\\\\ four tetrahedra"),
                   (xs[2] + 0.8, "(c) a Weyl chamber,\\\\ of volume $\\pi^3/24$")]:
        b.append("\\node[font=\\small, align=center, anchor=north] at (%.2f,%.2f) {%s};" % (x, ycap, txt))

    # ---- (d) the chamber as a map of the two-qubit gates
    Vd = N.View(az=az, el=el, R=6.0, shift=(1.2, 0.0))
    P = {k: Vd.P(v) for k, v in V0.items()}
    base1 = [V0["O"], V0["L"], V0["A2"]]
    base2 = [V0["A1"], V0["L"], V0["A2"]]
    b.append("\\fill[ndGold!40] %s -- %s -- %s -- cycle;" % tuple(Vd.pt(p) for p in base1))
    b.append("\\fill[ndGold!18] %s -- %s -- %s -- cycle;" % tuple(Vd.pt(p) for p in base2))
    b += draw_polytope(Vd, ch, fill="ndSky, fill opacity=0.4")
    b.append("\\draw[line width=0.5pt, ndGold!60!ndInk, densely dashed] %s -- %s;" % (Vd.pt(V0["L"]), Vd.pt(V0["A2"])))
    b.append("\\draw[line width=2.0pt, ndBlue] %s -- %s;" % (Vd.pt(V0["O"]), Vd.pt(V0["L"])))
    for k in V0:
        b.append("\\node[dot, fill=%s, minimum size=3.4pt] at %s {};" % ("ndRed" if k == "L" else "ndInk", N_pt(P[k])))
    lab = lambda k: "$%s$:\\ $%s$" % GATES[k]
    # points on the outline: labels beside them
    for k, (dx, dy, anc) in {"O": (-0.12, 0.0, "east"), "A1": (0.12, 0.0, "west"), "A3": (0.0, 0.12, "south"),
                             "L": (-0.08, -0.12, "north east"), "P": (-0.12, 0.05, "south east"),
                             "N": (0.12, 0.05, "west")}.items():
        b.append("\\node[font=\\scriptsize, anchor=%s] at %s {%s};" % (anc, N_pt(P[k] + np.array([dx, dy])), lab(k)))
    # points behind the front faces: labels outside, with pointers
    for k, pos, anc, bend in [("Q", (-0.55, 1.15), "east", "left"), ("A2", (7.6, 0.95), "west", "right"),
                              ("M", (7.6, 0.12), "west", "left")]:
        b.append("\\node[font=\\scriptsize, anchor=%s] (l%s) at (%.2f,%.2f) {%s};" % (anc, k, pos[0], pos[1], lab(k)))
        b.append("\\draw[pointer] (l%s.%s) to[bend %s=10] %s;" % (k, "east" if anc == "east" else "west", bend,
                                                                    N_pt(P[k] + np.array([0.06 if anc == "east" else -0.06, 0.0]))))
    mid = Vd.P(0.25 * V0["L"])
    b.append("\\node[note, anchor=north, text=ndBlue] (cu) at %s {the controlled-$U$ gates};" % N_pt(mid + np.array([-0.2, -0.85])))
    b.append("\\draw[pointer] (cu.north) -- %s;" % N_pt(mid + np.array([0.0, -0.06])))
    # the fold of the base
    p1 = Vd.P((V0["O"] + V0["L"] + V0["A2"]) / 3)
    p2 = Vd.P((V0["A1"] + V0["L"] + V0["A2"]) / 3)
    b.append("\\draw[<->, >=stealth, line width=0.7pt, ndGold!40!ndInk] %s to[bend right=50] %s;" % (N_pt(p1), N_pt(p2)))
    b.append("\\node[note, anchor=north west] (fold) at %s {the two halves of the base are one class,\\\\ $[c_1,c_2,0]\\sim[\\pi-c_1,c_2,0]$};"
             % N_pt(np.array([P["L"][0] + 1.0, P["A1"][1] - 0.3])))
    b.append("\\draw[pointer] (fold.north west) to[bend left=20] %s;" % N_pt(0.5 * (p1 + p2) + np.array([0.35, -0.12])))
    b.append("\\node[font=\\small, anchor=north] at %s {(d) the Weyl chamber as a map of the two-qubit gates};"
             % N_pt(np.array([P["L"][0], P["A1"][1] - 1.35])))
    return b

if __name__ == "__main__":
    import needham as N
    print("fraction of the chamber filled by perfect entanglers: %.3f" % checks())
    N.write("ch07_weyl", chamber_figure())
    print("written figures/ch07_weyl.tex")
