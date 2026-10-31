"""ch07_makhlin.py -- the Weyl chamber seen through Makhlin's local invariants.

Makhlin's invariants of a two-qubit gate U are G1 = tr^2(m) / (16 det U) and
G2 = (tr^2(m) - tr(m^2)) / (4 det U), where m = U_B^T U_B and U_B is U in the
magic basis; two gates are locally equivalent exactly when their invariants
agree.  In the coordinates c of the Weyl chamber O A1 A2 A3 (Zhang, Vala,
Sastry and Whaley, PRA 67, 042313, 2003)
    G1 = cos^2 c1 cos^2 c2 cos^2 c3 - sin^2 c1 sin^2 c2 sin^2 c3
         + (i/4) sin 2c1 sin 2c2 sin 2c3,
    G2 = 4 cos^2 c1 cos^2 c2 cos^2 c3 - 4 sin^2 c1 sin^2 c2 sin^2 c3
         - cos 2c1 cos 2c2 cos 2c3.
The map c -> (Re G1, Im G1, G2) carries the flat tetrahedron to a curved solid.
The reflection c1 -> pi - c1 conjugates G1, so the solid is symmetric in the
plane Im G1 = 0: the half c1 <= pi/2 of the chamber (O L A2 A3) fills the
side Im G1 >= 0, the other half the side Im G1 <= 0.  The base c3 = 0 and the
triangle L A2 A3 (c1 = pi/2) both land in the plane Im G1 = 0.  O and A1 meet
at the identity, and the base is folded along L A2: [c1, c2, 0] and
[pi - c1, c2, 0] have the same invariants.
(a) the solid in three dimensions, with the gates and the perfect entanglers;
(b) the fold of the base, in the plane of (c1, c2) and in that of (G1, G2).
Writes figures/ch07_makhlin.tex.
"""
import numpy as np
import needham as N
from ch07_weyl import (VERTICES, PI, makhlin, makhlin_c, nonlocal_gate, in_chamber, perfect_entangler_c,
                       checks, CNOT, SWAP, ISWAP)

V3 = {k: np.array(v, dtype=float) for k, v in VERTICES.items()}
# an oblique projection of (Re G1, Im G1, G2): Re G1 to the right, G2 upwards, and Im G1 receding up and
# to the left, so that the band of the plane Im G1 = 0 runs along one diagonal and the bulge across it
PROJ = np.array([[1.9, -5.0, 0.0],
                 [0.0, 4.4, 0.7]])
def G(c):
    """(Re G1, Im G1, G2) at the chamber point c."""
    g1, g2 = makhlin_c(c)
    return np.array([np.real(g1), np.imag(g1), np.real(g2)])


def image(a, b, n=161):
    """The image of the straight segment from a to b in the chamber."""
    return np.array([G((1 - s) * a + s * b) for s in np.linspace(0, 1, n)])


def check_figure():
    """The facts drawn below."""
    checks()
    v = V3
    assert np.allclose(G(v["O"]), [1, 0, 3]) and np.allclose(G(v["A1"]), [1, 0, 3])      # O and A1 meet
    for name, gate in [("L", CNOT), ("A3", SWAP), ("A2", ISWAP)]:
        g1, g2 = makhlin(gate)
        assert np.allclose(G(v[name]), [np.real(g1), np.imag(g1), np.real(g2)])
    assert np.allclose(G(v["L"]), [0, 0, 1]) and np.allclose(G(v["A2"]), [0, 0, -1]) and np.allclose(G(v["A3"]), [-1, 0, -3])
    assert np.allclose(G(v["P"]), [0, 0.25, 0]) and np.allclose(G(v["N"]), [0, -0.25, 0])
    assert np.allclose(G(v["Q"]), [0.25, 0, 1]) and np.allclose(G(v["M"]), G(v["Q"]))
    rng = np.random.default_rng(11)
    for c1, c2 in rng.uniform(0, PI, (200, 2)):                                           # the fold of the base
        assert np.allclose(G((c1, c2, 0)), G((PI - c1, c2, 0)))
        assert abs(G((c1, c2, 0))[1]) < 1e-12                                              # Im G1 = 0 on the base
    for c in rng.uniform(0, PI, (200, 3)):                                                 # c1 -> pi - c1 conjugates G1
        g, h = G(c), G((PI - c[0], c[1], c[2]))
        assert np.allclose([g[0], -g[1], g[2]], h)
    t = np.linspace(0, PI, 201)
    assert np.allclose([G((x, 0, 0)) for x in t], np.stack([np.cos(t) ** 2, 0 * t, 2 * np.cos(t) ** 2 + 1], 1))   # O L A1
    s = np.linspace(0, PI / 2, 101)
    assert np.allclose([G((PI / 2, x, 0)) for x in s], np.stack([0 * s, 0 * s, np.cos(2 * s)], 1))                # L A2
    assert np.allclose([G((x, x, 0)) for x in s],
                       np.stack([np.cos(s) ** 4, 0 * s, 4 * np.cos(s) ** 4 - np.cos(2 * s) ** 2], 1))               # O A2
    assert np.allclose([G((PI - x, x, 0)) for x in s], [G((x, x, 0)) for x in s])                                 # A1 A2
    assert np.allclose([G((PI / 2, PI / 2, x)) for x in s],
                       np.stack([-np.sin(s) ** 2, 0 * s, -2 * np.sin(s) ** 2 - 1], 1))                              # A2 A3
    # the range of Im G1 over the chamber: +-1/4, at P and at N
    g = np.linspace(0, PI, 121)
    pts = np.array([(a, b, c) for a in g for b in g[g <= PI / 2 + 1e-9] for c in g[g <= PI / 2 + 1e-9]
                    if in_chamber((a, b, c))])
    im = np.array([G(p)[1] for p in pts])
    hi, lo = pts[np.argmax(im)], pts[np.argmin(im)]
    assert np.isclose(im.max(), 0.25) and np.isclose(im.min(), -0.25)
    assert np.allclose(hi, v["P"]) and np.allclose(lo, v["N"])
    # a gate made from the coordinates lands where the formulas say, also after local gates (in checks)
    for c in rng.uniform(0, PI, (20, 3)):
        g1, g2 = makhlin(nonlocal_gate(c))
        assert np.allclose(G(c), [np.real(g1), np.imag(g1), np.real(g2)])
    # the B gate [pi/2, pi/4, 0] sits at the origin of the space of invariants
    assert np.allclose(G((PI / 2, PI / 4, 0)), 0)
    return im.max(), hi, im.min(), lo


class Page:
    """The oblique view of the space (Re G1, Im G1, G2); Im G1 > 0 lies behind the plane Im G1 = 0."""

    def __init__(self, shift):
        self.o = np.asarray(shift, dtype=float)

    def P(self, g):
        return self.o + PROJ @ np.asarray(g, dtype=float)

    def pt(self, g):
        return N.pt(self.P(g))

    def line(self, gs, style):
        return N.polyline([self.P(g) for g in gs], style)


def split(gs):
    """Split a curve into its front (Im G1 <= 0) and back pieces."""
    out, cur, side = [], [gs[0]], gs[0][1] <= 1e-9
    for g in gs[1:]:
        s = g[1] <= 1e-9
        if s != side:
            out.append((side, np.array(cur)))
            cur, side = [cur[-1]], s
        cur.append(g)
    out.append((side, np.array(cur)))
    return out


def panel_a(b):
    pg = Page(shift=(0.0, 0.0))
    v = V3
    # the region of the plane Im G1 = 0 covered by the base (O L A2) and by the triangle L A2 A3
    t = np.linspace(0, PI / 2, 121)
    outline = ([G((x, 0, 0)) for x in np.linspace(0, PI / 2, 60)]                        # identity -> CNOT (O L)
               + [G((PI / 2, x, x)) for x in t]                                         # CNOT -> SWAP (L A3)
               + [G((PI / 2, PI / 2, PI / 2 - x)) for x in t]                           # SWAP -> iSWAP (A3 A2)
               + [G((PI / 2 - x, PI / 2 - x, 0)) for x in t])                           # iSWAP -> identity (A2 O)
    b.append("\\fill[ndGold!30, fill opacity=0.9] %s -- cycle;" % " -- ".join(pg.pt(g) for g in outline))
    # faint iso-curves on the two curved faces of each half
    faces = [(v["O"], v["L"], v["A3"]), (v["O"], v["A2"], v["A3"])]
    for a, c, d in faces:
        for s_ in (0.25, 0.5, 0.75):                                                     # images of segments parallel to c d
            gs = image(a + s_ * (c - a), a + s_ * (d - a), 80)
            b.append(pg.line(gs, "line width=0.3pt, ndInk!25, densely dashed"))         # behind (Im G1 > 0)
            b.append(pg.line(gs * [1, -1, 1], "line width=0.35pt, ndBlue!30"))           # in front (mirror image)
    # the curved faces in front, washed
    for a, c, d in faces:
        rim = np.concatenate([image(a, c, 60), image(c, d, 60), image(d, a, 60)]) * [1, -1, 1]
        b.append("\\fill[ndSky, fill opacity=0.35] %s -- cycle;" % " -- ".join(pg.pt(g) for g in rim))
    # the edges of the chamber, mapped; L A3 (not an edge) bounds the gold region
    edges = [("O", "A1"), ("O", "A2"), ("A1", "A2"), ("O", "A3"), ("A1", "A3"), ("A2", "A3"), ("L", "A3")]
    for e in edges:
        for front, piece in split(image(v[e[0]], v[e[1]])):
            style = "ink" if front else "dashedink"
            if e == ("L", "A3"):
                style = "line width=0.5pt, ndGold!80!ndInk"
            b.append(pg.line(piece, style))
    # the perfect entanglers L M N P Q A2, edges mapped (front solid, back dashed)
    pe = [("L", "Q"), ("Q", "P"), ("P", "L"), ("L", "M"), ("M", "N"), ("N", "L"), ("N", "P"),
          ("P", "A2"), ("A2", "N"), ("M", "A2"), ("A2", "Q")]
    for e in pe:
        a, c = v[e[0]], v[e[1]]
        for x in np.linspace(0, 1, 21):
            assert perfect_entangler_c((1 - x) * a + x * c)
        for front, piece in split(image(a, c)):
            style = "line width=1.1pt, ndBlue" if front else "line width=0.7pt, ndBlue!70, dash pattern=on 2pt off 1.5pt"
            b.append(pg.line(piece, style))
    # the gates
    gates = [("O", "$\\Id$", "west", (0.1, 0.0)),
             ("L", "CNOT", "east", (-0.12, 0.05)),
             ("A2", "iSWAP", "west", (0.12, -0.05)),
             ("A3", "SWAP", "east", (-0.1, 0.0)),
             ("P", "$\\sqrt{\\mathrm{SWAP}}$", "south", (0.0, 0.1)),
             ("N", "$\\sqrt{\\mathrm{SWAP}}^{\\,\\dagger}$", "west", (0.1, 0.0)),
             ("Q", "$\\sqrt{\\mathrm{iSWAP}}$", "west", (0.12, 0.08))]
    for name, lab, anchor, off in gates:
        p = pg.P(G(v[name]))
        col = "ndBlue" if perfect_entangler_c(v[name]) else "ndInk"
        b.append("\\node[dot, fill=%s, minimum size=3.6pt] at %s {};" % (col, N.pt(p)))
        b.append("\\node[font=\\scriptsize, anchor=%s, text=%s] at %s {%s};" % (anchor, col, N.pt(p + np.array(off)), lab))
    # a small tripod for the axes, not to scale
    o = pg.P([0, 0, 0]) + np.array([-3.7, -1.3])
    for e, lab, anc in [((1, 0, 0), "\\mathrm{Re}\\,G_1", "west"), ((0, 1, 0), "\\mathrm{Im}\\,G_1", "south east"),
                        ((0, 0, 1), "G_2", "south")]:
        d = PROJ @ np.array(e, dtype=float)
        d = 0.6 * d / np.linalg.norm(d)
        b.append("\\draw[faint, ->, >=stealth, ndInk!60] %s -- %s node[anchor=%s, font=\\scriptsize, text=ndInk] {$%s$};"
                 % (N.pt(o), N.pt(o + d), anc, lab))
    # notes
    top = pg.P(G(v["O"]))
    b.append("\\node[note, anchor=east] (nI) at %s {$O$ and $A_1$, the local\\\\ gates, meet at $(1,0,3)$};" % N.pt(top + np.array([-1.0, 0.35])))
    b.append("\\draw[pointer] (nI.east) to[bend left=15] %s;" % N.pt(top + np.array([-0.08, 0.04])))
    b.append("\\node[note, anchor=west, text=ndGold!55!ndInk] (nb) at %s {gold: the base and the\\\\ triangle $LA_2A_3$, both in\\\\ the plane $\\mathrm{Im}\\,G_1=0$};"
             % N.pt(pg.P([1.0, 0, 0.6]) + np.array([0.55, -0.45])))
    b.append("\\draw[pointer] (nb.west) to[bend left=12] %s;" % N.pt(pg.P(G((PI / 3, PI / 10, 0)))))
    b.append("\\node[note, anchor=west, text=ndBlue] (npe) at %s {blue: the perfect\\\\ entanglers $LMNPQA_2$};"
             % N.pt(pg.P([1.0, 0, -1.6]) + np.array([0.85, -0.3])))
    b.append("\\draw[pointer] (npe.west) to[bend left=12] %s;" % N.pt(pg.P(image(v["M"], v["N"])[60]) + np.array([0.05, 0.0])))
    b.append("\\node[note, anchor=east] (nc) at %s {flat faces become curved;\\\\ the half $c_1\\le\\pi/2$ lies\\\\ behind (dashed), its\\\\ mirror image in front};"
             % N.pt(pg.P([-1.0, 0, 0.9]) + np.array([-0.75, -0.35])))
    b.append("\\draw[pointer] (nc.east) to[bend left=10] %s;" % N.pt(pg.P(image(v["O"], v["A3"])[60]) + np.array([-0.05, 0.03])))
    bottom = min(pg.P(G(v["A3"]))[1], o[1]) - 0.75
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) the Weyl chamber in the space of $(\\mathrm{Re}\\,G_1,\\mathrm{Im}\\,G_1,G_2)$};"
             % (pg.P([0, 0, 0])[0], bottom))
    return bottom


def panel_b(b, x0, ybase, bottom):
    """The base triangle O A1 A2 and its folded image in the plane of (G1, G2)."""
    # left: the base in the (c1, c2) plane
    L = N.Plot((0, PI), (0, PI / 2), 3.3, 1.65, origin=(x0, ybase + 1.6))
    O_, L_, A1_, A2_ = L.P(0, 0), L.P(PI / 2, 0), L.P(PI, 0), L.P(PI / 2, PI / 2)
    b.append("\\fill[ndSky, fill opacity=0.8] %s -- %s -- %s -- cycle;" % (N.pt(O_), N.pt(L_), N.pt(A2_)))
    b.append("\\fill[ndRose, fill opacity=0.8] %s -- %s -- %s -- cycle;" % (N.pt(L_), N.pt(A1_), N.pt(A2_)))
    b.append("\\draw[ink] %s -- %s -- %s -- cycle;" % (N.pt(O_), N.pt(A1_), N.pt(A2_)))
    b.append("\\draw[line width=0.6pt, ndInk, densely dashed] %s -- %s;" % (N.pt(L_), N.pt(A2_)))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk, font=\\scriptsize] {$c_1$};" % (N.pt(O_), N.pt(L.P(PI * 1.1, 0))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, text=ndInk, font=\\scriptsize] {$c_2$};" % (N.pt(O_), N.pt(L.P(0, PI / 2 * 1.25))))
    for c2 in (PI / 8, PI / 4, 3 * PI / 8):                                               # lines of constant c2, in both halves
        xs = np.linspace(c2, PI - c2, 50)
        b.append(L.curve(xs, 0 * xs + c2, "line width=0.3pt, ndInk!35"))
    for name, p, anc in [("O", O_, "north"), ("L", L_, "north"), ("A_1", A1_, "north"), ("A_2", A2_, "south"),
                         ("Q", L.P(PI / 4, PI / 4), "south east"), ("M", L.P(3 * PI / 4, PI / 4), "south west")]:
        b.append("\\node[dot, minimum size=2.8pt] at %s {};" % N.pt(p))
        b.append("\\node[font=\\scriptsize, anchor=%s, inner sep=1.5pt] at %s {$%s$};" % (anc, N.pt(p), name))
    pairs = [((PI * 0.3, PI * 0.12), "ndRed"), ((PI * 0.42, PI * 0.3), "ndGreen")]       # (c1, c2, 0) and (pi - c1, c2, 0)
    for (c1, c2), col in pairs:
        for x in (c1, PI - c1):
            b.append("\\node[circle, draw=%s, fill=white, line width=0.9pt, inner sep=0pt, minimum size=4.2pt] at %s {};" % (col, L.pt(x, c2)))
    b.append("\\node[note, anchor=north] at %s {the base $c_3=0$ of the chamber,\\\\ split by $LA_2$ into two halves};"
             % N.pt(L.P(PI / 2, 0) + np.array([0, -0.35])))

    # right: the image in the (G1, G2) plane
    R = N.Plot((0, 1.0), (-1.0, 3.0), 2.3, 3.4, origin=(x0 + 4.9, ybase + 0.2))
    b += R.axes("$G_1$", "$G_2$", xticks=[(1, "$1$")], yticks=[(3, "$3$")], yzero=0.0)
    t = np.linspace(0, PI / 2, 121)
    curve = np.stack([np.cos(t) ** 4, 4 * np.cos(t) ** 4 - np.cos(2 * t) ** 2], 1)          # image of O A2 and of A1 A2
    region = [R.P(*q) for q in curve] + [R.P(0, -1), R.P(0, 1), R.P(1, 3)]
    b.append("\\fill[ndGold!30] %s -- cycle;" % " -- ".join(N.pt(p) for p in region))
    b.append(N.polyline([R.P(*q) for q in curve], "ink"))
    b.append("\\draw[ink] %s -- %s;" % (R.pt(1, 3), R.pt(0, 1)))                              # O L and A1 L: G2 = 2 G1 + 1
    b.append("\\draw[ink] %s -- %s;" % (R.pt(0, 1), R.pt(0, -1)))                             # L A2: G1 = 0
    for c2 in (PI / 8, PI / 4, 3 * PI / 8):
        xs = np.linspace(c2, PI / 2, 60)
        gs = np.array([G((x, c2, 0)) for x in xs])
        assert np.allclose(gs, [G((PI - x, c2, 0)) for x in xs])
        b.append(R.curve(gs[:, 0], gs[:, 2], "line width=0.3pt, ndInk!35"))
    for (c1, c2), col in pairs:
        g = G((c1, c2, 0))
        assert np.allclose(g, G((PI - c1, c2, 0)))
        b.append("\\node[circle, draw=%s, fill=white, line width=0.9pt, inner sep=0pt, minimum size=4.2pt] at %s {};" % (col, R.pt(g[0], g[2])))
    for (g1, g2), lab, anc in [((1, 3), "$\\Id$", "west"), ((0, 1), "CNOT", "east"), ((0, -1), "iSWAP", "east"),
                               ((0.25, 1), "$\\sqrt{\\mathrm{iSWAP}}$", "west")]:
        b.append("\\node[dot, minimum size=3pt] at %s {};" % R.pt(g1, g2))
        b.append("\\node[font=\\scriptsize, anchor=%s, inner sep=2.5pt] at %s {%s};" % (anc, R.pt(g1, g2), lab))
    b.append("\\draw[->, >=stealth, line width=0.7pt, ndInk!70] %s to[bend left=25] %s;"
             % (N.pt(A2_ + np.array([0.35, 0.15])), N.pt(R.P(0.42, 1.75) + np.array([0.08, 0.06]))))
    b.append("\\node[note, anchor=north] at %s {both halves map onto\\\\ one region: each ring\\\\ comes from two points};"
             % N.pt(R.P(0.5, -1.0) + np.array([0.0, -0.35])))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) the fold of the base};" % (x0 + 3.6, bottom))


def main():
    hi, phi, lo, plo = check_figure()
    print("Im G1 ranges over [%.4f, %.4f] on the chamber; the maximum is at c/pi = %s (P), the minimum at %s (N)"
          % (lo, hi, np.round(phi / PI, 3), np.round(plo / PI, 3)))
    vals = [np.round(G(V3[k]), 3) for k in ("O", "L", "A2", "A3", "P", "N", "Q")] + [np.round(G((PI / 2, PI / 4, 0)), 3)]
    print("identity (O, A1): %s; CNOT (L): %s; iSWAP (A2): %s; SWAP (A3): %s; P: %s; N: %s; Q = M: %s; B [pi/2, pi/4, 0]: %s"
          % tuple(vals))
    b = []
    bottom = panel_a(b)
    panel_b(b, 6.4, bottom + 1.6, bottom)
    N.write("ch07_makhlin", b)
    print("written figures/ch07_makhlin.tex")


if __name__ == "__main__":
    main()
