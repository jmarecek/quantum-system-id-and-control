"""ch07_tn_notation.py -- a dictionary of the graphical calculus of tensors.

A tensor is drawn as a shape with one leg for each of its indices: a scalar
has no legs, a vector one, a matrix two, and the building block of a matrix
product state, A_{alpha s beta}, three (two bonds and a physical leg, drawn
upwards).  Joining two legs sums over the shared index, so that the matrix-
vector product, the matrix product, the trace (a loop) and the inner product
<phi|psi> (a bra over a ket) are all contractions.  Two shapes side by side
with no edge are a tensor product, and bundling two legs of dimensions d1, d2
into one leg of dimension d1 d2 is a reshaping.  Each identity drawn is
checked numerically with numpy.einsum.
Writes figures/ch07_tn_notation.tex.
"""
import numpy as np
import needham as N

GOLD = "draw=ndGold!70!ndInk, fill=ndGold!25, rounded corners=3pt, line width=0.9pt, inner sep=1pt, font=\\scriptsize"
KET = "circle, draw=ndBlue, fill=ndSky, line width=0.9pt, minimum size=0.52cm, inner sep=0pt, font=\\scriptsize"
BRA = "circle, draw=ndBlue, fill=ndSky!40, line width=0.9pt, minimum size=0.52cm, inner sep=0pt, font=\\scriptsize"
LEG = 0.5                                      # length of a free leg


def checks():
    rng = np.random.default_rng(7)
    c = lambda *s: rng.normal(size=s) + 1j * rng.normal(size=s)
    M, A, B, psi, phi = c(3, 3), c(3, 4), c(4, 5), c(3), c(3)
    assert np.allclose(np.einsum("ij,j->i", M, psi), M @ psi)                  # M psi
    assert np.allclose(np.einsum("ij,jk->ik", A, B), A @ B)                    # A B
    assert np.isclose(np.einsum("ii->", M), np.trace(M))                       # Tr M: a loop
    assert np.isclose(np.einsum("s,s->", phi.conj(), psi), np.vdot(phi, psi))  # <phi|psi>
    assert np.allclose(np.einsum("i,j->ij", psi, phi).reshape(-1), np.kron(psi, phi))   # no edge
    T, X = c(3, 2, 4), c(2, 4)
    assert np.allclose(np.einsum("ijk,jk->i", T, X), T.reshape(3, 8) @ X.reshape(8))   # bundled legs
    S = c(2, 3, 2)                                                             # A_{alpha s beta}
    assert S.ndim == 3


def node(style, name, xy, text, extra=""):
    return "\\node[%s%s] (%s) at (%.3f,%.3f) {%s};" % (style, extra, name, xy[0], xy[1], text)


def leg(frm, d, label=None, pos="above", width="0.9pt"):
    """A free leg from the anchor `frm` in the direction d = (dx, dy), with a label at its end."""
    lab = "" if label is None else " node[%s, font=\\scriptsize, inner sep=1.5pt] {%s}" % (pos, label)
    return "\\draw[ndInk, line width=%s] (%s) -- ++(%.3f,%.3f)%s;" % (width, frm, d[0], d[1], lab)


def main():
    checks()
    b = []
    # ---- (a) objects: a shape with one leg per index
    y = 4.55
    b.append(node(GOLD + ", minimum size=0.5cm", "c", (0.6, y), "$c$"))
    b.append(node(GOLD + ", minimum size=0.5cm", "v", (2.7, y), "$\\psi$"))
    b.append(leg("v.west", (-LEG, 0), "$i$", "left"))
    b.append(node(GOLD + ", minimum size=0.5cm", "m", (5.2, y), "$M$"))
    b.append(leg("m.west", (-LEG, 0), "$i$", "left"))
    b.append(leg("m.east", (LEG, 0), "$j$", "right"))
    b.append(node(KET, "a", (8.2, y), "$A$"))
    b.append(leg("a.west", (-LEG, 0), "$\\alpha$", "left"))
    b.append(leg("a.east", (LEG, 0), "$\\beta$", "right"))
    b.append(leg("a.north", (0, LEG), "$s$", "above"))
    for x, txt in [(0.6, "scalar $c$"), (2.45, "vector $\\psi_i$"), (5.2, "matrix $M_{ij}$"), (8.2, "order three, $A_{\\alpha s\\beta}$")]:
        b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {%s};" % (x, y - 0.5, txt))
    b.append("\\node[note, anchor=south] (nleg) at (5.2,%.3f) {a leg is an index};" % (y + 0.55))
    b.append("\\draw[pointer] (nleg.east) to[bend left=15] (5.62,%.3f);" % (y + 0.08))
    b.append("\\node[font=\\small] at (4.6,3.25) {(a) a tensor is a shape with one leg for each index};")

    # ---- (b) contraction: joining two legs sums over the index
    y = 1.55
    # M psi
    b.append(node(GOLD + ", minimum size=0.5cm", "m1", (0.75, y), "$M$"))
    b.append(node(GOLD + ", minimum size=0.5cm", "v1", (1.75, y), "$\\psi$"))
    b.append(leg("m1.west", (-0.45, 0), "$i$", "left"))
    b.append("\\draw[ndInk, line width=0.9pt] (m1.east) -- (v1.west);")
    b.append("\\node[font=\\scriptsize, anchor=north] at (1.15,%.3f) {$\\sum_jM_{ij}\\psi_j$};" % (y - 0.45))
    # A B
    b.append(node(GOLD + ", minimum size=0.5cm", "A", (3.45, y), "$A$"))
    b.append(node(GOLD + ", minimum size=0.5cm", "B", (4.45, y), "$B$"))
    b.append(leg("A.west", (-0.45, 0), "$i$", "left"))
    b.append(leg("B.east", (0.45, 0), "$k$", "right"))
    b.append("\\draw[ndInk, line width=0.9pt] (A.east) -- (B.west);")
    b.append("\\node[font=\\scriptsize, anchor=north] at (3.95,%.3f) {$\\sum_jA_{ij}B_{jk}$};" % (y - 0.45))
    # Tr M: a loop
    b.append(node(GOLD + ", minimum size=0.5cm", "m2", (6.25, y - 0.12), "$M$"))
    b.append("\\draw[ndInk, line width=0.9pt, rounded corners=5pt] (m2.east) -- ++(0.3,0) -- ++(0,0.62) -- ($(m2.west)+(-0.3,0.62)$) -- ++(0,-0.62) -- (m2.west);")
    b.append("\\node[font=\\scriptsize, anchor=north] at (6.25,%.3f) {$\\Tr M=\\sum_iM_{ii}$};" % (y - 0.45))
    # <phi|psi>: a bra over a ket
    b.append(node(KET, "k", (7.95, y - 0.42), "$\\psi$"))
    b.append(node(BRA, "br", (7.95, y + 0.42), "$\\bar\\phi$"))
    b.append("\\draw[ndInk, line width=0.9pt] (k.north) -- (br.south);")
    b.append("\\node[font=\\scriptsize, anchor=west] at (8.3,%.3f) {$\\langle\\phi|\\psi\\rangle=\\sum_s\\bar\\phi_s\\psi_s$};" % y)
    b.append("\\node[note, anchor=south] (njoin) at (3.0,2.4) {joining two legs sums over the index};")
    b.append("\\draw[pointer] (njoin.south) to[bend left=10] (3.93,%.3f);" % (y + 0.1))
    b.append("\\node[font=\\small] at (5.0,0.12) {(b) contraction: products, traces and inner products};")

    # ---- (c) no edge, and bundled legs; a column of its own on the right
    xc = 13.15
    y = 4.45
    b.append(node(KET, "p1", (xc - 0.75, y), "$\\psi$"))
    b.append(node(KET, "p2", (xc + 0.05, y), "$\\phi$"))
    b.append(leg("p1.north", (0, 0.42), "$i$", "above"))
    b.append(leg("p2.north", (0, 0.42), "$j$", "above"))
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$(\\psi\\otimes\\phi)_{ij}=\\psi_i\\phi_j$};" % (xc - 0.35, y - 0.42))
    b.append("\\node[note, anchor=west] at (%.3f,%.3f) {no edge:\\\\ a tensor\\\\ product};" % (xc + 0.55, y + 0.05))
    y = 1.75
    b.append(node(GOLD + ", minimum width=0.5cm, minimum height=0.62cm", "T1", (xc - 1.55, y), "$T$"))
    b.append("\\draw[ndInk, line width=0.9pt] ($(T1.east)+(0,0.16)$) -- ++(0.5,0) node[right, font=\\scriptsize, inner sep=1pt] {$d_1$};")
    b.append("\\draw[ndInk, line width=0.9pt] ($(T1.east)+(0,-0.16)$) -- ++(0.5,0) node[right, font=\\scriptsize, inner sep=1pt] {$d_2$};")
    b.append("\\draw[->, >=stealth, line width=0.6pt, ndInk!70] (%.3f,%.3f) -- (%.3f,%.3f);" % (xc - 0.28, y, xc + 0.17, y))
    b.append(node(GOLD + ", minimum width=0.5cm, minimum height=0.62cm", "T2", (xc + 0.68, y), "$T$"))
    b.append("\\draw[ndInk, line width=2.2pt] (T2.east) -- ++(0.5,0) node[right, font=\\scriptsize, inner sep=1.5pt] {$d_1d_2$};")
    b.append("\\node[font=\\scriptsize, anchor=north, align=center] at (%.3f,%.3f) {two legs bundled into one:\\\\ a reshaping};" % (xc - 0.2, y - 0.45))
    b.append("\\node[font=\\small] at (%.3f,0.12) {(c) tensor product, reshaping};" % (xc - 0.2))
    N.write("ch07_tn_notation", b)
    print("written figures/ch07_tn_notation.tex")


if __name__ == "__main__":
    main()
