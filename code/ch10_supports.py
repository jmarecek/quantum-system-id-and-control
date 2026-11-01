"""ch10_supports.py -- locality on three qubits, pictured.

A Pauli string on n = 3 qubits acts non-trivially on its support; a Hermitian
operator is k-local if it is a combination of strings with supports of at most
k qubits.  The 2-local directions d = span{-iP : |supp P| <= 2} of su(8) number
9 one-qubit + 27 two-qubit = 36, and the 27 three-qubit strings make up the
rest of the 63 dimensions of su(8).

(a) Supports.  The bracket of a 1-local and a 2-local string lies inside the
    support of the 2-local one, e.g. [-iZII, -iXXI] = 2(-iYXI).  The bracket of
    two 2-local strings that overlap on one qubit is 3-local,
    [-iZZI, -iIXX] = 2(-iZYX): this is what makes d bracket-generating, and it
    is also why a 2-local drift does not normalise d.
(b) Dimensions: d has dimension 36, and d + [d, d] = su(8), of dimension 63.
(c) How many of the 27 three-qubit directions the bracket [-iH_d, d] reaches
    (the rank of its component orthogonal to d): none for a 1-local drift (a
    generic combination of the nine one-qubit strings), 12 for H_d = Z1 Z2, and
    24 for a generic 2-local drift.  The condition [A_d, d] in d of the geometric
    interaction picture holds for the first and fails for the others.
Writes figures/ch10_supports.tex.
"""
import itertools
import numpy as np
import needham as N

I2 = np.eye(2, dtype=complex)
PAULI = {"I": I2, "X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0]).astype(complex)}
STRINGS = ["".join(s) for s in itertools.product("IXYZ", repeat=3) if "".join(s) != "III"]


def ps(s):
    M = np.array([[1.0 + 0j]])
    for c in s:
        M = np.kron(M, PAULI[c])
    return M


def weight(s):
    return sum(c != "I" for c in s)


D = [s for s in STRINGS if weight(s) <= 2]
PERP = [s for s in STRINGS if weight(s) == 3]


def vec(M):
    return np.concatenate([M.real.ravel(), M.imag.ravel()])


def comm(A, B):
    return A @ B - B @ A


def leak_rank(H):
    """Rank of the component of [-iH, d] orthogonal to d (coefficients along -iP, P three-local)."""
    A = -1j * H
    rows = [[np.real(np.trace(comm(A, -1j * ps(s)) @ (1j * ps(p)))) / 8 for p in PERP] for s in D]
    return np.linalg.matrix_rank(np.array(rows), 1e-9)


def checks():
    assert len(STRINGS) == 63 and len(D) == 36 and len(PERP) == 27
    assert sum(weight(s) == 1 for s in D) == 9
    # the two brackets of panel (a)
    assert np.allclose(comm(-1j * ps("ZII"), -1j * ps("XXI")), 2 * (-1j) * ps("YXI"))
    assert np.allclose(comm(-1j * ps("ZZI"), -1j * ps("IXX")), 2 * (-1j) * ps("ZYX"))
    # dimensions
    Bd = [-1j * ps(s) for s in D]
    assert np.linalg.matrix_rank(np.array([vec(M) for M in Bd])) == 36
    assert np.linalg.matrix_rank(np.array([vec(M) for M in Bd + [comm(a, b) for a in Bd for b in Bd]])) == 63
    # a 1-local drift normalises d; 2-local drifts do not
    rng = np.random.default_rng(0)
    H1 = sum(rng.normal() * ps(s) for s in STRINGS if weight(s) == 1)
    H2 = sum(rng.normal() * ps(s) for s in STRINGS if weight(s) <= 2)
    ranks = (leak_rank(H1), leak_rank(ps("ZZI")), leak_rank(H2))
    assert ranks == (0, 12, 24)
    return ranks


def string(b, x, y, s, fill, label=None, r=0.19, dx=0.52):
    """Three qubit sites at (x + k dx, y), the letters of s inside, the support filled."""
    pts = [(x + k * dx, y) for k in range(3)]
    sup = [k for k, c in enumerate(s) if c != "I"]
    if sup:
        a, c = pts[min(sup)], pts[max(sup)]
        b.append("\\draw[line width=7pt, %s!35, line cap=round] (%.3f,%.3f) -- (%.3f,%.3f);" % (fill, a[0], a[1], c[0] + 1e-3, c[1]))
    for k, (px, py) in enumerate(pts):
        if s[k] == "I":
            b.append("\\draw[faint, fill=white] (%.3f,%.3f) circle (%.3f);" % (px, py, r))
            b.append("\\node[font=\\scriptsize, text=ndInk!40] at (%.3f,%.3f) {$I$};" % (px, py))
        else:
            b.append("\\draw[ink, fill=%s!25] (%.3f,%.3f) circle (%.3f);" % (fill, px, py, r))
            b.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {$%s$};" % (px, py, s[k]))
    if label:
        b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {%s};" % (x - 0.3, y, label))


def main():
    ranks = checks()
    b = []
    # ---- (a) supports
    xa, xb = 1.55, 5.35
    for x in (xa, xb):
        for k in range(3):
            b.append("\\node[font=\\tiny, text=ndInk!60] at (%.3f,%.3f) {%d};" % (x + 0.52 * k, 4.55, k + 1))
    b.append("\\node[font=\\scriptsize, text=ndInk!70] at (%.3f,%.3f) {qubit};" % (xa - 0.75, 4.55))
    string(b, xa, 4.05, "ZII", "ndGreen", "$-\\imath Z_1$")
    string(b, xa, 3.4, "XXI", "ndBlue", "$-\\imath X_1X_2$")
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (xa - 0.25, 3.0, xa + 1.29, 3.0))
    string(b, xa, 2.55, "YXI", "ndBlue", "$\\tfrac12[\\,\\cdot\\,,\\cdot\\,]$")
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {a 1-local string with a\\\\ 2-local one: the bracket\\\\ stays inside the support};" % (xa + 0.52, 2.15))
    string(b, xb, 4.05, "ZZI", "ndBlue", "$-\\imath Z_1Z_2$")
    string(b, xb, 3.4, "IXX", "ndBlue", "$-\\imath X_2X_3$")
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (xb - 0.25, 3.0, xb + 1.29, 3.0))
    string(b, xb, 2.55, "ZYX", "ndRed", "$\\tfrac12[\\,\\cdot\\,,\\cdot\\,]$")
    b.append("\\node[note, anchor=north, text=ndRed] at (%.3f,%.3f) {two 2-local strings\\\\ sharing a qubit: the\\\\ bracket is 3-local};" % (xb + 0.52, 2.15))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(a) supports and brackets};" % ((xa + xb) / 2 + 0.3, 0.55))

    # ---- (b) dimensions
    pl = N.Plot((0, 2.2), (0, 70), 2.6, 3.6, origin=(8.7, 1.0))
    b += pl.axes(ylabel="{\\scriptsize dimension}", yticks=[(36, "$36$"), (63, "$63$")])
    for x, parts, lab in [(0.6, [(9, "ndGreen"), (27, "ndBlue")], "$\\mathfrak d$"),
                          (1.6, [(9, "ndGreen"), (27, "ndBlue"), (27, "ndRed")], "$\\mathfrak d+[\\mathfrak d,\\mathfrak d]$")]:
        y = 0
        for h, col in parts:
            b.append("\\filldraw[fill=%s!35, draw=%s!70!ndInk, line width=0.4pt] %s rectangle %s;" % (col, col, pl.pt(x - 0.3, y), pl.pt(x + 0.3, y + h)))
            y += h
        b.append("\\node[font=\\scriptsize, anchor=north] at %s {%s};" % (pl.pt(x, -1.5), lab))
    b.append("\\draw[dashedink] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$\\liesu[8]$};" % (pl.pt(0, 63), pl.pt(2.2, 63)))
    for y, txt, col in [(4.5, "1-qubit, 9", "ndGreen"), (22.5, "2-qubit, 27", "ndBlue"), (49.5, "3-qubit, 27", "ndRed")]:
        b.append("\\node[font=\\tiny, text=%s!70!ndInk] at %s {%s};" % (col, pl.pt(1.6, y), txt))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) one bracket fills $\\liesu[8]$};" % pl.pt(1.1, -12))

    # ---- (c) leakage of the drift
    pc = N.Plot((0, 3.3), (0, 30), 3.4, 3.6, origin=(12.75, 1.0))
    b += pc.axes(ylabel="{\\scriptsize 3-qubit directions reached}", yticks=[(12, "$12$"), (24, "$24$"), (27, "")])
    b.append("\\draw[dashedink] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$27$};" % (pc.pt(0, 27), pc.pt(3.3, 27)))
    for x, r, lab, col in [(0.55, ranks[0], "1-local", "ndGreen"), (1.65, ranks[1], "$Z_1Z_2$", "ndRed"),
                           (2.75, ranks[2], "2-local", "ndRed")]:
        if r:
            b.append("\\filldraw[fill=%s!35, draw=%s!70!ndInk, line width=0.4pt] %s rectangle %s;" % (col, col, pc.pt(x - 0.32, 0), pc.pt(x + 0.32, r)))
        else:
            b.append("\\draw[line width=1.2pt, %s] %s -- %s;" % (col, pc.pt(x - 0.32, 0.15), pc.pt(x + 0.32, 0.15)))
            b.append("\\node[font=\\scriptsize, anchor=south, text=ndGreen!70!ndInk] at %s {$0$};" % pc.pt(x, 0.6))
        b.append("\\node[font=\\scriptsize, anchor=north] at %s {%s};" % (pc.pt(x, -1.5), lab))
    b.append("\\node[note, anchor=north] at %s {drift $H_d$; bars: rank of\\\\ $[-\\imath H_d,\\mathfrak d]$ outside $\\mathfrak d$};" % pc.pt(1.65, -5.5))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) which drifts normalise $\\mathfrak d$};" % pc.pt(1.65, -12))
    N.write("ch10_supports", b)
    return ranks


if __name__ == "__main__":
    print("leak ranks (1-local, Z1Z2, generic 2-local):", main())
    print("written figures/ch10_supports.tex")
