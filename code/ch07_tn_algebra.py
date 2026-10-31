"""ch07_tn_algebra.py -- the arithmetic of matrix product operators, and the growth of the bond dimension.

(a) A sum of two MPOs is an MPO whose cores are the direct sums W (+) V
    (block diagonal in the middle, side by side at the ends): the bond
    dimensions add.
(b) A product is an MPO whose cores are the products of the stacked cores,
    with the two bonds fused: the bond dimensions multiply.
(c) Following Balada Gaggioli and Marecek (PRA 112, 062612, 2025), Table I,
    with H_0 and H_c both of bond dimension 3: the truncated Magnus generator
    Omega^(n) is a sum of 2, 2, 6, 12 words in {H_0, H_c} of lengths 1, 2, 3, 4,
    a word of length l having bond dimension 3^l, so that
        b_n = 6, 24, 186, 1158   for n = 1, 2, 3, 4,
    and the Chebyshev approximation of exp(Omega^(n)) of degree p, a polynomial
    sum_{k<=p} c_k Omega^k, has bond dimension sum_{k=0}^p b_n^k.  A Trotter
    step of the controlled Ising chain is a product of MPOs of bond dimensions
    1, 2, 2, so that K steps have bond dimension 4^K.
The sum and product rules are checked on the MPOs of ch07_tn_automaton.py
against dense arithmetic, and the counts against the Table of the paper.
Writes figures/ch07_tn_algebra.tex.
"""
import numpy as np
import needham as N
import ch07_tn_automaton as M

TABLE = {1: [7, 43, 259, 1.55e3], 2: [25, 601, 1.44e4, 3.46e5],          # the paper's Table I,
         3: [187, 3.47e4, 6.46e6, 1.20e9], 4: [1.15e3, 1.34e6, 1.55e9, 1.79e12]}   # three significant figures
WORDS = {1: 2, 2: 2, 3: 6, 4: 12}                                     # words of each length in Omega^(4)


def magnus_bond(n, r=3):
    return sum(WORDS[l] * r ** l for l in range(1, n + 1))


def chebyshev_bond(n, p, r=3):
    b = magnus_bond(n, r)
    return sum(b ** k for k in range(p + 1))


def truncate3(x):
    """Truncate to three significant figures, as the Table does."""
    e = int(np.floor(np.log10(x)))
    return np.floor(x / 10 ** (e - 2)) * 10 ** (e - 2)


def checks():
    n = 5
    H0, Hc = M.ising_mpo(n), M.control_mpo(n)
    D0, Dc = M.dense_chain(n)
    S = M.mpo_sum(H0, Hc)
    P = M.mpo_product(H0, Hc)
    assert M.bond_dims(S) == [5] * (n - 1) and np.allclose(M.mpo_to_dense(S), D0 + Dc)
    assert M.bond_dims(P) == [6] * (n - 1) and np.allclose(M.mpo_to_dense(P), D0 @ Dc)
    C = M.mpo_sum(P, M.mpo_scale(M.mpo_product(Hc, H0), -1.0))       # the commutator [H_0, H_c]
    assert M.bond_dims(C) == [12] * (n - 1) and np.allclose(M.mpo_to_dense(C), D0 @ Dc - Dc @ D0)
    # a Chebyshev-type polynomial of degree 2 in Omega = a H_0 + b H_c: bond 1 + 5 + 25
    Om = M.mpo_sum(M.mpo_scale(H0, -0.1j), M.mpo_scale(Hc, -0.05j))
    one = [np.eye(2, dtype=complex)[None, None] for _ in range(n)]
    poly = M.mpo_sum(M.mpo_sum(one, Om), M.mpo_scale(M.mpo_product(Om, Om), 0.5))
    Dom = -0.1j * D0 - 0.05j * Dc
    assert M.bond_dims(poly) == [31] * (n - 1)
    assert np.allclose(M.mpo_to_dense(poly), np.eye(2 ** n) + Dom + 0.5 * Dom @ Dom)
    # the paper's accounting
    assert [magnus_bond(k) for k in (1, 2, 3, 4)] == [6, 24, 186, 1158]
    for k in (1, 2, 3, 4):
        for p in (1, 2, 3, 4):
            v = chebyshev_bond(k, p)
            assert abs(truncate3(v) - TABLE[k][p - 1]) < 1e-9 * TABLE[k][p - 1], (k, p, v)
    # a Trotter step: bonds 1, 2, 2 multiply to 4
    assert 1 * 2 * 2 == 4
    print("sum of H_0 (r=3) and H_c (r=2): r = 5; product: r = 6; commutator: r = 12; I + Om + Om^2/2: r = 31")
    print("Magnus bonds b_n =", [magnus_bond(k) for k in (1, 2, 3, 4)])
    for k in (1, 2, 3, 4):
        print("  n = %d: Chebyshev p = 1..4 ->" % k, [chebyshev_bond(k, p) for p in (1, 2, 3, 4)])
    print("Trotter 4^K, K = 1..8:", [4 ** K for K in range(1, 9)])


def core(name, pos, label, fill="ndRose", draw="ndRed", size=0.58, legs=True, up=0.32, down=0.32):
    out = ["\\node[rectangle, rounded corners=3pt, minimum size=%.2fcm, fill=%s, draw=%s, line width=0.9pt, "
           "font=\\scriptsize, inner sep=1pt] (%s) at %s {%s};" % (size, fill, draw, name, N.pt(pos), label)]
    if legs:
        if up:
            out.append("\\draw[ink] (%s.north) -- ++(0,%.2f);" % (name, up))
        if down:
            out.append("\\draw[ink] (%s.south) -- ++(0,-%.2f);" % (name, down))
    return out


def chain(prefix, x0, y, dx, n, label, bond, fill="ndRose", draw="ndRed", width="0.9pt", up=0.32, down=0.32, size=0.58):
    out = []
    for k in range(n):
        out += core("%s%d" % (prefix, k), (x0 + k * dx, y), label, fill, draw, size=size, up=up, down=down)
    for k in range(n - 1):
        out.append("\\draw[ndInk, line width=%s] (%s%d.east) -- node[above, font=\\scriptsize] {%s} (%s%d.west);"
                   % (width, prefix, k, bond, prefix, k + 1))
    return out


def main():
    checks()
    b = []
    dx = 1.0
    # ---- (a) the sum: bonds add
    b += chain("a", 0.6, 6.9, dx, 3, "$W$", "$r$")
    b += chain("v", 0.6, 5.5, dx, 3, "$V$", "$r'$")
    b.append("\\node[font=\\large] at (-0.15,6.2) {$+$};")
    b.append("\\draw[->, >=stealth, line width=0.9pt] (3.05,6.2) -- (3.75,6.2);")
    b += chain("s", 4.45, 6.2, 1.95, 3, "$W\\!\\oplus\\!V$", "$r+r'$", size=0.95, width="1.6pt")
    bx, by, h = 5.95, 4.15, 0.8                                   # the block-diagonal middle core
    b.append("\\fill[ndRose] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (bx, by + h / 2, bx + h / 2, by + h))
    b.append("\\fill[ndSky] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (bx + h / 2, by, bx + h, by + h / 2))
    b.append("\\draw[ink] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (bx, by, bx + h, by + h))
    b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f) (%.2f,%.2f) -- (%.2f,%.2f);" % (bx + h / 2, by, bx + h / 2, by + h, bx, by + h / 2, bx + h, by + h / 2))
    b.append("\\node[font=\\scriptsize] at (%.2f,%.2f) {$W$};" % (bx + h / 4, by + 3 * h / 4))
    b.append("\\node[font=\\scriptsize] at (%.2f,%.2f) {$V$};" % (bx + 3 * h / 4, by + h / 4))
    b.append("\\node[font=\\scriptsize, text=ndInk!60] at (%.2f,%.2f) {$0$};" % (bx + 3 * h / 4, by + 3 * h / 4))
    b.append("\\node[font=\\scriptsize, text=ndInk!60] at (%.2f,%.2f) {$0$};" % (bx + h / 4, by + h / 4))
    b.append("\\draw[pointer] (s1.south) ++(0.2,-0.3) -- (%.2f,%.2f);" % (bx + h / 2 + 0.2, by + h + 0.04))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {a middle core is block\\\\ diagonal; the end cores\\\\ stand side by side};" % (bx + h + 0.15, by + h / 2))
    b.append("\\node[font=\\small] at (4.3,3.55) {(a) a sum: the bonds add};")
    # ---- (b) the product: bonds multiply
    b += chain("p", 0.6, 2.75, dx, 3, "$W$", "$r$", down=0.0)
    b += chain("q", 0.6, 1.75, dx, 3, "$V$", "$r'$", up=0.0)
    for k in range(3):
        b.append("\\draw[ink] (p%d.south) -- (q%d.north);" % (k, k))
    b.append("\\node[note, anchor=north] at (1.6,1.05) {$H_WH_V$: $W$ stacked\\\\ on $V$, which acts first};")
    b.append("\\draw[->, >=stealth, line width=0.9pt] (3.05,2.25) -- (3.75,2.25);")
    b += chain("f", 4.45, 2.25, 1.95, 3, "$WV$", "$rr'$", size=0.95, width="2.6pt")
    b.append("\\node[note, anchor=north] at (6.4,1.3) {the two bonds fuse into\\\\ one of dimension $rr'$};")
    b.append("\\node[font=\\small] at (4.3,-0.15) {(b) a product: the bonds multiply};")
    # ---- (c) growth of the bond dimension
    xc = 11.2
    P = N.Plot((0.6, 8.4), (0.0, 13.0), 5.4, 5.4, origin=(xc, 1.1))
    b += P.axes("$p$, $K$", "bond dimension",
                xticks=[(k, "$%d$" % k) for k in range(1, 9)],
                yticks=[(e, "$10^{%d}$" % e) for e in (0, 4, 8, 12)])
    b.append("\\fill[ndGold!30] %s rectangle %s;" % (P.pt(0.6, 3.7), P.pt(8.4, 4.3)))
    b.append("\\node[note, anchor=north west, text=ndGold!55!ndInk] at %s {about $10^4$:\\\\ a laptop};" % P.pt(6.35, 3.6))
    cols = {1: "ndBlue", 2: "ndGreen", 3: "ndGold!80!ndRed", 4: "ndRed"}
    for k in (1, 2, 3, 4):
        ps = np.arange(1, 5)
        ys = [np.log10(chebyshev_bond(k, p)) for p in ps]
        b.append(P.curve(ps, ys, "line width=1.2pt, %s" % cols[k]))
        for p, yv in zip(ps, ys):
            b.append("\\node[dot, fill=%s, minimum size=3pt] at %s {};" % (cols[k], P.pt(p, yv)))
        anchor = "south west" if k == 1 else "west"
        b.append("\\node[font=\\scriptsize, text=%s, anchor=%s] at %s {$n=%d$};" % (cols[k], anchor, N.pt(P.P(4, ys[-1]) + np.array([0.06, 0.02])), k))
    Ks = np.arange(1, 9)
    b.append(P.curve(Ks, np.log10(4.0 ** Ks), "dashedink, line width=1.0pt"))
    for K in Ks:
        b.append("\\node[dot, fill=ndInk!60, minimum size=2.6pt] at %s {};" % P.pt(K, np.log10(4.0 ** K)))
    b.append("\\node[font=\\scriptsize, text=ndInk!80, anchor=south east] at %s {Trotter, $4^K$};" % P.pt(8.1, 5.1))
    b.append("\\node[note, anchor=north west] at %s {Magnus order $n$,\\\\ Chebyshev degree $p$};" % P.pt(5.3, 12.2))
    b.append("\\node[font=\\small] at %s {(c) the growth of the bond dimension};" % N.pt(np.array([xc + 2.7, -0.15])))
    N.write("ch07_tn_algebra", b)
    print("written figures/ch07_tn_algebra.tex")


if __name__ == "__main__":
    main()
