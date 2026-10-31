"""ch07_tn_cut.py -- a cut through a matrix product state is a rank.

(a) A matrix product state of N = 6 qubits, cut between sites 3 and 4.  Bundling
    the legs on each side of the cut turns the state into an 8 x 8 matrix, which
    the network writes as a product L R through the bond: its rank is at most
    D_3.  The singular value decomposition U Lambda V^dagger of the same matrix
    gives the Schmidt coefficients across the cut, drawn for the ground state
    of the controlled Ising chain H = sum Z_i Z_{i+1} + u sum X_i at u = 1.
(b) The gauge: X X^{-1} = 1 inserted on a bond leaves the state unchanged and
    replaces the cores by A[3] X and X^{-1} A[4], as the similarity
    transformation of a realisation does in Chapter 4.
(c) The same picture in time and in space: the Hankel matrix of Chapter 5
    factorises through the state space, H = O C, and its rank is the least
    order of a realisation; the unfolding factorises through the bond,
    psi = L R, and its rank is the least bond dimension.
The checks: random matrix product states have unfoldings of rank equal to their
bond dimensions, TT-SVD recovers them, the gauge leaves the state unchanged,
and the squared Schmidt coefficients are the eigenvalues of the reduced state.
Writes figures/ch07_tn_cut.tex.
"""
import numpy as np
import needham as N
from ch07_tn_entropy import unfold, schmidt, tt_svd, mps_vector, bonds, random_mps, ground_state

R_CORE = 0.26


def core(b, x, y, label):
    """An MPS core (circle, its label inside) with its physical leg pointing up."""
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (x, y + R_CORE, x, y + 0.62))
    b.append("\\node[circle, minimum size=%.2fcm, inner sep=0pt, fill=ndSky, draw=ndBlue, line width=0.9pt, font=\\scriptsize] at (%.3f,%.3f) {%s};"
             % (2 * R_CORE, x, y, label))


def box(b, x, y, label, w=0.78, h=0.5, fill="ndGold!25", draw="ndGold!70!ndInk"):
    b.append("\\node[rounded corners=2pt, minimum width=%.2fcm, minimum height=%.2fcm, inner sep=1pt, fill=%s, draw=%s, line width=0.9pt, font=\\scriptsize] at (%.3f,%.3f) {%s};"
             % (w, h, fill, draw, x, y, label))


def gauge(b, x, y, label):
    b.append("\\node[minimum size=0.36cm, inner sep=0pt, fill=ndGreen!20, draw=ndGreen, line width=0.9pt, font=\\tiny] at (%.3f,%.3f) {%s};" % (x, y, label))


def bond(b, x0, x1, y, label=None, pos="above", style="ink"):
    b.append("\\draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);" % (style, x0, y, x1, y))
    if label:
        anchor = "south" if pos == "above" else "north"
        dy = 0.03 if pos == "above" else -0.03
        b.append("\\node[font=\\scriptsize, anchor=%s] at (%.3f,%.3f) {%s};" % (anchor, 0.5 * (x0 + x1), y + dy, label))


def checks(rng):
    n = 6
    D = [2, 3, 5, 4, 2]
    cores = random_mps(n, D, rng)
    psi = mps_vector(cores)
    ranks = [np.linalg.matrix_rank(unfold(psi, k, n), 1e-9) for k in range(1, n)]
    assert ranks == D                                                   # rank of each unfolding = bond (generic)
    assert bonds(tt_svd(psi, n)) == D                                   # TT-SVD recovers the bonds
    assert np.allclose(mps_vector(tt_svd(psi, n)), psi)
    X = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))          # gauge on the bond after site 3
    g = [c.copy() for c in cores]
    g[2] = np.tensordot(g[2], X, axes=([2], [0]))
    g[3] = np.tensordot(np.linalg.inv(X), g[3], axes=([1], [0]))
    assert np.allclose(mps_vector(g), psi) and not np.allclose(g[2], cores[2])
    phi = psi / np.linalg.norm(psi)                                     # Schmidt^2 = spectrum of the reduced state
    M = unfold(phi, 3, n)
    rhoL = M @ M.conj().T
    assert np.allclose(np.sort(np.linalg.eigvalsh(rhoL))[::-1], schmidt(phi, 3, n) ** 2)
    return D


def main():
    rng = np.random.default_rng(5)
    D = checks(rng)
    n, k = 6, 3
    _, psi = ground_state(n, 1.0)
    s = schmidt(psi, k, n)
    p = s ** 2
    assert abs(p.sum() - 1) < 1e-10 and len(p) == 2 ** k
    tail4 = p[4:].sum()
    assert 1e-9 < tail4 < 2e-8
    print("random MPS bonds %s recovered; Ising u=1, N=6, cut 3: Schmidt coefficients %s; weights log10 %s; "
          "weight beyond the fourth %.1e" % (D, np.round(s, 5), np.round(np.log10(np.maximum(p, 1e-30)), 2), tail4))

    b = []
    # ---- (a) the chain, the cut, the matrix and its SVD
    y0, dx, x1 = 8.0, 1.15, 0.6
    xs = [x1 + dx * i for i in range(n)]
    b.append("\\node[font=\\small, anchor=east] at (%.3f,%.3f) {$\\ket\\psi=$};" % (xs[0] - 0.4, y0))
    for i in range(n - 1):
        lab = "$D_%d$" % (i + 1)
        if i == k - 1:                                                  # the cut bond: label above the cut
            bond(b, xs[i] + R_CORE, xs[i + 1] - R_CORE, y0)
            b.append("\\node[font=\\scriptsize, anchor=south, text=ndRed] at (%.3f,%.3f) {%s};" % (0.5 * (xs[i] + xs[i + 1]), y0 + 0.3, lab))
        else:
            bond(b, xs[i] + R_CORE, xs[i + 1] - R_CORE, y0, lab)
    for i, x in enumerate(xs):
        core(b, x, y0, "$A^{[%d]}$" % (i + 1))
        b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,%.3f) {$s_%d$};" % (x, y0 + 0.62, i + 1))
    xc = 0.5 * (xs[k - 1] + xs[k])
    b.append("\\draw[ndRed, line width=0.9pt, dash pattern=on 3pt off 2pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xc, y0 - 0.5, xc, y0 + 0.3))
    b.append("\\node[note, text=ndRed, anchor=north] at (%.3f,%.3f) {a cut};" % (xc, y0 - 0.5))
    # the unfolding as a product L R
    y1 = 6.35
    b.append("\\node[font=\\small, anchor=east] at (%.3f,%.3f) {$=$};" % (xs[0] - 0.4, y1))
    xL, xR = xc - 1.05, xc + 1.05
    b.append("\\draw[ndInk, line width=2.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xs[0] - 0.3, y1, xL - 0.39, y1))
    b.append("\\draw[ndInk, line width=2.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xR + 0.39, y1, xs[-1] + 0.3, y1))
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$s_1s_2s_3$};" % (0.5 * (xs[0] - 0.3 + xL - 0.39), y1 - 0.08))
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$s_4s_5s_6$};" % (0.5 * (xR + 0.39 + xs[-1] + 0.3), y1 - 0.08))
    bond(b, xL + 0.39, xR - 0.39, y1)
    b.append("\\node[font=\\scriptsize, anchor=south west] at (%.3f,%.3f) {$D_3$};" % (xc + 0.06, y1 + 0.03))
    box(b, xL, y1, "$L$")
    box(b, xR, y1, "$R$")
    b.append("\\draw[ndRed, line width=0.9pt, dash pattern=on 3pt off 2pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xc, y1 - 0.4, xc, y1 + 0.4))
    b.append("\\node[note, anchor=west, align=left] (bun) at (%.3f,%.3f) {bundle the legs on each\\\\ side of the cut: an $8\\times8$\\\\ matrix of rank at most $D_3$};" % (xs[-1] + 0.45, 7.45))
    b.append("\\draw[pointer] (bun.south west) to[bend left=20] (%.3f,%.3f);" % (xs[-1] + 0.35, y1 + 0.12))
    # its singular value decomposition
    y2 = 4.95
    b.append("\\node[font=\\small, anchor=east] at (%.3f,%.3f) {$=$};" % (xs[0] - 0.4, y2))
    xU, xV = xc - 1.25, xc + 1.25
    b.append("\\draw[ndInk, line width=2.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xs[0] - 0.3, y2, xU - 0.39, y2))
    b.append("\\draw[ndInk, line width=2.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xV + 0.39, y2, xs[-1] + 0.3, y2))
    bond(b, xU + 0.39, xc - 0.26, y2, "$\\alpha$")
    bond(b, xc + 0.26, xV - 0.39, y2, "$\\alpha$")
    box(b, xU, y2, "$U$")
    box(b, xV, y2, "$V^\\dagger$")
    h = 0.3                                                             # the diagonal matrix: a diamond
    b.append("\\filldraw[fill=ndGold!55, draw=ndGold!70!ndInk, line width=0.9pt] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- cycle;"
             % (xc - h, y2, xc, y2 + h, xc + h, y2, xc, y2 - h))
    b.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {$\\Lambda$};" % (xc, y2))
    b.append("\\coordinate (lam) at (%.3f,%.3f);" % (xc + h, y2 + 0.05))
    b.append("\\node[note, anchor=north] (schm) at (%.3f,%.3f) {$\\Lambda=\\operatorname{diag}(\\lambda_\\alpha)$: the Schmidt coefficients};" % (xc, y2 - 0.42))
    # the Schmidt weights of the Ising ground state
    P = N.Plot((0.4, 2 ** k + 0.6), (-13.0, 0.6), 4.4, 3.2, origin=(11.2, 4.75))
    b += P.axes("$\\alpha$", "", xticks=[(a, "$%d$" % a) for a in range(1, 2 ** k + 1)],
                yticks=[(0, "$0$"), (-4, "$-4$"), (-8, "$-8$"), (-12, "$-12$")])
    for a, pa in enumerate(p, start=1):
        top = np.log10(pa)
        b.append("\\fill[ndGold!55, draw=ndGold!70!ndInk, line width=0.5pt] %s rectangle %s;" % (P.pt(a - 0.3, -13.0), P.pt(a + 0.3, top)))
    b.append("\\node[note, anchor=south, align=center] at %s {Ising chain, $u=1$, $N=6$: the\\\\ weights $\\log_{10}\\lambda_\\alpha^2$ across the cut};" % N.pt(P.P(4.5, 0.6) + np.array([0.0, 0.12])))
    b.append("\\draw[pointer] (schm.east) to[bend right=10] %s;" % N.pt(P.P(0.4, -10.0) + np.array([-0.5, 0.0])))
    b.append("\\node[note, anchor=west, align=left] at %s {the last four weigh\\\\ about $10^{-8}$ in all};" % P.pt(4.7, -4.2))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) a cut through the chain is a rank};" % (7.6, 3.75))

    # ---- (b) the gauge
    yg1, yg2 = 2.35, 0.95
    xa, xb = 0.9, 4.1
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (xa - 0.75, yg1, xa - R_CORE, yg1))
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (xb + R_CORE, yg1, xb + 0.75, yg1))
    bond(b, xa + R_CORE, xb - R_CORE, yg1)
    core(b, xa, yg1, "$A^{[3]}$")
    core(b, xb, yg1, "$A^{[4]}$")
    gauge(b, xa + 1.05, yg1, "$X$")
    gauge(b, xb - 1.05, yg1, "$X^{\\!-\\!1}$")
    b.append("\\node[note, anchor=south] at (%.3f,%.3f) {$XX^{-1}=\\Id$};" % (0.5 * (xa + xb), yg1 + 0.22))
    b.append("\\draw[pointer] (%.3f,%.3f) -- (%.3f,%.3f);" % (0.5 * (xa + xb), yg1 - 0.35, 0.5 * (xa + xb), yg2 + 0.75))
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (xa + 0.5 - 0.75, yg2, xa + 0.5 - R_CORE, yg2))
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (xb - 0.5 + R_CORE, yg2, xb - 0.5 + 0.75, yg2))
    bond(b, xa + 0.5 + R_CORE, xb - 0.5 - R_CORE, yg2)
    core(b, xa + 0.5, yg2, "$\\tilde A^{[3]}$")
    core(b, xb - 0.5, yg2, "$\\tilde A^{[4]}$")
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {$\\tilde A^{[3]}=A^{[3]}X$, $\\tilde A^{[4]}=X^{-1}A^{[4]}$};" % (0.5 * (xa + xb), yg2 - 0.32))
    b.append("\\node[note, anchor=west, align=left] at (%.3f,%.3f) {the same state,\\\\ other cores: a\\\\ similarity, as for\\\\ realisations};" % (xb + 0.95, 1.65))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) the gauge freedom on a bond};" % (3.0, -0.35))

    # ---- (c) a cut in time and a cut in space
    xl, xr, x0c = 10.0, 13.9, 8.65
    for y, left, right, edge, formula, tag in [
            (2.45, "$\\Obsv$: seen", "$\\Ctrb$: excited", "state $\\vec x\\in\\R^n$",
             "$\\Hankel=\\Obsv\\,\\Ctrb$; rank $=$ least order $n$", "time"),
            (0.95, "sites $1..k$", "sites $k{+}1..N$", "bond, $D_k$",
             "$\\psi=L\\,R$; rank $=$ least bond dimension $D_k$", "space")]:
        b.append("\\node[note, anchor=east] at (%.3f,%.3f) {%s};" % (x0c, y, tag))
        bond(b, xl + 0.875, xr - 0.875, y, edge)
        box(b, xl, y, left, w=1.75, h=0.5)
        box(b, xr, y, right, w=1.75, h=0.5)
        xm = 0.5 * (xl + xr)
        b.append("\\draw[ndRed, line width=0.9pt, dash pattern=on 3pt off 2pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xm, y - 0.28, xm, y + 0.12))
        b.append("\\node[note, anchor=north] at (%.3f,%.3f) {%s};" % (xm, y - 0.33, formula))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(c) a cut in time and a cut in space};" % (0.5 * (xl + xr), -0.35))
    N.write("ch07_tn_cut", b)
    print("written figures/ch07_tn_cut.tex")


if __name__ == "__main__":
    main()
