"""ch04_hankel.py -- persistency of excitation and the Hankel matrix.

(a) A scalar signal u_0, u_1, ... and its Hankel matrix H_L(u): column j is
    the window (u_j, ..., u_{j+L-1}), so H_L(u) is constant along
    anti-diagonals.  The signal is persistently exciting of order L when
    H_L(u) has full row rank.
(b) A minimal discrete-time system of order n = 3 with one input and one output
    (a damped rotation and a decay, sampled at tau = 0.5) is driven by one
    sinusoid, by two sinusoids, and by a random input.  The singular values of
    the data Hankel matrix [H_L(u); H_L(y)], L = 6, show its rank: 2 + 3 = 5,
    4 + 3 = 7, and the full mL + n = 9 of Willems' fundamental lemma only for
    the input that is exciting enough.
Writes figures/ch04_hankel.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N

A = np.array([[-0.15, -1.0, 0.0], [1.0, -0.15, 0.0], [0.0, 0.0, -0.35]])
B = np.array([[1.0], [0.0], [1.0]])
C = np.array([[1.0, 0.0, 1.0]])
TAU, T, L, NORD = 0.5, 120, 6, 3
AD = expm(A * TAU)
BD = np.linalg.solve(A, (AD - np.eye(3)) @ B)
X0 = np.array([0.5, -0.3, 0.2])


def hankel(sig, rows):
    return np.array([sig[i:len(sig) - rows + 1 + i] for i in range(rows)])


def simulate(u):
    x, ys = X0.copy(), []
    for uk in u:
        ys.append((C @ x).item())
        x = AD @ x + (BD * uk).ravel()
    return np.array(ys)


def main():
    assert np.linalg.matrix_rank(np.hstack([BD, AD @ BD, AD @ AD @ BD])) == 3
    assert np.linalg.matrix_rank(np.vstack([C, C @ AD, C @ AD @ AD])) == 3          # minimal, n = 3
    k = np.arange(T)
    rng = np.random.default_rng(2)
    inputs = [("one sinusoid", np.sin(0.7 * k), "ndBlue"),
              ("two sinusoids", np.sin(0.7 * k) + np.sin(1.6 * k + 0.4), "ndGold"),
              ("random input", rng.standard_normal(T), "ndRed")]
    svals, ranks = [], []
    for name, u, _ in inputs:
        H = np.vstack([hankel(u, L), hankel(simulate(u), L)])
        s = np.linalg.svd(H, compute_uv=False)
        s = s / s[0]
        r = int(np.sum(s > 1e-9))
        svals.append(s); ranks.append(r)
        print("%-14s rank H_L(u) = %d, rank of the data Hankel matrix = %d" % (name, np.linalg.matrix_rank(hankel(u, L), 1e-9), r))
    assert ranks == [5, 7, L + NORD]

    b = []
    # ---- (a) the signal and its windows
    u = inputs[1][1]
    Kshow, ncol = 14, 9
    Pl = N.Plot((-0.5, Kshow - 0.5), (-2.2, 2.2), 7.0, 2.2, origin=(0.0, 1.0))
    b.append("\\draw[faint] %s -- %s;" % (Pl.pt(-0.5, 0), Pl.pt(Kshow - 0.5, 0)))
    for i in range(Kshow):
        b.append("\\draw[line width=1.0pt, ndInk!70] %s -- %s;" % (Pl.pt(i, 0), Pl.pt(i, u[i])))
        b.append("\\fill[ndInk] %s circle (1.6pt);" % Pl.pt(i, u[i]))
    b.append("\\node[anchor=east, font=\\small] at %s {$u_k$};" % Pl.pt(-0.6, 0))
    wcols = ["ndBlue", "ndGreen", "ndRed"]
    for j, col in enumerate(wcols):
        y = 0.55 - 0.32 * j
        b.append("\\draw[line width=1.4pt, %s, |-|] %s -- %s;" % (col, N.pt((Pl.P(j - 0.3, 0)[0], y)), N.pt((Pl.P(j + L - 1 + 0.3, 0)[0], y))))
        b.append("\\node[font=\\scriptsize, text=%s, anchor=east] at %s {window %d};" % (col, N.pt((Pl.P(j - 0.3, 0)[0] - 0.05, y)), j))
    b.append("\\node[note, anchor=west] at %s {each window of $L=%d$ samples\\\\ becomes a column of $H_L(u)$};" % (N.pt((Pl.P(8.6, 0)[0], 0.25)), L))
    # the Hankel matrix as a grid of cells
    cw, x0g, y0g = 0.42, 1.55, -1.35
    vmax = np.abs(u[:L + ncol]).max()
    for i in range(L):
        for j in range(ncol):
            v = u[i + j] / vmax
            col = "ndBlue" if v > 0 else "ndRed"
            b.append("\\fill[%s!%d] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, int(10 + 70 * abs(v)), x0g + j * cw, y0g - i * cw, x0g + (j + 1) * cw, y0g - (i + 1) * cw))
    b.append("\\draw[ink] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0g, y0g, x0g + ncol * cw, y0g - L * cw))
    for j, col in enumerate(wcols):
        b.append("\\draw[line width=1.4pt, %s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, x0g + j * cw, y0g, x0g + (j + 1) * cw, y0g - L * cw))
    b.append("\\node[font=\\small, above] at (%.3f,%.3f) {$H_L(u)$};" % (x0g + ncol * cw / 2, y0g + 0.05))
    b.append("\\draw[pointer, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (x0g + 0.15, y0g - 0.2, x0g + 2.0 * cw + 0.05, y0g - 2.0 * cw - 0.05))
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {constant along anti-diagonals;\\\\ full row rank: $u$ is \\emph{persistently}\\\\ \\emph{exciting} of order $L$};" % (x0g + ncol * cw / 2, y0g - L * cw - 0.1))
    b.append("\\node[font=\\small] at (3.4,-5.6) {(a) a signal and its Hankel matrix};")
    # ---- (b) singular values of the data Hankel matrix
    lo = -12.0
    Q = N.Plot((0.5, 2 * L + 0.5), (lo, 0.6), 7.4, 5.2, origin=(9.2, -4.4))
    b += Q.axes("index $i$", "$s_i/s_1$", xticks=[(i, "$%d$" % i) for i in (1, 5, 7, 9, 12)],
                yticks=[(0, "$1$"), (-6, "$10^{-6}$"), (-12, "$10^{-12}$")])
    b.append("\\draw[dashedink] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$mL+n=%d$};" % (Q.pt(L + NORD + 0.5, lo), Q.pt(L + NORD + 0.5, 0.6), L + NORD))
    for (name, _, col), s, off in zip(inputs, svals, (-0.22, 0.0, 0.22)):
        for i, v in enumerate(s, start=1):
            if v > 1e-9:
                b.append("\\draw[line width=1.1pt, %s] %s -- %s;" % (col, Q.pt(i + off, lo), Q.pt(i + off, np.log10(v))))
                b.append("\\fill[%s] %s circle (1.6pt);" % (col, Q.pt(i + off, np.log10(v))))
            else:
                b.append("\\draw[line width=0.7pt, %s, fill=white] %s circle (1.6pt);" % (col, Q.pt(i + off, lo + 0.35)))
    xn = Q.P(L + NORD + 0.75, 0)[0]
    for j, ((name, _, col), r) in enumerate(zip(inputs, ranks)):
        b.append("\\node[note, anchor=west, text=%s] at (%.2f,%.2f) {%s: rank $%d$};" % (col, xn, Q.P(0, -1.2 - 1.5 * j)[1], name, r))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {open circles: zero};" % (xn, Q.P(0, -6.3)[1]))
    b.append("\\node[font=\\small] at %s {(b) singular values of the data Hankel matrix $\\left[\\begin{smallmatrix}H_L(u)\\\\ H_L(y)\\end{smallmatrix}\\right]$};"
             % N.pt(Q.P(L + 0.5, lo) + np.array([0, -0.85])))
    N.write("ch04_hankel", b)
    print("written figures/ch04_hankel.tex")


if __name__ == "__main__":
    main()
