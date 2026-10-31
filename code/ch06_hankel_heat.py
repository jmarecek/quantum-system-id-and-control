"""ch06_hankel_heat.py -- what the shots do to the Hankel matrix.

The set-up of ch06_hankel_noise: the qubit of Chapter 5 measured along n,
K = 60 samples at tau = 0.25, Hankel matrix H(0) with R = 10 block rows and
S = 40 columns.  Each sample is the mean of N shots, so that the Hankel matrix
of the data is H_hat = H + E, where E is the Hankel matrix of the errors.
  (a) H, E and H_hat for N = 100 shots per sample, on one colour scale (blue
      positive, red negative).  E is constant along anti-diagonals, like H:
      the error of each sample is repeated along its anti-diagonal, so that
      E has only K - 1 = R + S - 1 independent entries.
  (b) The singular values of H (dots) with the bands sigma_i +- ||E||_2 within
      which Weyl's inequality keeps those of H_hat (open circles), for N = 10^2
      and N = 10^4; for i > 4 the band is [0, ||E||_2], the noise floor.  At
      N = 10^2, ||E||_2 is about 0.77 and exceeds sigma_4 = 0.58, so that the
      fourth mode may be lost in the floor; at N = 10^4 every band is narrow.  The bound (6.2)
      of Proposition prop:ch06:hokalman at delta = 0.05 (bars on the right)
      is about ten times the actual ||E||_2.
Writes figures/ch06_hankel_heat.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import hankel, X0
from ch06_shots import sample_mean
from ch06_hankel_noise import data, R, S, K

DELTA = 0.05
SHOTS = (100, 10000)


def bound(shots, p=1):
    """The right-hand side of (6.2)."""
    return np.sqrt(R * S * p) * np.sqrt(2.0 / shots * np.log(2 * (R + S - 1) * p / DELTA))


def heat(b, M, vmax, x0, y0, cw, ch):
    """Draw the matrix M as cells of width cw and height ch, top-left corner at (x0, y0)."""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = np.clip(M[i, j] / vmax, -1, 1)
            col = "ndBlue" if v > 0 else "ndRed"
            b.append("\\fill[%s!%d] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                     % (col, int(round(4 + 86 * abs(v))), x0 + j * cw, y0 - i * ch, x0 + (j + 1) * cw, y0 - (i + 1) * ch))
    b.append("\\draw[ink, line width=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0, y0, x0 + M.shape[1] * cw, y0 - M.shape[0] * ch))


def main():
    rng = np.random.default_rng(3)
    y = data(X0)
    H = hankel(y[:, None], R, S)
    sH = np.linalg.svd(H, compute_uv=False)
    assert sH[4] < 1e-12 and abs(sH[3] - 0.58) < 0.01
    # the size of E over many repetitions, and one draw for the pictures
    norms = {n: np.median([np.linalg.norm(hankel((sample_mean(y, n, rng) - y)[:, None], R, S), 2) for _ in range(300)])
             for n in SHOTS}
    for n in SHOTS:
        print("N = %5d: median ||E||_2 = %.3f, bound (6.2) = %.3f, ratio %.1f" % (n, norms[n], bound(n), bound(n) / norms[n]))
        assert 7 < bound(n) / norms[n] < 14                                  # the bound is about ten times too large
    assert norms[100] > sH[3] > 3 * norms[10000]                             # N = 100 lies below N*, N = 10^4 well above
    draws = {}
    for n in SHOTS:
        while True:                                                          # a typical draw: ||E|| near its median
            E = hankel((sample_mean(y, n, rng) - y)[:, None], R, S)
            if abs(np.linalg.norm(E, 2) / norms[n] - 1) < 0.05:
                break
        draws[n] = (E, np.linalg.norm(E, 2), np.linalg.svd(H + E, compute_uv=False))
        assert np.all(np.abs(draws[n][2] - sH) <= draws[n][1] + 1e-12)     # Weyl's inequality
    for i in range(R):                                                       # E is constant along anti-diagonals
        for j in range(S - 1):
            if i + 1 < R:
                assert abs(draws[100][0][i + 1, j] - draws[100][0][i, j + 1]) < 1e-14

    b = []
    # ---- (a) the three matrices
    E = draws[100][0]
    vmax = np.abs(H).max()
    cw, ch, x0 = 0.118, 0.118, 0.9
    tops = [2.15, 0.45, -1.25]
    for M, top, lab in zip((H, E, H + E), tops, ("$\\Hankel$", "$E$", "$\\hat\\Hankel$")):
        heat(b, M, vmax, x0, top, cw, ch)
        b.append("\\node[font=\\small, anchor=east] at (%.2f,%.2f) {%s};" % (x0 - 0.12, top - R * ch / 2, lab))
    for top, sym in zip(tops[1:], ("$+$", "$=$")):
        b.append("\\node[font=\\small] at (%.2f,%.2f) {%s};" % (x0 + S * cw / 2, top + 0.24, sym))
    xr = x0 + S * cw
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {$40$ columns,\\\\ $10$ rows};" % (xr + 0.1, tops[0] - R * ch / 2))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {each error\\\\ repeats along\\\\ its anti-diagonal};" % (xr + 0.1, tops[1] - R * ch / 2))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {$N=100$\\\\ shots per\\\\ sample};" % (xr + 0.1, tops[2] - R * ch / 2))
    b.append("\\node[font=\\small] at (%.2f,%.2f) {(a) the Hankel matrix, its error, and the data};" % (x0 + S * cw / 2 + 0.6, -2.95))
    # ---- (b) singular values and the Weyl bands
    lo, hi = -2.0, 1.2
    Q = N.Plot((0.4, 7.7), (lo, hi), 6.4, 4.6, origin=(9.2, -2.25))
    b += Q.axes("$i$", "singular value", xticks=[(i, "$%d$" % i) for i in range(1, 7)] + [(7.2, "bound")],
                yticks=[(k, "$10^{%d}$" % k) for k in (-2, -1, 0, 1)])
    styles = [(100, -0.17, "ndRed", "ndRose"), (10000, 0.17, "ndBlue", "ndSky")]
    for n, off, col, wash in styles:
        _, e, sh = draws[n]
        for i in range(6):
            s = sH[i] if i < 4 else 0.0
            top = np.log10(s + e)
            bot = np.log10(s - e) if s - e > 10 ** lo else lo
            b.append("\\fill[%s] %s rectangle %s;" % (wash, Q.pt(i + 1 + off - 0.11, bot), Q.pt(i + 1 + off + 0.11, top)))
            b.append("\\draw[%s, line width=0.4pt] %s rectangle %s;" % (col, Q.pt(i + 1 + off - 0.11, bot), Q.pt(i + 1 + off + 0.11, top)))
            if s - e <= 10 ** lo:                                            # the band reaches down to zero
                b.append("\\draw[%s, line width=0.6pt, ->, >=stealth] %s -- %s;" % (col, Q.pt(i + 1 + off, lo + 0.25), Q.pt(i + 1 + off, lo - 0.02)))
            b.append("\\draw[%s, line width=0.6pt, fill=white] %s circle (1.5pt);" % (col, Q.pt(i + 1 + off, np.log10(max(sh[i], 10 ** lo)))))
            if i < 4:
                b.append("\\fill[ndInk] %s circle (0.9pt);" % Q.pt(i + 1 + off, np.log10(s)))
        # the bound (6.2) on the right
        xb = 7.2 + off
        b.append("\\fill[%s] %s rectangle %s;" % (wash, Q.pt(xb - 0.11, lo), Q.pt(xb + 0.11, np.log10(bound(n)))))
        b.append("\\draw[%s, line width=0.4pt] %s rectangle %s;" % (col, Q.pt(xb - 0.11, lo), Q.pt(xb + 0.11, np.log10(bound(n)))))
        b.append("\\draw[%s, line width=0.9pt] %s -- %s;" % (col, Q.pt(xb - 0.16, np.log10(e)), Q.pt(xb + 0.16, np.log10(e))))
    b.append("\\node[note, anchor=north west, text=ndRed] (r) at %s {$N=10^2$: $\\lVert E\\rVert_2=%.2f$\\\\ exceeds $\\sigma_4=%.2f$};"
             % (Q.pt(0.55, -0.55), draws[100][1], sH[3]))
    b.append("\\draw[pointer] (r.east) to[bend left=10] %s;" % Q.pt(4 - 0.17 - 0.14, -1.0))
    b.append("\\node[note, anchor=north west, text=ndBlue] (u) at %s {$N=10^4$: $\\lVert E\\rVert_2=%.2f$};" % (Q.pt(0.55, -1.38), draws[10000][1]))
    b.append("\\draw[pointer] (u.east) to[bend right=10] %s;" % Q.pt(5 + 0.17 - 0.14, -1.3))
    b.append("\\node[note, anchor=east] at %s {line: $\\lVert E\\rVert_2$;\\\\ box: up to\\\\ the bound \\eqref{eq:ch06:hankelerror}};" % Q.pt(6.95, 0.42))
    b.append("\\node[note, anchor=north west] at %s {dots: $\\sigma_i(\\Hankel)$;\\\\ circles: $\\sigma_i(\\hat\\Hankel)$;\\\\ boxes: $\\sigma_i(\\Hankel)\\pm\\lVert E\\rVert_2$};" % Q.pt(1.4, 1.3))
    b.append("\\node[font=\\small] at %s {(b) the singular values move by at most $\\lVert E\\rVert_2$};" % N.pt(Q.P(3.5, lo) + np.array([0.0, -0.75])))
    N.write("ch06_hankel_heat", b)
    print("written figures/ch06_hankel_heat.tex; sigma(H) =", np.round(sH[:4], 3))


if __name__ == "__main__":
    main()
