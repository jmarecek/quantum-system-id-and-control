"""ch06_snapshots.py -- how a classical shadow is made.

The state is the pair of Chapter 4 at t = 0.5 (ch06_tomography.pair_state).
Each snapshot measures each qubit in a basis X, Y or Z chosen at random and
records the outcomes +-1.  A snapshot informs <ZI> only if the first qubit was
measured along Z, which happens once in three times, and then its estimate is
3 s_1; it informs <XX> only if both qubits were measured along X, once in nine
times, and then its estimate is 9 s_1 s_2; otherwise its estimate is 0.  The
estimates are unbiased, with variances 3 - <ZI>^2 and 9 - <XX>^2.
  (a) The first 18 snapshots of a run: the bases (coloured tiles) and the
      outcomes, and the estimate that each gives of <ZI> and of <XX>.
  (b) The running means of these estimates over the run, against the number
      N of snapshots, with the bands <P> +- sqrt((3^w - <P>^2)/N) of one
      standard deviation; the band of <XX> (w = 2) is wider.
Writes figures/ch06_snapshots.tex.
"""
import numpy as np

import needham as N
from ch06_tomography import pair_state, pauli

SHOW = 18
RUN = 10 ** 4
BASES = "XYZ"
WASH = {"X": "ndRose", "Y": "ndGold!30", "Z": "ndSky"}


def outcome_probabilities(rho):
    """p[b1, b2, k]: probability of the outcomes k = (s1, s2) in (++, +-, -+, --) for the bases b1, b2."""
    I2 = np.eye(2)
    p = np.zeros((3, 3, 4))
    for i, a in enumerate(BASES):
        for j, c in enumerate(BASES):
            for k, (s1, s2) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
                proj = np.kron((I2 + s1 * pauli(a)) / 2, (I2 + s2 * pauli(c)) / 2)
                p[i, j, k] = np.real(np.trace(proj @ rho))
    return p


def snapshots(rho, n, rng):
    """n snapshots: bases (n, 2) as indices into XYZ, and outcomes (n, 2) in {+1, -1}."""
    p = outcome_probabilities(rho)
    bases = rng.integers(0, 3, size=(n, 2))
    u = rng.random(n)
    cum = np.cumsum(p[bases[:, 0], bases[:, 1]], axis=1)
    k = np.minimum((u[:, None] > cum).sum(axis=1), 3)
    signs = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)])[k]
    return bases, signs


def estimates(bases, signs):
    zi = np.where(bases[:, 0] == 2, 3.0 * signs[:, 0], 0.0)
    xx = np.where((bases[:, 0] == 0) & (bases[:, 1] == 0), 9.0 * signs[:, 0] * signs[:, 1], 0.0)
    return zi, xx


def main():
    rho = pair_state()
    zi_true = np.real(np.trace(pauli("ZI") @ rho))
    xx_true = np.real(np.trace(pauli("XX") @ rho))
    # a run whose first snapshots show a typical number of matches
    for seed in range(100):
        rng = np.random.default_rng(seed)
        bases, signs = snapshots(rho, RUN, rng)
        zi, xx = estimates(bases, signs)
        nz, nx = np.sum(bases[:SHOW, 0] == 2), np.sum((bases[:SHOW, 0] == 0) & (bases[:SHOW, 1] == 0))
        if nz == 6 and nx == 3 and zi[:SHOW].sum() == -6 and xx[:SHOW].sum() == 9:
            break
    assert nz == 6 and nx == 3
    zi_mean, xx_mean = np.mean(zi), np.mean(xx)
    assert abs(zi_mean - zi_true) < 3 * np.sqrt(3 / RUN) and abs(xx_mean - xx_true) < 3 * np.sqrt(9 / RUN)
    assert abs(np.var(zi) - (3 - zi_true ** 2)) < 0.15 and abs(np.var(xx) - (9 - xx_true ** 2)) < 0.6
    b = []
    # ---- (a) the strip
    w, h, x0 = 0.5, 0.62, 2.0
    rows = [("first qubit", 3.2), ("second qubit", 2.5), ("for $\\langle ZI\\rangle$", 1.55), ("for $\\langle XX\\rangle$", 0.85)]
    for lab, y in rows:
        b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {%s};" % (x0 - 0.1, y + h / 2, lab))
    for k in range(SHOW):
        x = x0 + k * w
        for q, (_, y) in enumerate(rows[:2]):
            bb = BASES[bases[k, q]]
            b.append("\\fill[%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (WASH[bb], x + 0.03, y, x + w - 0.03, y + h))
            b.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {$%s$};" % (x + w / 2, y + 0.42, bb))
            b.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {$%s$};" % (x + w / 2, y + 0.16, "+" if signs[k, q] > 0 else "-"))
        if bases[k, 0] == 2:
            b.append("\\draw[line width=0.9pt, ndBlue] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x + 0.03, rows[0][1], x + w - 0.03, rows[0][1] + h))
        if bases[k, 0] == 0 and bases[k, 1] == 0:
            b.append("\\draw[line width=0.9pt, ndRed] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x + 0.03, rows[1][1], x + w - 0.03, rows[0][1] + h))
        for v, y, col in [(zi[k], rows[2][1], "ndBlue"), (xx[k], rows[3][1], "ndRed")]:
            if v != 0:
                b.append("\\fill[%s!15] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, x + 0.03, y + 0.05, x + w - 0.03, y + h - 0.05))
                b.append("\\node[font=\\scriptsize, text=%s] at (%.3f,%.3f) {$%d$};" % (col, x + w / 2, y + h / 2, int(v)))
            else:
                b.append("\\node[font=\\scriptsize, text=ndInk!40] at (%.3f,%.3f) {$0$};" % (x + w / 2, y + h / 2))
    xe = x0 + SHOW * w + 0.1
    for v, y, col in [(np.mean(zi[:SHOW]), rows[2][1], "ndBlue"), (np.mean(xx[:SHOW]), rows[3][1], "ndRed")]:
        b.append("\\node[font=\\scriptsize, text=%s, anchor=west] at (%.3f,%.3f) {mean $%.2f$};" % (col, xe, y + h / 2, v))
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {blue frame: measured along $Z$, so it counts for $\\langle ZI\\rangle$ (one in three);\\\\ red frame: both along $X$, so it counts for $\\langle XX\\rangle$ (one in nine)};" % (x0 + SHOW * w / 2, 0.72))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) eighteen snapshots of the pair, and what each says};" % (x0 + SHOW * w / 2 - 0.4, -0.55))
    # ---- (b) the running means
    n = np.arange(1, RUN + 1)
    run_zi, run_xx = np.cumsum(zi) / n, np.cumsum(xx) / n
    Pl = N.Plot((0.7, 4.0), (-1.5, 1.5), 3.6, 3.6, origin=(13.4, 0.2))
    b += Pl.axes("$N$", "estimate", xticks=[(k, "$10^{%d}$" % k) for k in (1, 2, 3, 4)],
                 yticks=[(-1, "$-1$"), (0, "$0$"), (1, "$1$")])
    ls = np.linspace(0.7, 4.0, 120)
    for true, wgt, col in [(zi_true, 1, "ndBlue"), (xx_true, 2, "ndRed")]:
        sd = np.sqrt((3.0 ** wgt - true ** 2) / 10 ** ls)
        top, bot = np.minimum(true + sd, 1.5), np.maximum(true - sd, -1.5)
        poly = [Pl.P(x, y) for x, y in zip(ls, top)] + [Pl.P(x, y) for x, y in zip(ls[::-1], bot[::-1])]
        b.append("\\fill[%s, fill opacity=0.13] %s -- cycle;" % (col, " -- ".join(N.pt(p) for p in poly)))
        b.append("\\draw[dashedink, %s] %s -- %s;" % (col, Pl.pt(0.7, true), Pl.pt(4.0, true)))
    idx = np.unique(np.logspace(np.log10(5), 4, 300).astype(int)) - 1
    b.append("\\begin{scope}\\clip %s rectangle %s;" % (Pl.pt(0.7, -1.5), Pl.pt(4.0, 1.5)))
    for r, col in [(run_zi, "ndBlue"), (run_xx, "ndRed")]:
        b.append(Pl.curve(np.log10(idx + 1), r[idx], "line width=0.8pt, %s" % col))
    b.append("\\end{scope}")
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=west] at %s {$\\langle ZI\\rangle$};" % Pl.pt(4.05, zi_true))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at %s {$\\langle XX\\rangle$};" % Pl.pt(4.05, xx_true))
    b.append("\\node[font=\\small] at %s {(b) the running means};" % N.pt(Pl.P(2.35, -1.5) + np.array([0.0, -0.75])))
    N.write("ch06_snapshots", b)
    print("written figures/ch06_snapshots.tex; seed %d; <ZI> = %.3f, <XX> = %.3f; after %d: %.3f, %.3f; "
          "first %d: %d for ZI, %d for XX" % (seed, zi_true, xx_true, RUN, zi_mean, xx_mean, SHOW, nz, nx))


if __name__ == "__main__":
    main()
