"""ch06_coins.py -- a measurement is a coin toss.

Measuring a Pauli observable gives +1 or -1, with probabilities (1 + mu)/2 and
(1 - mu)/2, where mu is the expectation value; here mu = 0.4.  The mean of N
outcomes estimates mu, with variance (1 - mu^2)/N.
  (a) the running mean of five sequences of outcomes, N = 1, ..., 2000 on a
      logarithmic axis, within the band mu +- sqrt(1 - mu^2)/sqrt(N) of one
      standard deviation (and, fainter, of two);
  (b) the distribution of the estimate for N = 10, 100 and 1000, drawn as a
      density over the value of the mean: each is sqrt(10) times narrower and
      taller than the one before.
Writes figures/ch06_coins.tex.
"""
import numpy as np
from scipy.stats import binom
import needham as N

MU = 0.4
SIG = np.sqrt(1 - MU ** 2)
NMAX = 2000


def running_means(rng, paths=5):
    p_plus = (1 + MU) / 2
    out = rng.random((paths, NMAX)) < p_plus
    o = np.where(out, 1.0, -1.0)
    return np.cumsum(o, axis=1) / np.arange(1, NMAX + 1)


def mean_distribution(n):
    """Values 1 - 2k/n of the mean and their probabilities, k = number of -1 outcomes."""
    k = np.arange(n + 1)
    return 1 - 2 * k / n, binom.pmf(k, n, (1 - MU) / 2)


def main():
    rng = np.random.default_rng(7)
    rm = running_means(rng)
    ns = np.unique(np.round(np.logspace(0, np.log10(NMAX), 260)).astype(int))
    # every path ends within three standard deviations
    assert np.all(np.abs(rm[:, -1] - MU) < 3 * SIG / np.sqrt(NMAX))
    inside = np.mean(np.abs(rm[:, 99:] - MU) < 2 * SIG / np.sqrt(np.arange(100, NMAX + 1)))
    assert inside > 0.85
    b = []
    # ---- (a) running means
    Pl = N.Plot((0.0, np.log10(NMAX)), (-1.05, 1.05), 7.6, 4.6, origin=(0.0, 0.0))
    b += Pl.axes("$N$", "mean of $N$ outcomes", xticks=[(k, "$10^{%d}$" % k) for k in range(4)],
                 yticks=[(-1, "$-1$"), (0, "$0$"), (MU, "$\\mu$"), (1, "$1$")])
    lx = np.log10(ns)
    for m, op in [(2.0, 0.25), (1.0, 0.55)]:
        up = np.clip(MU + m * SIG / np.sqrt(ns), -1.05, 1.05)
        lo = np.clip(MU - m * SIG / np.sqrt(ns), -1.05, 1.05)
        poly = [Pl.P(x, y) for x, y in zip(lx, up)] + [Pl.P(x, y) for x, y in zip(lx[::-1], lo[::-1])]
        b.append("\\fill[ndSky, fill opacity=%.2f] %s -- cycle;" % (op, " -- ".join(N.pt(p) for p in poly)))
    b.append("\\draw[dashedink] %s -- %s;" % (Pl.pt(0, MU), Pl.pt(np.log10(NMAX), MU)))
    cols = ["ndBlue", "ndRed", "ndGreen", "ndGold", "ndInk!60"]
    for path, col in zip(rm, cols):
        b.append(Pl.curve(lx, path[ns - 1], "line width=0.6pt, %s" % col))
    b.append("\\node[note, anchor=south west] (bd) at %s {one standard deviation:\\\\ $\\mu\\pm\\sqrt{1-\\mu^2}/\\sqrt N$};" % Pl.pt(1.75, 0.72))
    b.append("\\draw[pointer] (bd.south west) to[bend right=20] %s;" % Pl.pt(1.55, MU + SIG / np.sqrt(10 ** 1.55) - 0.02))
    b.append("\\node[note, anchor=north west] at %s {five sequences of outcomes,\\\\ each $+1$ or $-1$, with $\\mu=%.1f$};" % (Pl.pt(1.55, -0.3), MU))
    b.append("\\node[font=\\small] at %s {(a) the running mean of the outcomes};" % N.pt(Pl.P(np.log10(NMAX) / 2, -1.05) + np.array([0.0, -0.95])))
    # ---- (b) the distribution of the estimate
    xs = (-0.45, 1.05)
    Q = N.Plot(xs, (0.0, 14.0), 6.6, 4.6, origin=(9.2, 0.0))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk] {value of the mean};" % (Q.pt(xs[0], 0), Q.pt(xs[1] + 0.06, 0)))
    for v in (0.0, MU, 1.0):
        lab = "$\\mu$" if v == MU else "$%g$" % v
        b.append("\\draw[faint] %s -- %s node[below, font=\\scriptsize, text=ndInk] {%s};" % (Q.pt(v, 0.25), Q.pt(v, -0.25), lab))
    b.append("\\draw[dashedink] %s -- %s;" % (Q.pt(MU, 0), Q.pt(MU, 13.6)))
    spec = [(10, "ndGold"), (100, "ndGreen"), (1000, "ndBlue")]
    peaks = []
    for n, col in spec:
        v, p = mean_distribution(n)
        dens = p / (2.0 / n)                                  # probability per unit of the mean
        sd = np.sqrt(np.sum(p * (v - MU) ** 2))
        assert abs(sd - SIG / np.sqrt(n)) < 1e-9             # variance (1 - mu^2)/N
        keep = (v >= xs[0]) & (v <= xs[1])
        if n == 10:
            for vv, dd in zip(v[keep], dens[keep]):
                b.append("\\filldraw[fill=%s!35, draw=%s, line width=0.6pt] %s rectangle %s;" % (col, col, Q.pt(vv - 0.09, 0), Q.pt(vv + 0.09, dd)))
        else:
            order = np.argsort(v[keep])
            b.append(Q.curve(v[keep][order], dens[keep][order], "line width=1.1pt, %s" % col))
        i = np.argmax(dens)
        peaks.append((v[i], dens[i]))
    assert 2.8 < peaks[1][1] / peaks[0][1] < 3.6 and 2.9 < peaks[2][1] / peaks[1][1] < 3.4
    b.append("\\node[font=\\small, text=ndGold!90!ndInk, anchor=south west] at %s {$N=10$};" % Q.pt(0.62, 1.45))
    b.append("\\node[font=\\small, text=ndGreen, anchor=west] at %s {$N=100$};" % Q.pt(0.6, 3.9))
    b.append("\\node[font=\\small, text=ndBlue, anchor=west] at %s {$N=1000$};" % Q.pt(0.47, 12.2))
    b.append("\\node[note, anchor=west] at %s {each distribution is\\\\ $\\sqrt{10}$ times narrower\\\\ and taller than the last};" % Q.pt(0.62, 8.0))
    b.append("\\node[font=\\small] at %s {(b) the distribution of the estimate};" % N.pt(Q.P(0.3, 0) + np.array([0.0, -0.95])))
    N.write("ch06_coins", b)
    print("written figures/ch06_coins.tex; fraction within two standard deviations %.3f" % inside)


if __name__ == "__main__":
    main()
