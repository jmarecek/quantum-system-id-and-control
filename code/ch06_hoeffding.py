"""ch06_hoeffding.py -- Hoeffding's inequality and the number of shots.

  (a) For N = 100 shots of an observable with outcomes +-1 and mean mu = 0,
      the probability that the mean of the outcomes misses mu by at least eps:
      exactly (from the binomial distribution, a staircase), in the Gaussian
      approximation 2 Phi(-eps sqrt N), and Hoeffding's bound
      2 exp(-N eps^2 / 2), which lies above the exact tail everywhere.
  (b) The number of shots N = 2 ln(2/delta) / eps^2 that Hoeffding's
      inequality requires for accuracy eps with probability 1 - delta, for
      delta = 0.05 and 0.001: a tenfold gain in accuracy costs a hundredfold
      increase in shots, whereas a fiftyfold gain in confidence only about
      doubles them.
Writes figures/ch06_hoeffding.tex.
"""
import numpy as np
from scipy.stats import binom, norm
import needham as N

NSH = 100


def exact_tail(eps):
    """P(|mean - 0| >= eps) for the mean of NSH outcomes +-1 with mean 0."""
    k = np.arange(NSH + 1)
    p = binom.pmf(k, NSH, 0.5)
    m = 1 - 2 * k / NSH
    return np.array([p[np.abs(m) >= e - 1e-12].sum() for e in np.atleast_1d(eps)])


def shots(eps, delta):
    return 2 * np.log(2 / delta) / eps ** 2


def main():
    eps = np.linspace(0.0, 0.5, 501)
    ex = exact_tail(eps)
    ga = np.minimum(1.0, 2 * norm.sf(eps * np.sqrt(NSH)))
    ho = np.minimum(1.0, 2 * np.exp(-NSH * eps ** 2 / 2))
    assert np.all(ho >= ex - 1e-15)                                       # the bound is a bound
    assert abs(np.log10(ho[-1]) - np.log10(2 * np.exp(-12.5))) < 1e-9
    b = []
    # ---- (a) tails, on a logarithmic scale
    lo = -9.0
    Pl = N.Plot((0.0, 0.5), (lo, 0.4), 7.4, 4.6, origin=(0.0, 0.0))
    b += Pl.axes("$\\eps$", "$\\Prob(\\abs{\\hat o-\\mu}\\ge\\eps)$", xticks=[(0.1, "$0.1$"), (0.2, "$0.2$"), (0.3, "$0.3$"), (0.4, "$0.4$"), (0.5, "$0.5$")],
                 yticks=[(0, "$1$"), (-3, "$10^{-3}$"), (-6, "$10^{-6}$"), (-9, "$10^{-9}$")])
    clip = lambda v: np.maximum(np.log10(np.maximum(v, 1e-300)), lo)
    # the exact tail is a staircase: draw it with vertical risers
    xs, ys = [], []
    for e0, e1, v in zip(eps[:-1], eps[1:], ex[:-1]):
        xs += [e0, e1]
        ys += [v, v]
    b.append(Pl.curve(np.array(xs), clip(np.array(ys)), "line width=1.1pt, ndInk"))
    b.append(Pl.curve(eps, clip(ga), "line width=1.0pt, ndBlue, dash pattern=on 3pt off 1.5pt"))
    b.append(Pl.curve(eps, clip(ho), "line width=1.3pt, ndRed"))
    b.append("\\node[note, anchor=south west, text=ndRed] (h) at %s {Hoeffding's bound\\\\ $2\\,\\ee^{-N\\eps^2/2}$};" % Pl.pt(0.34, -1.25))
    b.append("\\draw[pointer] (h.south west) to[bend right=20] %s;" % Pl.pt(0.32, np.log10(2 * np.exp(-NSH * 0.32 ** 2 / 2)) + 0.12))
    b.append("\\node[note, anchor=north east, text=ndBlue] (g) at %s {Gaussian\\\\ approximation};" % Pl.pt(0.24, -5.4))
    b.append("\\draw[pointer] (g.north east) to[bend right=15] %s;" % Pl.pt(0.3, np.log10(2 * norm.sf(0.3 * 10)) - 0.1))
    b.append("\\node[note, anchor=north west] (x) at %s {exact, for\\\\ $N=%d$ shots};" % (Pl.pt(0.38, -7.0), NSH))
    b.append("\\draw[pointer] (x.north west) to[bend left=15] %s;" % Pl.pt(0.36, clip(exact_tail(0.36))[0] - 0.05))
    b.append("\\node[font=\\small] at %s {(a) the probability of an error of at least $\\eps$};" % N.pt(Pl.P(0.25, lo) + np.array([0.0, -0.95])))
    # ---- (b) shots required, log-log
    le = np.linspace(np.log10(0.003), np.log10(0.3), 200)
    Q = N.Plot((-2.6, -0.4), (1.0, 6.6), 6.4, 4.6, origin=(9.4, 0.0))
    b += Q.axes("$\\eps$", "shots $N$", xticks=[(-2, "$10^{-2}$"), (-1, "$10^{-1}$")],
                yticks=[(k, "$10^{%d}$" % k) for k in range(2, 7)])
    for delta, col, pos in [(0.05, "ndBlue", -0.75), (0.001, "ndRed", -0.75)]:
        ly = np.log10(shots(10 ** le, delta))
        keep = (ly <= 6.6) & (le <= -0.4)
        b.append(Q.curve(le[keep], ly[keep], "line width=1.2pt, %s" % col))
    assert abs(shots(0.01, 0.05) / shots(0.1, 0.05) - 100) < 1e-9
    ratio = shots(0.1, 0.001) / shots(0.1, 0.05)
    assert 2.0 < ratio < 2.1
    # a factor of ten in accuracy is a factor of a hundred in shots
    e1, e2 = -1.0, -2.0
    y1, y2 = np.log10(shots(0.1, 0.05)), np.log10(shots(0.01, 0.05))
    b.append("\\draw[dashedink] %s -- %s -- %s;" % (Q.pt(e1, y1), Q.pt(e1, y2), Q.pt(e2, y2)))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % Q.pt(e1, y1))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % Q.pt(e2, y2))
    b.append("\\node[note, anchor=west] at %s {ten times the accuracy\\\\ needs a hundred times\\\\ the shots};" % Q.pt(-0.98, 4.3))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=north east] at %s {$\\delta=0.05$};" % Q.pt(-0.42, np.log10(shots(10 ** -0.42, 0.05)) - 0.05))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=south west] at %s {$\\delta=0.001$};" % Q.pt(-0.92, np.log10(shots(10 ** -0.92, 0.001)) + 0.05))
    b.append("\\node[note, anchor=north west] at %s {the confidence is cheap:\\\\ $\\delta=0.001$ needs only about\\\\ twice the shots of $\\delta=0.05$};" % Q.pt(-2.55, 2.5))
    b.append("\\node[font=\\small] at %s {(b) shots for accuracy $\\eps$ with probability $1-\\delta$};" % N.pt(Q.P(-1.5, 1.0) + np.array([0.0, -0.95])))
    N.write("ch06_hoeffding", b)
    print("written figures/ch06_hoeffding.tex; N(0.1, 0.05) = %.0f, N(0.01, 0.05) = %.0f, ratio delta = %.2f"
          % (shots(0.1, 0.05), shots(0.01, 0.05), ratio))


if __name__ == "__main__":
    main()
