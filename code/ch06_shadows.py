"""ch06_shadows.py -- classical shadows from random Pauli measurements.

Each snapshot measures every qubit in a basis X, Y or Z chosen at random.  For
a Pauli observable P of weight w (acting on w qubits), the snapshot gives the
estimate 3^w s when the w random bases happen to match P, which they do with
probability 3^{-w}, and 0 otherwise; s = +-1 is the product of the w outcomes,
with mean <P>.  The estimate is unbiased, and its variance is 3^w - <P>^2.
  (a) Monte Carlo: the root-mean-square error of the average of N snapshots
      of the pair of Chapter 4 at t = 0.5 (ch06_tomography.pair_state), for
      ZI (w = 1) and XX (w = 2), and of two copies of the pair, for
      ZI XX (w = 3) and XX XX (w = 4), against the lines
      sqrt((3^w - <P>^2)/N).
  (b) The number of copies needed to estimate all M Pauli observables of
      weight at most 2 on q qubits (M = 3q + 9 q(q-1)/2) to the accuracy
      eps = 0.05 with confidence 1 - delta = 0.95: with classical shadows,
      of the order of 3^2 * 2 ln(2M/delta) / eps^2, and by measuring each
      observable on copies of its own, of the order of
      M * 2 ln(2M/delta) / eps^2.  The constants are indicative.
Writes figures/ch06_shadows.tex.
"""
import numpy as np

import needham as N
from ch06_tomography import pair_state, pauli

EPS, DELTA, WMAX = 0.05, 0.05, 2
COLS = {1: "ndBlue", 2: "ndGreen", 3: "ndGold", 4: "ndRed"}


def observables():
    """(label, weight, <P>) for ZI and XX of the pair, and ZI XX and XX XX of two copies of it."""
    rho = pair_state()
    zi, xx = [np.real(np.trace(pauli(s) @ rho)) for s in ("ZI", "XX")]
    return [("ZI", 1, zi), ("XX", 2, xx), ("ZI\,XX", 3, zi * xx), ("XX\,XX", 4, xx * xx)]


def shadow_rmse(w, mean, n_snap, rng, reps=600):
    """Root-mean-square error of the shadow estimate of <P> = mean from n_snap snapshots."""
    matched = rng.binomial(n_snap, 3.0 ** (-w), size=reps)
    plus = rng.binomial(matched, (1 + mean) / 2)
    est = 3.0 ** w * (2 * plus - matched) / n_snap
    return np.sqrt(np.mean((est - mean) ** 2))


def n_obs(q):
    return 3 * q + 9 * q * (q - 1) // 2


def main():
    rng = np.random.default_rng(5)
    Ns = np.logspace(1, 5, 9)
    b = []
    # ---- (a) error against the number of snapshots
    Pl = N.Plot((0.8, 5.2), (-3.0, 1.0), 7.0, 4.6, origin=(0.0, 0.0))
    b += Pl.axes("$N$", "error", xticks=[(k, "$10^{%d}$" % k) for k in range(1, 6)],
                 yticks=[(k, "$10^{%d}$" % k) for k in range(-3, 2)])
    for name, w, mean in observables():
        err = np.array([shadow_rmse(w, mean, int(n), rng) for n in Ns])
        theory = np.sqrt((3.0 ** w - mean ** 2) / Ns)
        assert np.all(np.abs(np.log(err / theory)) < 0.15)                 # the variance is 3^w - <P>^2
        b.append(Pl.curve(np.log10(Ns), np.log10(theory), "line width=0.9pt, %s" % COLS[w]))
        b += ["\\node[dot, fill=%s] at %s {};" % (COLS[w], Pl.pt(x, y)) for x, y in zip(np.log10(Ns), np.log10(err))]
        b.append("\\node[font=\\scriptsize, text=%s, anchor=west] at %s {$%s$, $w=%d$};" % (COLS[w], Pl.pt(5.25, np.log10(theory[-1])), name, w))
    b.append("\\node[note, anchor=north east] at %s {each weight costs a factor of\\\\ three in variance: dots, Monte Carlo;\\\\ lines, $\\sqrt{(3^w-\\langle P\\rangle^2)/N}$};"
             % Pl.pt(5.2, 1.05))
    b.append("\\node[font=\\small] at %s {(a) the error for observables of the pair, from $N$ snapshots};" % N.pt(Pl.P(3.0, -3.0) + np.array([0.0, -0.95])))
    # ---- (b) copies for all observables of weight at most 2 on q qubits
    qs = np.arange(2, 51)
    M = np.array([n_obs(q) for q in qs], dtype=float)
    shadows = 3.0 ** WMAX * 2 * np.log(2 * M / DELTA) / EPS ** 2
    separate = M * 2 * np.log(2 * M / DELTA) / EPS ** 2
    assert shadows[-1] / shadows[0] < 2.5 and separate[-1] / separate[0] > 500    # logarithmic against quadratic in q
    Q = N.Plot((0, 52), (4.0, 9.0), 6.8, 4.6, origin=(9.3, 0.0))
    b += Q.axes("qubits $q$", "copies", xticks=[(10, "$10$"), (20, "$20$"), (30, "$30$"), (40, "$40$"), (50, "$50$")],
                yticks=[(k, "$10^{%d}$" % k) for k in range(4, 10)])
    b.append(Q.curve(qs, np.log10(shadows), "line width=1.3pt, ndGreen"))
    b.append(Q.curve(qs, np.log10(separate), "line width=1.3pt, ndRed"))
    b.append("\\node[note, anchor=south, text=ndGreen] at %s {classical shadows: grows like $\\log M$};" % Q.pt(30, np.log10(shadows[28]) + 0.2))
    b.append("\\node[note, anchor=west, text=ndRed] (s) at %s {each observable measured\\\\ on copies of its own:\\\\ grows like $M\\log M$};" % Q.pt(3, 7.9))
    b.append("\\draw[pointer] (s.south) to[bend right=15] %s;" % Q.pt(18, np.log10(separate[16]) + 0.05))
    b.append("\\node[note, anchor=north west] at %s {all $M=3q+9\\binom q2$ observables\\\\ of weight at most $2$, to $\\pm%g$\\\\ with confidence $%g$};" % (Q.pt(26, 7.0), EPS, 1 - DELTA))
    b.append("\\node[font=\\small] at %s {(b) the copies needed for many observables};" % N.pt(Q.P(26, 4.0) + np.array([0.0, -0.95])))
    N.write("ch06_shadows", b)
    print("written figures/ch06_shadows.tex; q=50: M=%d, shadows %.2e, separate %.2e" % (M[-1], shadows[-1], separate[-1]))


if __name__ == "__main__":
    main()
