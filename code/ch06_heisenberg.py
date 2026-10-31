"""ch06_heisenberg.py -- time as a resource, and what decoherence does to it.

A frequency omega (here omega = 3, as in the example of Chapter 5) is encoded
in the phase omega t of a qubit that precesses for the time t.
  (a) The phase measured after the time t determines omega only up to
      multiples of 2 pi / t: the candidates form a comb.  Robust phase
      estimation measures after t = 1, 2, 4, 8 in turn and at each stage keeps
      the one tooth that lies in the interval left by the stage before; the
      interval is halved at every stage.
  (b) The error against the total evolution time T, on logarithmic scales:
        - repeated experiments of the fixed length 1 in a closed system,
          error 1/sqrt(T), the standard quantum limit;
        - a Monte-Carlo simulation of robust phase estimation in a closed
          system (stages t_k = 2^k, a(K - k) + g shots of each of two
          quadratures at stage k), error proportional to 1/T, the Heisenberg
          limit;
        - the Cramer-Rao bound for an open system with Gamma = 0.5, for
          repeated experiments of the best length t = min(T, 1/(2 Gamma)):
          one experiment of length t carries the Fisher information
          t^2 e^{-2 Gamma t}, so that the error is 1/sqrt(T t e^{-2 Gamma t}),
          which falls like 1/T only while T < 1/(2 Gamma) and like
          sqrt(2 Gamma e / T) afterwards.
Writes figures/ch06_heisenberg.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import OMEGA, GAMMA

STAGES = (1, 2, 4, 8)


def rpe_rmse(K, rng, a=6, g=2, trials=600):
    """Total time and root-mean-square error of robust phase estimation with K stages."""
    Ms = [a * (K - k) + g for k in range(K)]
    T = sum(2 * M * 2 ** k for k, M in enumerate(Ms))
    errs = []
    for _ in range(trials):
        est = None
        for k, M in enumerate(Ms):
            t = 2 ** k
            c = 2 * rng.binomial(M, (1 + np.cos(OMEGA * t)) / 2) / M - 1
            s = 2 * rng.binomial(M, (1 + np.sin(OMEGA * t)) / 2) / M - 1
            ph = np.arctan2(s, c)
            if est is None:
                est = ph % (2 * np.pi)                      # omega is known to lie in [0, 2 pi)
            else:
                cands = (ph + 2 * np.pi * np.arange(-2 * t, 2 * t + 1)) / t
                est = cands[np.argmin(np.abs(cands - est))]
        errs.append(est - OMEGA)
    return T, np.sqrt(np.mean(np.square(errs)))


def open_bound(T):
    t = np.minimum(T, 1 / (2 * GAMMA))
    return 1 / np.sqrt(T * t * np.exp(-2 * GAMMA * t))


def vernier(b):
    """(a) the combs of candidate frequencies and the nested intervals."""
    sx, dy, top = 1.1, 0.95, 4.2
    X = lambda w: sx * w
    b.append("\\draw[faint, ->, >=stealth] (%.2f,%.2f) -- (%.2f,%.2f) node[right, font=\\small, text=ndInk] {$\\omega$};"
             % (X(0), top - len(STAGES) * dy - 0.25, X(2 * np.pi) + 0.35, top - len(STAGES) * dy - 0.25))
    for v, lab in [(0, "$0$"), (np.pi, "$\\pi$"), (2 * np.pi, "$2\\pi$")]:
        b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f) node[below, font=\\scriptsize, text=ndInk] {%s};"
                 % (X(v), top - len(STAGES) * dy - 0.2, X(v), top - len(STAGES) * dy - 0.3, lab))
    b.append("\\draw[dashedink] (%.2f,%.2f) -- (%.2f,%.2f);" % (X(OMEGA), top + 0.45, X(OMEGA), top - len(STAGES) * dy - 0.25))
    b.append("\\node[font=\\scriptsize, anchor=south] at (%.2f,%.2f) {$\\omega=%g$};" % (X(OMEGA), top + 0.45, OMEGA))
    lo, hi = 0.0, 2 * np.pi
    widths = []
    for j, t in enumerate(STAGES):
        y = top - j * dy
        b.append("\\fill[ndSky, fill opacity=0.7] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (X(lo), y - 0.05, X(hi), y + 0.42))
        b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f);" % (X(0), y, X(2 * np.pi), y))
        b.append("\\node[font=\\scriptsize, anchor=east] at (%.2f,%.2f) {$t=%d$};" % (X(0) - 0.15, y + 0.15, t))
        phase = (OMEGA * t) % (2 * np.pi)
        teeth = (phase + 2 * np.pi * np.arange(0, t + 1)) / t
        teeth = teeth[teeth <= 2 * np.pi + 1e-9]
        inside = [w for w in teeth if lo - 1e-9 <= w <= hi + 1e-9]
        assert len(inside) == 1 and abs(inside[0] - OMEGA) < 1e-9       # one tooth survives, the true one
        for w in teeth:
            keep = abs(w - OMEGA) < 1e-9
            col, lw = ("ndRed", 1.4) if keep else ("ndInk!55", 0.8)
            b.append("\\draw[line width=%.1fpt, %s] (%.2f,%.2f) -- (%.2f,%.2f);" % (lw, col, X(w), y, X(w), y + 0.36))
        lo, hi = OMEGA - np.pi / (2 * t), OMEGA + np.pi / (2 * t)
        widths.append(hi - lo)
        b.append("\\draw[line width=0.9pt, ndRed, |-|] (%.2f,%.2f) -- (%.2f,%.2f);" % (X(lo), y - 0.17, X(hi), y - 0.17))
    assert np.allclose(np.array(widths[1:]) / np.array(widths[:-1]), 0.5)      # the interval halves
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {grey: the values of $\\omega$ that agree with the phase at time $t$;\\\\"
             " shaded: the interval left by the stage before;\\\\ red: the value kept, and the new interval, half as wide};"
             % (X(np.pi), top - len(STAGES) * dy - 0.75))
    b.append("\\node[font=\\small] at (%.2f,%.2f) {(a) robust phase estimation};" % (X(np.pi), top - len(STAGES) * dy - 2.35))


def main():
    rng = np.random.default_rng(3)
    b = []
    vernier(b)
    # ---- (b) error against total time
    Ks = range(2, 12)
    data = np.array([rpe_rmse(K, rng) for K in Ks])
    lT, le = np.log10(data[:, 0]), np.log10(data[:, 1])
    slope = np.polyfit(lT, le, 1)[0]
    assert -1.15 < slope < -0.8                                         # the Heisenberg limit
    Tg = np.logspace(-1, 5, 300)
    ob = open_bound(Tg)
    big = Tg > 10
    s_open = np.polyfit(np.log10(Tg[big]), np.log10(ob[big]), 1)[0]
    assert abs(s_open + 0.5) < 1e-6                                     # back to the standard quantum limit
    assert abs(open_bound(np.array([1e4]))[0] - np.sqrt(2 * GAMMA * np.e / 1e4)) < 1e-12
    Pl = N.Plot((-1.0, 5.0), (-4.5, 1.2), 7.2, 4.6, origin=(9.3, -0.45))
    b += Pl.axes("total time $T$", "error in $\\omega$", xticks=[(k, "$10^{%d}$" % k) for k in range(-1, 6)],
                 yticks=[(k, "$10^{%d}$" % k) for k in range(-4, 2, 1)])
    T1 = np.logspace(0, 5, 50)
    b.append(Pl.curve(np.log10(T1), np.log10(1 / np.sqrt(T1)), "line width=1.0pt, ndInk!55, dash pattern=on 3pt off 2pt"))
    b.append(Pl.curve(np.log10(Tg), np.log10(ob), "line width=1.3pt, ndRed"))
    c_rpe = np.median(data[:, 0] * data[:, 1])
    b.append(Pl.curve(np.log10(T1[T1 > 30]), np.log10(c_rpe / T1[T1 > 30]), "line width=0.8pt, ndBlue, dash pattern=on 3pt off 2pt"))
    b += ["\\node[dot, fill=ndBlue] at %s {};" % Pl.pt(x, y) for x, y in zip(lT, le)]
    b.append("\\node[note, anchor=west, text=ndInk!70] (s) at %s {repeated experiments\\\\ of length $1$: $T^{-1/2}$};" % Pl.pt(1.6, 0.6))
    b.append("\\draw[pointer] (s.south west) to[bend right=15] %s;" % Pl.pt(1.3, -0.65 - 0.02))
    b.append("\\node[note, anchor=north east, text=ndBlue] (h) at %s {robust phase estimation,\\\\ closed system: $T^{-1}$};" % Pl.pt(2.3, -2.9))
    b.append("\\draw[pointer] (h.north east) to[bend right=15] %s;" % Pl.pt(lT[3] + 0.05, le[3] - 0.05))
    b.append("\\node[note, anchor=west, text=ndRed] (o) at %s {open system, $\\Gamma=%g$:\\\\ at best $T^{-1/2}$ again\\\\ once $T>1/(2\\Gamma)$};" % (Pl.pt(3.1, -0.4), GAMMA))
    b.append("\\draw[pointer] (o.south) to[bend left=15] %s;" % Pl.pt(3.9, np.log10(open_bound(np.array([10 ** 3.9]))[0]) + 0.08))
    b.append("\\node[font=\\small] at %s {(b) the error against the total time};" % N.pt(Pl.P(2.0, -4.5) + np.array([0.0, -0.95])))
    N.write("ch06_heisenberg", b)
    print("written figures/ch06_heisenberg.tex; RPE slope %.2f; open slope %.2f" % (slope, s_open))


if __name__ == "__main__":
    main()
