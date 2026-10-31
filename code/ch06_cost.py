"""ch06_cost.py -- identifiable, but at what cost.

The qubit of Chapter 5 (ch06_shots, ch05_spectrum) is prepared with its Bloch
vector at the angle phi from the south pole, x0 = (sin phi, 0, -cos phi), and
sigma_n is measured, n = (x + z)/sqrt 2, at the times k tau, tau = 0.25, over
[0, 10].  For phi = 0 the precession is not excited and omega cannot be
identified (Chapter 5); for every phi > 0 it can.  But the amplitude of the
precession is sin phi, and the Fisher information about omega is proportional
to its square, so that the number of shots needed for a given accuracy grows
like 1/sin^2 phi.
  (a) data from N = 200 shots per time, with one-standard-deviation bars,
      for phi = 60 and 6 degrees;
  (b) the shots per time needed to estimate omega to a standard deviation of
      0.01, from the Cramer-Rao bound with (omega, gamma_1, gamma_phi) all
      unknown, against phi, and the curve c / sin^2 phi (dashed).
Writes figures/ch06_cost.tex.
"""
import numpy as np
import needham as N
from ch06_shots import tilted, expect, sample_mean, fisher, crb, THETA_TRUE, CN, TAU

TIMES = TAU * np.arange(0, 41)
SHOTS = 200
TARGET = 0.01


def shots_needed(phi):
    model = lambda th: expect(th, CN, tilted(phi), TIMES)
    return crb(fisher(model, THETA_TRUE, 1))[0, 0] / TARGET ** 2


def main():
    rng = np.random.default_rng(4)
    b = []
    # ---- (a) the data for two preparations
    tt = np.linspace(0, 10, 401)
    panels = [(60, 2.9, "ndBlue"), (6, 0.0, "ndRed")]
    for deg, oy, col in panels:
        phi = np.radians(deg)
        Pl = N.Plot((0, 10), (-0.85, 0.95), 7.4, 2.35, origin=(0.0, oy))
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\small, text=ndInk] {$t$};" % (Pl.pt(0, 0), Pl.pt(10.4, 0)))
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (Pl.pt(0, -0.85), Pl.pt(0, 0.98)))
        for v in (-0.5, 0.5):
            b.append("\\draw[faint] %s -- %s node[left, font=\\scriptsize, text=ndInk] {$%g$};" % (Pl.pt(0.12, v), Pl.pt(-0.12, v), v))
        y = expect(np.asarray(THETA_TRUE), CN, tilted(phi), TIMES)
        yh = sample_mean(y, SHOTS, rng)
        sd = np.sqrt(np.clip(1 - yh ** 2, 0, None) / SHOTS)
        for t, v, s in zip(TIMES, yh, sd):
            b.append("\\draw[line width=0.5pt, %s!70] %s -- %s;" % (col, Pl.pt(t, v - s), Pl.pt(t, v + s)))
            b.append("\\fill[%s] %s circle (0.9pt);" % (col, Pl.pt(t, v)))
        b.append(Pl.curve(tt, expect(np.asarray(THETA_TRUE), CN, tilted(phi), tt), "line width=0.8pt, ndInk!75"))
        b.append("\\node[font=\\small, text=%s, anchor=west] at %s {$\\phi=%d^\\circ$};" % (col, Pl.pt(7.6, -0.55), deg))
    amp6 = np.sin(np.radians(6)) / np.sqrt(2)
    assert amp6 < 1.2 * np.sqrt(1.0 / SHOTS)                     # the precession is of the size of the noise
    Pb = N.Plot((0, 10), (-0.85, 0.95), 7.4, 2.35, origin=(0.0, 0.0))
    b.append("\\node[note, anchor=north west] (p) at %s {the precession, of amplitude\\\\ $\\sin\\phi/\\sqrt2$, is lost in the noise};" % Pb.pt(0.3, 0.93))
    t1 = 1.6
    b.append("\\draw[pointer] (p.south west) to[bend right=20] %s;" % Pb.pt(t1, expect(np.asarray(THETA_TRUE), CN, tilted(np.radians(6)), [t1])[0] + 0.06))
    b.append("\\node[font=\\small] at (3.7,-0.9) {(a) $\\langle\\sigma_n\\rangle$ from $%d$ shots per time, for two preparations};" % SHOTS)
    # ---- (b) shots needed against phi
    degs = np.concatenate([np.linspace(1.5, 10, 18), np.linspace(11, 90, 80)])
    need = np.array([shots_needed(np.radians(d)) for d in degs])
    c = need[0] * np.sin(np.radians(degs[0])) ** 2
    small = degs < 10
    assert np.allclose(need[small] * np.sin(np.radians(degs[small])) ** 2, c, rtol=0.01)   # 1/sin^2 phi
    n60, n6 = shots_needed(np.radians(60)), shots_needed(np.radians(6))
    lo, hi = 3.4, 7.0
    Q = N.Plot((0, 90), (lo, hi), 5.6, 4.6, origin=(9.6, 0.15))
    b += Q.axes("$\\phi$", "shots per time", xticks=[(0, "$0^\\circ$"), (30, "$30^\\circ$"), (60, "$60^\\circ$"), (90, "$90^\\circ$")],
                yticks=[(k, "$10^{%d}$" % k) for k in (4, 5, 6, 7)])
    dd = np.linspace(1.3, 90, 300)
    b.append(Q.curve(dd, np.log10(c / np.sin(np.radians(dd)) ** 2), "dashedink"))
    for d in range(2, 91, 4):
        b.append("\\fill[ndInk] %s circle (1.2pt);" % Q.pt(d, np.log10(shots_needed(np.radians(d)))))
    for d, n, col, anc in [(60, n60, "ndBlue", "above right"), (6, n6, "ndRed", "right")]:
        b.append("\\node[dot, fill=%s, minimum size=4.5pt, label={[font=\\scriptsize, text=%s]%s:$%s$}] at %s {};"
                 % (col, col, anc, "{:,}".format(int(round(n, -2))).replace(",", "\\,"), Q.pt(d, np.log10(n))))
    b.append("\\node[note, anchor=west] at %s {at $\\phi=0$ the precession\\\\ is not excited, and $\\omega$\\\\ cannot be identified};" % Q.pt(12, 6.75))
    b.append("\\node[note, anchor=west] (s) at %s {dots: the Cram\\'er--Rao bound\\\\ dashed: $c/\\sin^2\\phi$};" % Q.pt(40, 5.3))
    b.append("\\node[font=\\small, align=center] at %s {(b) shots needed for a standard\\\\ deviation of $%g$ in $\\omega$};" % (N.pt(Q.P(45, lo) + np.array([0.0, -1.05])), TARGET))
    N.write("ch06_cost", b)
    print("written figures/ch06_cost.tex; shots needed at 60 and 6 degrees: %.0f, %.0f; c = %.0f" % (n60, n6, c))


if __name__ == "__main__":
    main()
