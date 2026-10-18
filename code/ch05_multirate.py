"""ch05_multirate.py -- two sampling rates, one frequency.

Samples taken every tau determine a frequency omega only up to multiples of
2 pi / tau: the candidates  omega + 2 pi k / tau,  k in Z,  form a comb.
(a) tau_1 = 1.6 and tau_2 = sqrt(2) tau_1: the combs share exactly one tooth
    in the window, the true omega = 3 of the running example.
(b) tau_1 = 1.6 and tau_2 = (3/2) tau_1 = 2.4: rational ratio, the combs
    coincide every 2 (2 pi / tau_1) = 7.85 and the ambiguity remains.
Each rate on its own reports the tooth inside its Nyquist band |nu| < pi/tau
(shaded): omega folded to -0.93 (tau_1), 0.22 (sqrt 2 tau_1), 0.38 (3/2 tau_1).
Writes figures/ch05_multirate.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import OMEGA

TAU1 = 1.6
NU_MIN, NU_MAX = -8.6, 12.6
SX, X0 = 0.66, 6.1                  # cm per unit frequency; page x of nu = 0
TOOTH = 0.7


def comb(tau):
    step = 2 * np.pi / tau
    k = np.arange(np.ceil((NU_MIN - OMEGA) / step), np.floor((NU_MAX - OMEGA) / step) + 1)
    return OMEGA + step * k


def X(nu):
    return X0 + SX * nu


def folded(tau):
    """The representative of omega in the Nyquist band (-pi/tau, pi/tau]."""
    return np.angle(np.exp(1j * OMEGA * tau)) / tau


def common(c1, c2, tol=1e-9):
    return np.array([a for a in c1 if np.min(np.abs(c2 - a)) < tol])


def row(y, tau2, lab2, b):
    c1, c2 = comb(TAU1), comb(tau2)
    shared = common(c1, c2)
    # Nyquist bands
    for tau, col, sgn in [(TAU1, "ndBlue", 1), (tau2, "ndRed", -1)]:
        nq = np.pi / tau
        b.append("\\fill[%s!18] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, X(-nq), y, X(nq), y + sgn * (TOOTH + 0.12)))
    b.append("\\draw[faint, ->, >=stealth] (%.3f,%.3f) -- (%.3f,%.3f) node[right, text=ndInk, font=\\small] {$\\nu$};" % (X(NU_MIN - 0.2), y, X(NU_MAX + 0.3), y))
    for v in (0,):
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (X(v), y - 0.08, X(v), y + 0.08))
    for tau, cs, col, sgn in [(TAU1, c1, "ndBlue", 1), (tau2, c2, "ndRed", -1)]:
        f = folded(tau)
        for c in cs:
            tip = y + sgn * TOOTH
            b.append("\\draw[line width=1.1pt, %s] (%.3f,%.3f) -- (%.3f,%.3f);" % (col, X(c), y, X(c), tip))
            if abs(c - f) < 1e-9:
                b.append("\\fill[%s] (%.3f,%.3f) circle (2.0pt);" % (col, X(c), tip))
            else:
                b.append("\\draw[%s, line width=0.7pt, fill=white] (%.3f,%.3f) circle (1.6pt);" % (col, X(c), tip))
    for s in shared:
        b.append("\\draw[ndGold, line width=1.6pt] (%.3f,%.3f) circle (5pt);" % (X(s), y))
    b.append("\\node[font=\\small, text=ndBlue, anchor=east] at (%.3f,%.3f) {$\\tau_1=%.1f$};" % (X(NU_MIN) - 0.15, y + 0.42, TAU1))
    b.append("\\node[font=\\small, text=ndRed, anchor=east] at (%.3f,%.3f) {%s};" % (X(NU_MIN) - 0.15, y - 0.42, lab2))
    return c1, c2, shared


def main():
    b = []
    tau2a, tau2b = np.sqrt(2) * TAU1, 1.5 * TAU1
    ya, yb = 2.6, -1.0
    c1, c2, sa = row(ya, tau2a, "$\\tau_2=\\sqrt2\\,\\tau_1$", b)
    assert len(sa) == 1 and abs(sa[0] - OMEGA) < 1e-12
    others = [abs(a - c) for a in c1 for c in c2 if abs(a - OMEGA) > 1e-9]
    assert min(others) > 0.1                                       # no near-coincidence in the window
    d1, d2, sb = row(yb, tau2b, "$\\tau_2=\\tfrac32\\,\\tau_1$", b)
    period = 2 * (2 * np.pi / TAU1)
    assert np.allclose(np.sort(sb), OMEGA + period * np.array([-1, 0, 1]))     # 3 - 7.85, 3, 3 + 7.85
    assert np.allclose(np.diff(np.sort(sb)), period)
    fo = [folded(t) for t in (TAU1, tau2a, tau2b)]
    assert np.allclose(fo, [3 - 2 * np.pi / TAU1, 3 - 2 * np.pi / tau2a, 3 - 2 * np.pi / tau2b])
    # tick at omega on both rows
    for y in (ya, yb):
        b.append("\\node[font=\\scriptsize, anchor=north west, inner sep=1pt] at (%.3f,%.3f) {$\\omega$};" % (X(OMEGA) + 0.12, y - 0.05))
        b.append("\\node[font=\\scriptsize, anchor=north east, inner sep=1pt] at (%.3f,%.3f) {$0$};" % (X(0) - 0.04, y - 0.05))
    # notes (a)
    b.append("\\node[note, anchor=south] (g) at (%.3f,%.3f) {the only tooth common to both combs:\\\\ the true frequency $\\omega$};" % (X(OMEGA), ya + TOOTH + 0.25))
    b.append("\\draw[pointer] (g.south) -- (%.3f,%.3f);" % (X(OMEGA) + 0.18, ya + 0.22))
    b.append("\\node[note, anchor=south] (fb) at (%.3f,%.3f) {filled dots: the tooth each rate reports\\\\ on its own, in its Nyquist band (shaded)};" % (X(-5.3), ya + TOOTH + 0.25))
    b.append("\\draw[pointer] (fb.south east) to[bend left=10] (%.3f,%.3f);" % (X(fo[0]) - 0.06, ya + TOOTH + 0.06))
    b.append("\\node[note, anchor=west] at (%.3f,%.3f) {spacing $2\\pi/\\tau_1$ (up)\\\\ and $2\\pi/\\tau_2$ (down):\\\\ irrational ratio};" % (X(NU_MAX) + 0.65, ya))
    # notes (b)
    b.append("\\node[note, anchor=west] at (%.3f,%.3f) {rational ratio: the\\\\ combs coincide\\\\ every $4\\pi/\\tau_1$};" % (X(NU_MAX) + 0.65, yb))
    b.append("\\node[note, anchor=north, text=ndInk] (amb) at (%.3f,%.3f) {three candidates remain in the window, and the ambiguity persists};" % (X(OMEGA), yb - TOOTH - 0.25))
    for s in sb:
        b.append("\\draw[pointer] (amb.north) -- (%.3f,%.3f);" % (X(s) + (0.05 if s > OMEGA else -0.05 if s < OMEGA else 0.0), yb - 0.22))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) irrational ratio of the sampling periods};" % (X(2.0), ya - TOOTH - 0.55))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) rational ratio of the sampling periods};" % (X(2.0), yb - TOOTH - 1.15))
    N.write("ch05_multirate", b)
    print("written figures/ch05_multirate.tex; folded frequencies", np.round(fo, 3), "rational coincidences", np.round(np.sort(sb), 3))


if __name__ == "__main__":
    main()
