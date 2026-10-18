"""ch05_stroboscope.py -- what sampling cannot see.

The arrow e^{i omega t} with omega = 3 is looked at only every tau = 1.6 time
units.  Between two looks it turns by omega tau = 4.8 rad (275 degrees)
anticlockwise, which on the circle is indistinguishable from turning by
2 pi - omega tau = 1.48 rad (85 degrees) clockwise: the samples suggest a slow
backward rotation at the alias frequency omega - 2 pi / tau = -0.93.
(a) the vector and its samples, numbered, with the true and the apparent turn.
(b) the fast cos(omega t) and the slow cos((omega - 2 pi/tau) t) pass through
    exactly the same samples.  (A common damping envelope e^{-Gamma t} would
    multiply both and change nothing; it is left out to keep the waves visible.)
Writes figures/ch05_stroboscope.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import OMEGA

TAU_S = 1.6
ALIAS = OMEGA - 2 * np.pi / TAU_S
KMAX = 7


def main():
    k = np.arange(KMAX + 1)
    tk = TAU_S * k
    assert np.allclose(np.exp(1j * OMEGA * tk), np.exp(1j * ALIAS * tk))     # same samples
    assert np.allclose(np.cos(OMEGA * tk), np.cos(ALIAS * tk))
    assert np.pi / TAU_S < OMEGA                                             # above the Nyquist frequency
    b = []
    # ---- (a) the sampled arrow
    R = 1.55
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[above, text=ndInk, font=\\small] {$\\mathrm{Re}$};" % (-1.25 * R, 1.35 * R))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[right, text=ndInk, font=\\small] {$\\mathrm{Im}$};" % (-1.25 * R, 1.3 * R))
    b.append("\\draw[ink] (0,0) circle (%.3f);" % R)
    th1 = np.degrees(OMEGA * TAU_S)
    # true turn 0 -> 1: anticlockwise by omega tau (blue); apparent turn: clockwise by 2 pi - omega tau (red)
    b.append("\\draw[vec, ndBlue, line width=0.9pt] (%.3f,0) arc (0:%.2f:%.3f);" % (0.45 * R, th1, 0.45 * R))
    b.append("\\draw[vec, ndRed, line width=1.0pt, dash pattern=on 3pt off 1.5pt] (%.3f,0) arc (0:%.2f:%.3f);" % (0.78 * R, th1 - 360, 0.78 * R))
    b.append("\\draw[vec, ndInk!70, line width=0.8pt] (0,0) -- (%.3f,0);" % (0.97 * R))
    for kk in k:
        a = OMEGA * tk[kk]
        p = R * np.array([np.cos(a), np.sin(a)])
        b.append("\\node[dot, minimum size=4.2pt] at %s {};" % N.pt(p))
        aq = a if kk else np.radians(-22)
        b.append("\\node[font=\\scriptsize] at %s {$%d$};" % (N.pt((R + 0.3) * np.array([np.cos(aq), np.sin(aq)])), kk))
    b.append("\\node[note, anchor=north, text=ndBlue] at (0,%.2f) {blue: the true turn between samples,\\\\ $\\omega\\tau=%.1f$ rad anticlockwise};" % (-1.25 * R - 0.1, OMEGA * TAU_S))
    b.append("\\node[note, anchor=north, text=ndRed] at (0,%.2f) {red: the apparent turn,\\\\ $2\\pi-\\omega\\tau=%.2f$ rad clockwise};" % (-1.25 * R - 1.0, 2 * np.pi - OMEGA * TAU_S))
    yl = -1.25 * R - 2.25
    b.append("\\node[font=\\small] at (0.0,%.2f) {(a) the vector $\\ee^{\\imath\\omega t}$, observed at intervals $\\tau$};" % yl)
    # ---- (b) two waves through the same samples
    Pl = N.Plot((0, 12.0), (-1.2, 1.15), 9.0, 3.9, origin=(4.0, -1.95))
    b += Pl.axes("$t$", "", xticks=[(TAU_S * j, "$%d\\tau$" % j if j > 1 else "$\\tau$") for j in (1, 2, 3, 4, 5, 6, 7)],
                 yticks=[(-1, "$-1$"), (0, "$0$"), (1, "$1$")])
    b.append("\\draw[faint] %s -- %s;" % (Pl.pt(0, 0), Pl.pt(12.0, 0)))
    t = np.linspace(0, 12.0, 1500)
    b.append(Pl.curve(t, np.cos(OMEGA * t), "line width=0.7pt, ndBlue"))
    b.append(Pl.curve(t, np.cos(ALIAS * t), "line width=1.4pt, ndRed"))
    for kk in k:
        b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(tk[kk], -1.2), Pl.pt(tk[kk], np.cos(OMEGA * tk[kk]))))
        b.append("\\node[dot, minimum size=4.6pt] at %s {};" % Pl.pt(tk[kk], np.cos(OMEGA * tk[kk])))
    tf = 8 * np.pi / OMEGA
    b.append("\\node[note, anchor=south, text=ndBlue] (f) at %s {rapid: $\\cos\\omega t$, $\\omega=%g$};" % (Pl.pt(10.0, 1.2), OMEGA))
    b.append("\\draw[pointer, ndBlue!70] (f.west) to[bend right=15] %s;" % Pl.pt(tf + 0.05, 1.03))
    b.append("\\node[note, anchor=south, text=ndRed] (sl) at %s {slow: $\\cos(\\omega-2\\pi/\\tau)t$, $\\abs{\\omega-2\\pi/\\tau}=%.2f$};" % (Pl.pt(3.6, 1.2), abs(ALIAS)))
    b.append("\\draw[pointer, ndRed!70] (sl.south) to[bend left=10] %s;" % Pl.pt(4.25, np.cos(ALIAS * 4.25) + 0.05))
    b.append("\\node[note, anchor=west] (d) at %s {the samples (dots)\\\\ lie on both curves};" % Pl.pt(12.35, 0.55))
    b.append("\\draw[pointer] (d.west) to[bend right=15] %s;" % Pl.pt(tk[7] + 0.12, np.cos(OMEGA * tk[7]) + 0.06))
    b.append("\\node[note, anchor=west] at %s {$\\pi/\\tau=%.2f<\\omega$:\\\\ fewer than two\\\\ samples per period};" % (Pl.pt(12.35, -0.6), np.pi / TAU_S))
    b.append("\\node[font=\\small] at %s {(b) a rapid and a slow oscillation with the same samples};" % N.pt(np.array([Pl.P(6.0, 0)[0], yl])))
    N.write("ch05_stroboscope", b)
    print("written figures/ch05_stroboscope.tex; omega tau = %.3f rad, alias frequency %.4f" % (OMEGA * TAU_S, ALIAS))


if __name__ == "__main__":
    main()
