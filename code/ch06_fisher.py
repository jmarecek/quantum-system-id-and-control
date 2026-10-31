"""ch06_fisher.py -- when to measure: the Fisher information of one shot.

The qubit of Chapter 5, prepared in X0 (ch06_shots).  A shot of sigma_x at the
time t has outcomes +-1 with mean y(t) = a e^{-Gamma t} cos(omega t),
a = sin(2 pi/3); its Fisher information about a parameter theta is
    I_theta(t) = (d y / d theta)^2 / (1 - y^2).
  (a) y(t) for omega = 3 and for omega = 3.15: a measurement is informative
      about omega where the two curves differ;
  (b) I_omega and I_Gamma for sigma_x, and I_gamma1 for a shot of sigma_z,
      whose mean is z(t) = 1 - (1 - z0) e^{-gamma_1 t}; each is scaled to a
      maximum of 1.  The envelope a^2 t^2 e^{-2 Gamma t} of I_omega is greatest
      at t = 1/Gamma; omega is best measured where y crosses zero, Gamma near the
      extremes of y.  The squared sensitivity of z to gamma_1 is greatest at
      t = 1/gamma_1, but the shot noise 1 - z^2 shrinks as z approaches 1, so
      that I_gamma1 is greatest later.
Writes figures/ch06_fisher.tex.
"""
import numpy as np
import needham as N
from ch06_shots import expect, X0, CX, CZ, OMEGA, GAMMA, GAMMA1, GPHI

DW = 0.3


def main():
    a, z0 = X0[0], X0[2]
    t = np.linspace(1e-3, 14, 5601)
    y = a * np.exp(-GAMMA * t) * np.cos(OMEGA * t)
    assert np.allclose(y[::400], expect(np.array([OMEGA, GAMMA1, GPHI]), CX, X0, t[::400]))
    I_w = (a * t * np.exp(-GAMMA * t) * np.sin(OMEGA * t)) ** 2 / (1 - y ** 2)
    I_G = (t * y) ** 2 / (1 - y ** 2)
    z = 1 - (1 - z0) * np.exp(-GAMMA1 * t)
    assert np.allclose(z[::400], expect(np.array([OMEGA, GAMMA1, GPHI]), CZ, X0, t[::400]))
    sens = ((1 - z0) * t * np.exp(-GAMMA1 * t)) ** 2
    I_g = sens / (1 - z ** 2)
    env = a ** 2 * t ** 2 * np.exp(-2 * GAMMA * t)
    t_env = t[np.argmax(env)]
    assert abs(t_env - 1 / GAMMA) < 0.01                                   # the envelope peaks at 1/Gamma
    assert abs(t[np.argmax(sens)] - 1 / GAMMA1) < 0.01                     # sensitivity peaks at 1/gamma_1
    tw, tG, tg = t[np.argmax(I_w)], t[np.argmax(I_G)], t[np.argmax(I_g)]
    assert abs(np.cos(OMEGA * tw)) < 0.1 and abs(abs(np.cos(OMEGA * tG)) - 1) < 0.02
    assert tg > 1.5 / GAMMA1
    b = []
    # ---- (a) two values of omega
    tt = np.linspace(0, 6, 601)
    y0 = a * np.exp(-GAMMA * tt) * np.cos(OMEGA * tt)
    y1 = a * np.exp(-GAMMA * tt) * np.cos((OMEGA + DW) * tt)
    Pl = N.Plot((0, 6), (-0.9, 0.95), 5.6, 4.6)
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\small, text=ndInk] {$t$};" % (Pl.pt(0, 0), Pl.pt(6.3, 0)))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\small, text=ndInk] {$\\langle\\sigma_x\\rangle$};" % (Pl.pt(0, -0.9), Pl.pt(0, 1.0)))
    for v in (-0.5, 0.5):
        b.append("\\draw[faint] %s -- %s node[left, font=\\scriptsize, text=ndInk] {$%g$};" % (Pl.pt(0.08, v), Pl.pt(-0.08, v), v))
    poly = [Pl.P(s, v) for s, v in zip(tt, y0)] + [Pl.P(s, v) for s, v in zip(tt[::-1], y1[::-1])]
    b.append("\\fill[ndGold, fill opacity=0.45] " + " -- ".join(N.pt(p) for p in poly) + " -- cycle;")
    b.append(Pl.curve(tt, y0, "line width=1.1pt, ndBlue"))
    b.append(Pl.curve(tt, y1, "line width=0.8pt, ndBlue!60, dash pattern=on 3pt off 1.5pt"))
    b.append("\\node[note, anchor=west] (s) at %s {shaded: the difference made\\\\ by a change of $\\omega$ from\\\\ $%g$ to $%g$ (dashed); it grows\\\\ with $t$ until the decay\\\\ overtakes it};" % (Pl.pt(1.75, 0.72), OMEGA, OMEGA + DW))
    b.append("\\draw[pointer] (s.south west) to[bend right=20] %s;" % Pl.pt(1.62, 0.05))
    b.append("\\node[font=\\small] at %s {(a) $\\langle\\sigma_x\\rangle$ for two values of $\\omega$};" % N.pt(Pl.P(3, -0.9) + np.array([0.0, -0.55])))
    # ---- (b) one shot of sigma_x: omega and Gamma
    Q = N.Plot((0, 7), (0, 1.12), 5.6, 1.75, origin=(7.0, 2.85))
    b += Q.axes("$t$", "", xticks=[(1 / GAMMA, "$\\frac1\\Gamma$"), (5, "$5$")], yticks=[(0, "$0$"), (1, "$1$")])
    tb = t < 7
    b.append(Q.curve(t[tb], env[tb] / I_w.max(), "dashedink"))
    b.append(Q.curve(t[tb], I_G[tb] / I_G.max(), "line width=0.8pt, ndGold"))
    b.append(Q.curve(t[tb], I_w[tb] / I_w.max(), "line width=0.9pt, ndBlue"))
    for tm, col in [(tw, "ndBlue"), (tG, "ndGold")]:
        b.append("\\node[dot, fill=%s, minimum size=4.2pt] at %s {};" % (col, Q.pt(tm, 1.0)))
    xr = Q.P(7.5, 0)[0]
    b.append("\\node[note, anchor=north west, text=ndBlue] at (%.2f,%.2f) {blue, $\\omega$: greatest where\\\\ $\\langle\\sigma_x\\rangle$ crosses zero, near\\\\ $t=1/\\Gamma$ (dashed: $a^2t^2\\ee^{-2\\Gamma t}$)};" % (xr, Q.P(0, 1.15)[1]))
    b.append("\\node[note, anchor=north west, text=ndGold!80!ndInk] at (%.2f,%.2f) {gold, $\\Gamma$: greatest near\\\\ the extremes of $\\langle\\sigma_x\\rangle$};" % (xr, Q.P(0, 0.42)[1]))
    b.append("\\node[font=\\scriptsize, anchor=east] at %s {(b)};" % Q.pt(-0.9, 1.0))
    # ---- (c) one shot of sigma_z: gamma_1
    R = N.Plot((0, 14), (0, 1.12), 5.6, 1.75, origin=(7.0, 0.0))
    b += R.axes("$t$", "", xticks=[(1 / GAMMA1, "$\\frac1{\\gamma_1}$"), (10, "$10$")], yticks=[(0, "$0$"), (1, "$1$")])
    b.append(R.curve(t, sens / sens.max(), "dashedink"))
    b.append(R.curve(t, I_g / I_g.max(), "line width=1.1pt, ndRed"))
    b.append("\\node[dot, fill=ndRed, minimum size=4.2pt] at %s {};" % R.pt(tg, 1.0))
    b.append("\\node[note, anchor=north west, text=ndRed] at (%.2f,%.2f) {red, $\\gamma_1$ from $\\sigma_z$: greatest\\\\ at $t=%.1f$; the sensitivity alone\\\\ (dashed) is greatest at $1/\\gamma_1$,\\\\ but the noise $1-z^2$ shrinks\\\\ as $z$ approaches $1$};" % (xr, R.P(0, 1.2)[1], tg))
    b.append("\\node[font=\\scriptsize, anchor=east] at %s {(c)};" % R.pt(-1.8, 1.0))
    b.append("\\node[font=\\small] at %s {(b), (c) the Fisher information of one shot of $\\sigma_x$ and of $\\sigma_z$, each scaled};" % N.pt(R.P(10.5, 0) + np.array([0.0, -0.75])))
    N.write("ch06_fisher", b)
    print("written figures/ch06_fisher.tex; maxima at t = %.2f (omega), %.2f (Gamma), %.2f (gamma_1); envelope %.2f"
          % (tw, tG, tg, t_env))


if __name__ == "__main__":
    main()
