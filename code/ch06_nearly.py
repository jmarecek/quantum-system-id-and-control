"""ch06_nearly.py -- weakly excited, weakly seen, nearly merged.

Chapter 5 found three ways in which a mode is lost from the data: it is not
excited, not seen, or merged with another mode by the sampling.  Here the
same three are followed continuously.  For the qubit of Chapter 5, with the
Hankel set-up of ch05_hankel (K = 60 samples, R = 10, S = 40), the smallest
non-zero singular value sigma_4 of H(0), relative to sigma_1, is plotted
against
  (a) the tilt phi of the preparation from |1> in the x-z plane (measured
      along n, tau = 0.25); it vanishes at phi = 0, where the precession is
      not excited;
  (b) the angle psi of the measurement axis from z in the x-z plane (prepared
      in X0, tau = 0.25); it vanishes at psi = 0 (sigma_z does not see the
      precession) and at psi = 90 degrees (sigma_x does not see the
      relaxation);
  (c) the sampling interval tau near pi/omega (X0, measured along n); it
      vanishes at tau = pi/omega, where the two modes of the precession merge.
Near each zero sigma_4 is proportional to the distance from it, and since the
noise floor falls like 1/sqrt(N), the shots needed grow like 1/sigma_4^2.
Writes figures/ch06_nearly.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N
from ch05_spectrum import qubit_A, hankel, X0, CN, TAU, OMEGA
from ch06_shots import tilted

K, R, S = 60, 10, 40
LO, HI = -3.5, -0.6


def ratio(x0, C, tau):
    """sigma_4 / sigma_1 of the Hankel matrix H(0) of the data C e^{A k tau} x0."""
    G = expm(qubit_A() * tau)
    y = np.array([(C @ np.linalg.matrix_power(G, k) @ x0).item() for k in range(K)])
    s = np.linalg.svd(hankel(y[:, None], R, S), compute_uv=False)
    assert s[4] < 1e-12 * s[0]                      # never more than four modes
    return s[3] / s[0]


def readout(psi):
    return np.array([[np.sin(psi), 0.0, np.cos(psi), 0.0]])


def panel(b, x0p, xlim, xs, vals, xticks, xlabel, ylabel, zeros, title, notes, mark=None):
    Q = N.Plot(xlim, (LO, HI), 4.1, 4.2, origin=(x0p, 0.0))
    b += Q.axes(xlabel, ylabel, xticks=xticks, yticks=[(k, "$10^{%d}$" % k) for k in (-3, -2, -1)])
    lv = np.log10(np.clip(vals, 1e-30, None))
    keep = lv > LO                                    # leave out what falls below the axis
    idx = np.where(keep)[0]
    for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
        if len(seg) > 1:
            b.append(Q.curve(xs[seg], lv[seg], "line width=1.2pt, ndBlue"))
    for z in zeros:
        b.append("\\draw[line width=0.7pt, ndRed, fill=white] %s circle (2.2pt);" % Q.pt(z, LO))
    if mark is not None:
        b.append("\\node[dot, fill=ndInk, minimum size=4.5pt] at %s {};" % Q.pt(*mark))
    for (x, y, txt, anchor, tgt) in notes:
        b.append("\\node[note, anchor=%s] (n) at %s {%s};" % (anchor, Q.pt(x, y), txt))
        if tgt is not None:
            b.append("\\draw[pointer] (n) to[bend left=10] %s;" % Q.pt(*tgt))
    b.append("\\node[font=\\small, align=center] at %s {%s};" % (N.pt(Q.P(0.5 * (xlim[0] + xlim[1]), LO) + np.array([0.0, -0.95])), title))
    return Q


def main():
    deg = np.radians
    # (a) excitation
    phis = np.r_[np.geomspace(0.005, 1.0, 40), np.linspace(1.0, 90.0, 300)[1:]]
    ra = np.array([ratio(tilted(deg(p)), CN, TAU) for p in phis])
    assert ratio(tilted(0.0), CN, TAU) < 1e-12                               # not excited
    assert abs(ratio(tilted(deg(1.0)), CN, TAU) / ratio(tilted(deg(5.0)), CN, TAU) - 0.2) < 0.01   # linear near 0
    # (b) observation
    psis = np.r_[np.geomspace(0.005, 1.0, 40), np.linspace(1.0, 89.0, 300)[1:-1], 90.0 - np.geomspace(1.0, 0.005, 40)]
    rb = np.array([ratio(X0, readout(deg(p)), TAU) for p in psis])
    assert ratio(X0, readout(0.0), TAU) < 1e-12 and ratio(X0, readout(deg(90.0)), TAU) < 1e-12
    # (c) sampling
    tc = np.pi / OMEGA
    taus = np.r_[np.linspace(0.75, tc - 0.02, 200), tc - np.geomspace(0.02, 1e-4, 40), tc + np.geomspace(1e-4, 0.02, 40), np.linspace(tc + 0.02, 1.35, 200)]
    rc = np.array([ratio(X0, CN, t) for t in taus])
    assert ratio(X0, CN, tc) < 1e-12                                        # merged
    assert np.allclose(X0, tilted(deg(60.0)))
    ex = ratio(X0, CN, TAU)
    print("sigma_4/sigma_1 for the example: %.4f; at phi = 15: %.4f" % (ex, ratio(tilted(deg(15)), CN, TAU)))

    b = []
    panel(b, 0.0, (0.0, 90.0), phis, ra, [(0, "$0^\\circ$"), (30, "$30^\\circ$"), (60, "$60^\\circ$"), (90, "$90^\\circ$")],
          "$\\phi$", "$\\sigma_4/\\sigma_1$", [0.0], "(a) the tilt $\\phi$ of the preparation:\\\\ weakly excited",
          [(35.0, -3.0, "$\\phi=0$, the state $\\ket1$:\\\\ the precession is\\\\ not excited", "south west", (2.5, LO + 0.05))],
          mark=(60.0, np.log10(ex)))
    panel(b, 5.6, (0.0, 90.0), psis, rb, [(0, "$0^\\circ$"), (45, "$45^\\circ$"), (90, "$90^\\circ$")],
          "$\\psi$", "", [0.0, 90.0], "(b) the axis $\\psi$ of the measurement:\\\\ weakly seen",
          [(45.0, -2.45, "$\\sigma_z$ does not\\\\ see the precession,\\\\ $\\sigma_x$ the relaxation", "north", None)],
          mark=(45.0, np.log10(ex)))
    panel(b, 11.2, (0.75, 1.35), taus, rc, [(0.8, "$0.8$"), (tc, "$\\frac\\pi\\omega$"), (1.3, "$1.3$")],
          "$\\tau$", "", [tc], "(c) the sampling interval $\\tau$:\\\\ nearly merged",
          [(0.77, -3.35, "two modes\\\\ merge at\\\\ $\\tau=\\pi/\\omega$", "south west", (tc - 0.012, LO + 0.06))])
    b.append("\\node[note, anchor=north] at (8.0,-1.55) {dots: the example ($\\phi=60^\\circ$, $\\psi=45^\\circ$, $\\tau=0.25$);"
             " the shots needed for the weakest mode grow like $(\\sigma_1/\\sigma_4)^2$};")
    N.write("ch06_nearly", b)
    print("written figures/ch06_nearly.tex")


if __name__ == "__main__":
    main()
