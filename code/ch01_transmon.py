"""ch01_transmon.py -- the transmon: a weakly anharmonic oscillator.

  (a) The circuit: a Josephson junction (the boxed cross, Josephson energy E_J)
      shunted by a capacitor C (charging energy E_C = e^2 / 2C), so that the
      island between them holds n Cooper pairs and the junction carries a
      phase difference phi; the control pulse u(t) reaches the island through
      a small coupling capacitor.
  (b) The potential -E_J cos(phi) (ink) against its harmonic approximation
      E_J phi^2 / 2 - E_J (dashed), with the lowest levels of
          H = 4 E_C (n - n_g)^2 - E_J cos(phi),
      computed in the basis of charge states |n>, |n| <= 30, in which cos(phi)
      moves n by one, for E_J / E_C = 50 and n_g = 0.  The levels draw
      together as they rise: hbar omega_12 = hbar omega_01 + alpha with the
      anharmonicity alpha close to -E_C, so that a pulse at omega_01 can
      address |0>, |1> and leave |2> alone, the three-level system of
      Example C.
Writes figures/ch01_transmon.tex.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

EJ, EC, NG = 50.0, 1.0, 0.0


def levels(ej=EJ, ec=EC, ng=NG, nmax=30, k=4):
    """The k lowest eigenvalues of 4 E_C (n - n_g)^2 - E_J cos(phi) in the charge basis."""
    n = np.arange(-nmax, nmax + 1)
    H = np.diag(4 * ec * (n - ng) ** 2) - ej / 2 * (np.eye(len(n), k=1) + np.eye(len(n), k=-1))
    return np.linalg.eigvalsh(H)[:k]


def capacitor_v(x, y0, y1, w=0.30, gap=0.16):
    """A vertical branch from (x, y0) to (x, y1) with a capacitor in the middle."""
    m = (y0 + y1) / 2
    return ["\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);" % (x, y0, x, m - gap / 2, x, m + gap / 2, x, y1),
            "\\draw[line width=1.4pt, ndInk] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
            % (x - w, m - gap / 2, x + w, m - gap / 2, x - w, m + gap / 2, x + w, m + gap / 2)]


def main():
    E = levels()
    w01, w12, w23 = np.diff(E)
    alpha = w12 - w01
    print("E_m / E_C:", E.round(3), " omega_01 = %.2f, omega_12 = %.2f, alpha = %.3f E_C" % (w01, w12, alpha))
    assert abs(alpha / EC + 1) < 0.2                                   # the anharmonicity is about -E_C
    assert abs(w01 - (np.sqrt(8 * EJ * EC) - EC)) < 0.2                 # omega_01 ~ sqrt(8 E_J E_C) - E_C
    assert np.ptp([levels(ng=g)[1] - levels(ng=g)[0] for g in (0.0, 0.25, 0.5)]) < 1e-3 * w01   # charge offsets hardly matter
    b = []
    # ---- (a) the circuit
    top, bot, xj, xc, xs = 2.0, 0.0, 2.75, 4.05, 0.55
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);" % (xj, top, xc, top, xs, bot, xc, bot))
    # the junction: a boxed cross
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);" % (xj, bot, xj, 0.72, xj, 1.28, xj, top))
    b.append("\\draw[ink, fill=ndSky] (%.3f,0.72) rectangle (%.3f,1.28);" % (xj - 0.28, xj + 0.28))
    b.append("\\draw[line width=1.2pt, ndInk] (%.3f,0.72) -- (%.3f,1.28) (%.3f,1.28) -- (%.3f,0.72);" % (xj - 0.28, xj + 0.28, xj - 0.28, xj + 0.28))
    b.append("\\node[anchor=east, font=\\small] at (%.3f,1.0) {$E_J$};" % (xj - 0.3))
    b += capacitor_v(xc, bot, top)
    b.append("\\node[anchor=west, font=\\small] at (%.3f,1.0) {$C$};" % (xc + 0.32))
    # the drive: a source with a pulse in it, and a coupling capacitor
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f);" % (xs, 1.32, xs, top, 1.55, top))
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (xs, 0.68, xs, bot))
    b.append("\\draw[line width=1.4pt, ndInk] (1.55,%.3f) -- (1.55,%.3f) (1.71,%.3f) -- (1.71,%.3f);" % (top - 0.24, top + 0.24, top - 0.24, top + 0.24))
    b.append("\\draw[ink] (1.71,%.3f) -- (%.3f,%.3f);" % (top, xj, top))
    b.append("\\draw[ink, fill=white] (%.3f,1.0) circle (0.32);" % xs)
    t = np.linspace(-1, 1, 80)
    b.append(N.polyline([(xs + 0.24 * s, 1.0 + 0.2 * np.exp(-4 * s * s) * np.sin(9 * s)) for s in t], "line width=0.7pt, ndRed"))
    b.append("\\node[anchor=east, text=ndRed] at (%.3f,1.0) {$u(t)$};" % (xs - 0.34))
    b.append("\\node[anchor=south, font=\\small] at (1.63,%.3f) {$C_g$};" % (top + 0.24))
    # ground
    gx = (xj + xc) / 2
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (gx, bot, gx, bot - 0.22))
    for k, w in enumerate((0.26, 0.17, 0.08)):
        b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,%.3f);" % (gx - w, bot - 0.22 - 0.08 * k, gx + w, bot - 0.22 - 0.08 * k))
    b.append("\\fill[ndInk] (%.3f,%.3f) circle (1.6pt);" % (gx, top))
    b.append("\\node[note, anchor=south] at (%.3f,%.3f) {the island:\\\\ $n$ Cooper pairs,\\\\ phase $\\varphi$};" % (gx, top + 0.08))
    b.append("\\node[font=\\small] at (2.1,-0.95) {(a) the circuit};")
    # ---- (b) the potential and its levels
    Pl = N.Plot((-np.pi, np.pi), (-52.0, 24.0), 4.6, 3.5, origin=(5.6, -0.25))
    b += Pl.axes("$\\varphi$", "energy", xticks=[(-np.pi, "$-\\pi$"), (0.0, "$0$"), (np.pi, "$\\pi$")])
    ph = np.linspace(-np.pi, np.pi, 400)
    U = -EJ * np.cos(ph)
    keep = U <= 24.0
    b.append(Pl.curve(ph[keep], U[keep], "ink"))
    ph_h = np.linspace(-1.4, 1.4, 200)
    b.append(Pl.curve(ph_h, EJ * ph_h ** 2 / 2 - EJ, "dashedink"))
    col = ("ndBlue", "ndBlue", "ndRed")
    for m, Em in enumerate(E[:3]):
        a = np.arccos(-Em / EJ)
        b.append("\\draw[line width=1.0pt, %s] %s -- %s;" % (col[m], Pl.pt(-a, Em), Pl.pt(a, Em)))
        b.append("\\node[anchor=west, font=\\small, text=%s] at %s {$\\ket{%d}$};" % (col[m], Pl.pt(a + 0.05, Em), m))
    b.append("\\draw[<->, >=stealth, line width=0.6pt, ndInk!80] %s -- %s node[midway, left, font=\\footnotesize] {$\\omega_{01}$};"
             % (Pl.pt(0.0, E[0]), Pl.pt(0.0, E[1])))
    b.append("\\draw[<->, >=stealth, line width=0.6pt, ndInk!80] %s -- %s node[midway, right, font=\\footnotesize] {$\\omega_{12}$};"
             % (Pl.pt(0.3, E[1]), Pl.pt(0.3, E[2])))
    b.append("\\node[note, anchor=east, text=ndInk!70] (hh) at %s {harmonic};" % Pl.pt(-1.2, -40.0))
    b.append("\\draw[pointer] (hh.north) to[bend left=20] %s;" % Pl.pt(-1.13, EJ * 1.13 ** 2 / 2 - EJ))
    b.append("\\node[note, anchor=west] at %s {$-E_J\\cos\\varphi$};" % Pl.pt(2.12, 21.0))
    b.append("\\node[font=\\small, align=center] at %s {(b) the levels for $E_J=%d\\,E_C$:\\\\ $\\hbar\\omega_{01}=%.1f\\,E_C$, $\\hbar\\omega_{12}=%.1f\\,E_C$};"
             % (N.pt(Pl.P(0.0, -52.0) + np.array([0.0, -0.95])), EJ / EC, w01, w12))
    N.write("ch01_transmon", b)
    print("written figures/ch01_transmon.tex")


if __name__ == "__main__":
    main()
