"""ch06_gibbs.py -- learning a Hamiltonian from its Gibbs state: temperature as excitation.

Two examples, each in thermal equilibrium at the inverse temperature beta:
  * the qubit of Chapter 5, H = -(omega/2) sigma_z, omega = 3;
  * the qubit with its defect of Chapter 4, with the frequency of Chapter 5,
        H = -(omega/2)(Z_S + Z_T) + g (s+_S s-_T + s-_S s+_T),  omega = 3, g = 1,
    whose levels are -omega (|00>), -g, g (the antisymmetric and symmetric
    single excitations) and omega (|11>).  The ground state |00> does not
    depend on g; the gap to the first excited level is Delta = omega - g = 2.
The eigenvectors do not depend on the parameter, so that the Gibbs state is
diagonal in a fixed basis and its quantum Fisher information is that of the
populations, F = beta^2 Var(dE/dtheta), attained by measuring the energy.  The
number of copies needed for a standard deviation eps = 0.01 is 1/(F eps^2).
  (a) the four levels of the pair, with discs of area proportional to the
      populations at three temperatures: hot (beta = 0.1), the best for g
      (beta = 1.07) and cold (beta = 4);
  (b) the copies needed against beta, on logarithmic axes: about 4/(beta eps)^2
      for omega and 2/(beta eps)^2 for g when hot, where the state is nearly
      I/d and shows H only at the strength beta; and about
      e^{beta Delta}/(beta eps)^2 when cold, where only the ground state is
      populated.  The least numbers are 2.0e5 copies (omega, beta = 0.80) and
      8.1e4 copies (g, beta = 1.07).
Writes figures/ch06_gibbs.tex.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import needham as N

OMEGA, G = 3.0, 1.0
EPS = 0.01
QUBIT = (np.array([-OMEGA / 2, OMEGA / 2]), np.array([-0.5, 0.5]))       # levels and dE/domega
PAIR = (np.array([-OMEGA, -G, G, OMEGA]), np.array([0.0, -1.0, 1.0, 0.0]))  # levels and dE/dg


def populations(E, beta):
    p = np.exp(-beta * (E - E.min()))
    return p / p.sum()


def copies(model, beta):
    """Copies of the Gibbs state needed for a standard deviation EPS, 1/(F EPS^2)."""
    E, dE = model
    p = populations(E, beta)
    F = beta ** 2 * (p @ (dE - p @ dE) ** 2)
    return 1.0 / (F * EPS ** 2)


def best(model):
    res = minimize_scalar(lambda lb: copies(model, np.exp(lb)), bounds=(-3.0, 2.0), method="bounded")
    return np.exp(res.x), res.fun


def main():
    bq, nq = best(QUBIT)
    bp, n_p = best(PAIR)
    assert abs(bq - 0.800) < 0.005 and abs(nq / 2.049e5 - 1) < 0.01
    assert abs(bp - 1.071) < 0.005 and abs(n_p / 8.113e4 - 1) < 0.01
    # the asymptotes: (beta eps)^2 N -> 4 and 2 when hot, -> e^{beta Delta} when cold
    assert abs(copies(QUBIT, 1e-3) * (1e-3 * EPS) ** 2 / 4 - 1) < 1e-3
    assert abs(copies(PAIR, 1e-3) * (1e-3 * EPS) ** 2 / 2 - 1) < 1e-3
    assert abs(copies(PAIR, 12.0) * (12.0 * EPS) ** 2 * np.exp(-12.0 * (OMEGA - G)) - 1) < 1e-3
    assert abs(copies(QUBIT, 12.0) * (12.0 * EPS) ** 2 * np.exp(-12.0 * OMEGA) - 1) < 1e-3

    b = []
    # ---- (a) the ladder of the pair, with the populations as discs
    ylev = lambda E: 2.35 + 0.68 * E
    x0, x1 = 1.35, 6.75
    cols = [(0.1, 2.35, "hot", "ndRed"), (bp, 4.0, "best for $g$", "ndGreen"), (4.0, 5.65, "cold", "ndBlue")]
    E, _ = PAIR
    for e, lab in zip(E, ["-\\omega", "-g", "g", "\\omega"]):
        b.append("\\draw[ink] (%.2f,%.2f) -- (%.2f,%.2f);" % (x0, ylev(e), x1, ylev(e)))
        b.append("\\node[font=\\small, anchor=east] at (%.2f,%.2f) {$%s$};" % (x0 - 0.05, ylev(e), lab))
    b.append("\\draw[faint, ->, >=stealth] (0.55,%.2f) -- (0.55,%.2f) node[above, text=ndInk, font=\\small] {$E$};" % (ylev(-3.4), ylev(3.55)))
    for beta, x, name, col in cols:
        p = populations(E, beta)
        for e, q in zip(E, p):
            r = 0.40 * np.sqrt(q)
            if r > 0.012:
                b.append("\\fill[%s, fill opacity=0.8] (%.2f,%.2f) circle (%.3f);" % (col, x, ylev(e), r))
        b.append("\\node[font=\\scriptsize, align=center, text=%s] at (%.2f,%.2f) {%s\\\\ $\\beta=%s$};"
                 % (col, x, ylev(3.0) + 0.62, name, ("%.2f" % beta) if name.startswith("best") else ("%g" % beta)))
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {the ground state $\\ket{00}$ does not depend on $g$;\\\\ in the cold, nothing else is populated};"
             % ((x0 + x1) / 2 + 0.1, ylev(-3.0) - 0.22))
    b.append("\\draw[ndGold, line width=0.9pt, <->, >=stealth] (%.2f,%.2f) -- (%.2f,%.2f);" % (x1 + 0.22, ylev(-G), x1 + 0.22, ylev(G)))
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndGold] at (%.2f,%.2f) {$2g$};" % (x1 + 0.28, ylev(0)))
    b.append("\\node[font=\\small] at (%.2f,-1.05) {(a) the levels of the pair and their populations};" % ((x0 + x1) / 2 - 0.2))
    # ---- (b) copies needed against beta
    lx = (np.log10(0.05), np.log10(8.0))
    ly = (4.0, 10.0)
    Q = N.Plot(lx, ly, 6.6, 4.6, origin=(9.5, 0.0))
    b += Q.axes("$\\beta$", "copies", xticks=[(np.log10(v), "$%g$" % v) for v in (0.1, 1, 5)],
                yticks=[(k, "$10^{%d}$" % k) for k in (4, 6, 8, 10)])
    bb = np.logspace(lx[0], lx[1], 300)
    for model, col in [(QUBIT, "ndBlue"), (PAIR, "ndRed")]:
        n = np.array([copies(model, v) for v in bb])
        keep = np.log10(n) <= ly[1]
        b.append(Q.curve(np.log10(bb[keep]), np.log10(n[keep]), "line width=1.2pt, %s" % col))
    for beta, n, col in [(bq, nq, "ndBlue"), (bp, n_p, "ndRed")]:
        b.append("\\node[dot, fill=%s, minimum size=4.6pt] at %s {};" % (col, Q.pt(np.log10(beta), np.log10(n))))
    b.append("\\node[note, anchor=west, text=ndBlue] (q) at %s {$\\omega$ of the qubit: least,\\\\ $2\\cdot10^5$ at $\\beta=%.1f$};" % (Q.pt(-1.27, 9.55), bq))
    b.append("\\draw[pointer] (q.south) to[bend right=12] %s;" % Q.pt(np.log10(bq) - 0.02, np.log10(nq) + 0.12))
    b.append("\\node[note, anchor=west, text=ndRed] (p) at %s {$g$ of the pair: least,\\\\ $8\\cdot10^4$ at $\\beta=%.2f$};" % (Q.pt(0.13, 4.45), bp))
    b.append("\\draw[pointer] (p.north west) to[bend left=12] %s;" % Q.pt(np.log10(bp) + 0.01, np.log10(n_p) - 0.1))
    b.append("\\node[note, anchor=west] (h) at %s {hot: about\\\\ $4/(\\beta\\eps)^2$, $2/(\\beta\\eps)^2$};" % Q.pt(-1.27, 4.95))
    b.append("\\draw[pointer] (h.north) to[bend left=10] %s;" % Q.pt(-1.1, np.log10(copies(PAIR, 10 ** -1.1)) - 0.1))
    b.append("\\node[note, anchor=east] (c) at %s {cold: about\\\\ $\\ee^{\\beta\\Delta}/(\\beta\\eps)^2$};" % Q.pt(0.5, 8.4))
    b.append("\\draw[pointer] (c.south) to[bend right=12] %s;" % Q.pt(np.log10(2.8), np.log10(copies(QUBIT, 2.8)) + 0.05))
    b.append("\\node[font=\\small, align=center] at %s {(b) copies for a standard deviation of $%g$};" % (N.pt(Q.P(np.mean(lx), ly[0]) + np.array([0.0, -1.05])), EPS))
    N.write("ch06_gibbs", b)
    print("written figures/ch06_gibbs.tex; least copies: omega %.3g at beta %.3f, g %.3g at beta %.3f" % (nq, bq, n_p, bp))
    for v in (0.1, 4.0):
        print("  beta = %g: populations of the pair" % v, np.round(populations(PAIR[0], v), 4),
              " copies omega %.3g, g %.3g" % (copies(QUBIT, v), copies(PAIR, v)))


if __name__ == "__main__":
    main()
