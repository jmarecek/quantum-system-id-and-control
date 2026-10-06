"""ch03_pseudomode.py -- Markovian embedding of a non-Markovian qubit.

The qubit of ch03_nonmarkov.py, coupled to a bath with Lorentzian spectral
density of width lambda, is reproduced exactly by an enlarged *Markovian*
system: the qubit coupled with strength Omega = sqrt(gamma0 lambda / 2) to a
single damped cavity mode (the pseudomode), which leaks into a flat bath at
rate kappa = 2 lambda.  We simulate the GKSL equation of qubit + mode
(mode truncated at 3 excitations) and check that the excited population of the
qubit equals the exact |G(t)|^2.  Writes figures/ch03_pseudomode.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N
from ch03_decoherence import lindbladian, vec, unvec
from ch03_nonmarkov import G, GAMMA0

LAM = 0.4
NMAX = 4                                            # mode levels 0..3


def embedded_population(times):
    a = np.diag(np.sqrt(np.arange(1, NMAX)), 1)      # annihilation operator
    sm = np.array([[0, 0], [1, 0]], complex)        # |g><e| with |e> = (1,0), |g> = (0,1)
    Om = np.sqrt(GAMMA0 * LAM / 2)
    H = Om * (np.kron(sm.conj().T, a) + np.kron(sm, a.conj().T))
    L = lindbladian(H, [np.sqrt(2 * LAM) * np.kron(np.eye(2), a)])
    psi = np.zeros(2 * NMAX, complex); psi[0] = 1.0  # |e> (x) |0>
    rho0 = np.outer(psi, psi.conj())
    Pe = np.kron(np.diag([1.0, 0.0]), np.eye(NMAX))
    return np.array([np.real(np.trace(Pe @ unvec(expm(L * t) @ vec(rho0), 2 * NMAX))) for t in times])


def main():
    t = np.linspace(0, 14, 700)
    exact = G(t, LAM) ** 2
    ts = np.linspace(0, 14, 36)
    sim = embedded_population(ts)
    assert np.allclose(sim, G(ts, LAM) ** 2, atol=1e-8)
    b = []
    # (a) schematic: qubit -- mode -- flat bath
    q, m = np.array([0.0, 0.0]), np.array([3.0, 0.0])
    b.append("\\shade[ball color=ndSky!80!white] %s circle (0.55);" % N.pt(q))
    b.append("\\draw[ink] %s circle (0.55);" % N.pt(q))
    b.append("\\node at %s {qubit};" % N.pt(q + (0, -0.85)))
    for dx in (-0.75, 0.75):                          # the two mirrors of the cavity
        b.append("\\draw[line width=2.2pt, ndInk] %s -- %s;" % (N.pt(m + (dx, -0.6)), N.pt(m + (dx, 0.6))))
    s = np.linspace(-0.75, 0.75, 60)
    b.append(N.polyline([m + (x, 0.35 * np.cos(np.pi * x / 1.5)) for x in s], "line width=1.2pt, ndGold"))
    b.append(N.polyline([m + (x, -0.35 * np.cos(np.pi * x / 1.5)) for x in s], "line width=1.2pt, ndGold"))
    b.append("\\node at %s {pseudomode};" % N.pt(m + (0, -0.95)))
    b.append("\\draw[<->, >=stealth, line width=0.9pt, ndBlue] %s -- %s node[midway, above, font=\\footnotesize] {$g_{\\rm p}$};" % (N.pt(q + (0.6, 0)), N.pt(m + (-0.85, 0))))
    w = np.linspace(0, 1.3, 60)
    b.append(N.polyline([m + (0.85 + x, 0.12 * np.sin(14 * x)) for x in w], "->, >=stealth, line width=0.9pt, ndRed"))
    b.append("\\node[font=\\footnotesize, text=ndRed, above] at %s {$\\kappa=2\\lambda$};" % N.pt(m + (1.5, 0.15)))
    b.append("\\node[note, anchor=north] at %s {flat bath:\\\\ Markovian};" % N.pt(m + (2.2, -0.35)))
    b.append("\\node[note, anchor=south] at %s {qubit $+$ mode obey a GKSL equation};" % N.pt((1.6, 1.05)))
    b.append("\\node[font=\\small] at (1.8,-2.15) {(a) the embedding};")
    # (b) spectral density
    Pj = N.Plot((-4, 4), (0, 1.15), 4.6, 3.0, origin=(6.4, -1.0))
    om = np.linspace(-4, 4, 300)
    J = LAM ** 2 / (om ** 2 + LAM ** 2)
    b += Pj.axes("$\\omega-\\omega_0$", "$J(\\omega)$")
    b.append("\\fill[ndRose, fill opacity=0.6] %s -- %s -- %s -- cycle;" % (Pj.pt(-4, 0), " -- ".join(Pj.pt(x, y) for x, y in zip(om, J)), Pj.pt(4, 0)))
    b.append(Pj.curve(om, J, "line width=1.4pt, ndRed"))
    b.append("\\draw[line width=1.2pt, ndBlue, dash pattern=on 4pt off 2pt] %s -- %s;" % (Pj.pt(-4, 0.25), Pj.pt(4, 0.25)))
    b.append("\\draw[<->, >=stealth, ndInk, line width=0.5pt] %s -- %s node[midway, above, font=\\scriptsize] {$2\\lambda$};" % (Pj.pt(-LAM, 0.5), Pj.pt(LAM, 0.5)))
    b.append("\\node[note, anchor=west, text=ndBlue] at %s {flat: Markov};" % Pj.pt(1.2, 0.33))
    b.append("\\node[note, anchor=west, text=ndRed] at %s {Lorentzian:\\\\ memory $\\sim1/\\lambda$};" % Pj.pt(1.0, 0.85))
    b.append("\\node[font=\\small] at %s {(b) spectral density};" % N.pt(Pj.P(0, 0) + np.array([0, -1.15])))
    # (c) excited population
    Pp = N.Plot((0, 14), (0, 1.05), 5.6, 3.0, origin=(13.6, -1.0))
    b += Pp.axes("$t$", "$P_e(t)$", yticks=[(0, "$0$"), (1, "$1$")])
    b.append(Pp.curve(t, np.exp(-GAMMA0 * t), "line width=1.2pt, ndBlue, dash pattern=on 4pt off 2pt"))
    b.append(Pp.curve(t, exact, "line width=1.5pt, ndRed"))
    for x, y in zip(ts, sim):
        b.append("\\fill[ndInk] %s circle (1.1pt);" % Pp.pt(x, y))
    b.append("\\node[note, anchor=south west, text=ndRed] at %s {exact: revivals};" % Pp.pt(5.0, 0.22))
    b.append("\\node[note, anchor=south west] at %s {dots: the embedded\\\\ GKSL simulation};" % Pp.pt(5.4, 0.55))
    b.append("\\node[note, anchor=south west, text=ndBlue] at %s {Markov: $\\ee^{-\\gamma_0t}$};" % Pp.pt(1.0, 0.85))
    b.append("\\node[font=\\small] at %s {(c) excited population};" % N.pt(Pp.P(7, 0) + np.array([0, -1.15])))
    N.write("ch03_pseudomode", b)
    print("written figures/ch03_pseudomode.tex; max |sim - exact| = %.1e" % abs(sim - G(ts, LAM) ** 2).max())


if __name__ == "__main__":
    main()
