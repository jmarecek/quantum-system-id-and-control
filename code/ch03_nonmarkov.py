"""ch03_nonmarkov.py -- Markovian versus non-Markovian decay of distinguishability.

A qubit coupled resonantly to a bosonic bath with Lorentzian spectral density
J(omega) = gamma0 lambda^2 / (2 pi ((omega0 - omega)^2 + lambda^2))
is exactly solvable (Breuer--Petruccione, Sec. 10.1): the excited amplitude is
multiplied by
    G(t) = e^{-lambda t/2} [cosh(d t/2) + (lambda/d) sinh(d t/2)],  d = sqrt(lambda^2 - 2 gamma0 lambda).
The trace distance between the evolved states of |+> and |->, a measure of how
well they can be told apart, is D(t) = |G(t)|.  The time-dependent rate of the
corresponding master equation is gamma(t) = -2 d/dt log|G(t)|.
Here gamma0 = 2.  Weak coupling (lambda = 40 >> gamma0): D decays monotonically, gamma(t) >= 0.
Strong coupling (lambda = 0.4 < 2 gamma0): D revives, gamma(t) < 0 on the revivals:
information flows back from the bath.  Writes figures/ch03_nonmarkov.tex.
"""
import numpy as np
import needham as N

GAMMA0 = 2.0


def G(t, lam, g0=GAMMA0):
    d = np.sqrt(complex(lam * lam - 2 * g0 * lam))
    return np.real(np.exp(-lam * t / 2) * (np.cosh(d * t / 2) + lam / d * np.sinh(d * t / 2)))


def main():
    t = np.linspace(0, 14, 2800)
    Dm, Dn = np.abs(G(t, 40.0)), np.abs(G(t, 0.4))
    rate = -2 * np.gradient(np.log(np.maximum(Dn, 1e-12)), t)
    assert np.all(np.diff(Dm) <= 1e-12)                           # Markovian: monotone
    assert (np.diff(Dn) > 1e-6).any() and (rate < -0.1).any()   # non-Markovian: revivals, negative rate
    b = []
    Pl = N.Plot((0, 14), (0, 1.1), 7.5, 3.8)
    b += Pl.axes("$t$", "$|G(t)|$", yticks=[(0, "$0$"), (1, "$1$")])
    up = np.diff(Dn) > 0
    idx = np.where(up)[0]
    for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
        if len(seg) > 2:
            b.append("\\fill[ndRose, fill opacity=0.7] %s rectangle %s;" % (Pl.pt(t[seg[0]], 0), Pl.pt(t[seg[-1]], 1.1)))
    b.append(Pl.curve(t, Dm, "line width=1.5pt, ndBlue"))
    b.append(Pl.curve(t, Dn, "line width=1.5pt, ndRed"))
    b.append("\\node[note, anchor=south west, text=ndBlue] at %s {weak coupling: decays\\\\ monotonically (Markovian)};" % Pl.pt(4.6, 0.55))
    b.append("\\node[note, anchor=north west, text=ndRed] (r) at %s {strong coupling: revivals,\\\\ information flows back};" % Pl.pt(7.2, 1.08))
    b.append("\\node[font=\\small] at %s {(a) distinguishability $|G(t)|$ of $\\ket\\pm$};" % N.pt(Pl.P(7, 0) + np.array([0, -1.15])))
    Q = N.Plot((0, 14), (-4.2, 4.2), 7.5, 3.8, origin=(9.4, 0.0))
    b += Q.axes("$t$", "$\\gamma(t)$", yticks=[(-4, "$-4$"), (0, "$0$"), (4, "$4$")], yzero=0.0)
    r = np.clip(rate, -4.2, 4.2)
    b.append(Q.curve(t, r, "line width=1.3pt, ndRed"))
    neg = rate < 0
    idx = np.where(neg)[0]
    for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
        if len(seg) > 2:
            b.append("\\fill[ndRose, fill opacity=0.7] %s rectangle %s;" % (Q.pt(t[seg[0]], -4.2), Q.pt(t[seg[-1]], 0)))
    b.append(Q.curve(t, r, "line width=1.3pt, ndRed"))
    b.append("\\draw[line width=1.3pt, ndBlue] %s -- %s;" % (Q.pt(0, GAMMA0), Q.pt(14, GAMMA0)))
    b.append("\\node[note, anchor=south, text=ndBlue] at %s {weak coupling: $\\gamma\\approx\\gamma_0$};" % Q.pt(3.0, 2.3))
    b.append("\\node[note, anchor=north, text=ndRed] at %s {$\\gamma(t)<0$ exactly on the revivals};" % Q.pt(7.0, -4.3))
    b.append("\\node[font=\\small] at %s {(b) the rate in $\\dot\\rhoS=\\gamma(t)\\,\\mathcal D[\\sigma_-]\\rhoS$};" % N.pt(Q.P(7, -4.2) + np.array([0, -1.15])))
    N.write("ch03_nonmarkov", b)
    print("written figures/ch03_nonmarkov.tex; min rate %.2f" % rate.min())


if __name__ == "__main__":
    main()
