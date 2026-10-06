"""ch03_spectrum.py -- the spectrum of a Lindbladian.

The eigenvalues of the vectorised generator of the qubit of Section 3.2
(omega = 2, gamma_1 = 0.4, gamma_phi = 0.3) are exactly
    0 (the steady state),  -1/T_1,  -1/T_2 +- i omega,
and those of a random two-qubit Lindbladian (grey) all lie in the closed left
half-plane, in complex-conjugate pairs.  Writes figures/ch03_spectrum.tex.
"""
import numpy as np
import tikzexport as T
import needham as N
from ch03_decoherence import lindbladian

I, X, Y, Z = T.pauli()


def random_lindbladian(n, rng, k=3):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = 0.2 * (A + A.conj().T)
    jumps = [0.35 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) for _ in range(k)]
    return lindbladian(H, jumps)


def main():
    omega, g1, gphi = 2.0, 0.4, 0.3
    T1, T2 = 1 / g1, 1 / (g1 / 2 + gphi)
    L = lindbladian(-0.5 * omega * Z, [np.sqrt(g1) * np.array([[0, 1], [0, 0]], complex), np.sqrt(gphi / 2) * Z])
    ev = np.linalg.eigvals(L)
    expected = np.array([0, -1 / T1, -1 / T2 + 1j * omega, -1 / T2 - 1j * omega])
    assert all(np.min(abs(ev - e)) < 1e-10 for e in expected)
    ev2 = np.linalg.eigvals(random_lindbladian(4, np.random.default_rng(5)))
    assert ev2.real.max() < 1e-9
    allz = np.concatenate([ev, ev2])
    xm, ym = 1.15 * abs(allz.real).max() + 0.2, 1.15 * abs(allz.imag).max() + 0.2
    s = 2.4 / ym
    P = lambda z: (s * z.real, s * z.imag)
    b = []
    b.append("\\fill[ndSky, fill opacity=0.4] (%.2f,%.2f) rectangle (0,%.2f);" % (-s * xm, -s * ym, s * ym))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[right, text=ndInk] {$\\mathrm{Re}\\,\\mu$};" % (-s * xm - 0.2, 1.2))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[above, text=ndInk] {$\\mathrm{Im}\\,\\mu$};" % (-s * ym - 0.1, s * ym + 0.3))
    for z in ev2:
        b.append("\\fill[ndInk!40] %s circle (1.6pt);" % N.pt(P(z)))
    labels = {0: ("$0$: steady state", "above right"), 1: ("$-1/T_1$", "below"),
              2: ("$-1/T_2+\\imath\\omega$", "left"), 3: ("$-1/T_2-\\imath\\omega$", "left")}
    for k, z in enumerate(expected):
        b.append("\\node[dot, fill=ndRed, minimum size=5pt, label={[font=\\scriptsize, text=ndRed]%s:%s}] at %s {};"
                 % (labels[k][1], labels[k][0], N.pt(P(z))))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {$-\\mathrm{Re}\\,\\mu$: decay rates\\\\ $\\mathrm{Im}\\,\\mu$: oscillation frequencies};" % (1.5, 1.6))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {red: the qubit of\\\\ the example; grey: a\\\\ random two-qubit\\\\ Lindbladian};" % (1.5, -1.4))
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {all eigenvalues in the left half-plane};" % (-s * xm / 2, -s * ym - 0.05))
    N.write("ch03_spectrum", b)
    print("written figures/ch03_spectrum.tex; random spectrum Re range [%.2f, %.2f]" % (ev2.real.min(), ev2.real.max()))


if __name__ == "__main__":
    main()
