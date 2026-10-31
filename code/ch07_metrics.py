"""ch07_metrics.py -- four measures of distance between qubit states along a path.

We fix rho_0 = |0><0| and move rho(s) along the straight segment of Bloch
vectors (0, 0, 1 - 2s), s in [0, 1], from |0> through the maximally mixed
state (s = 1/2) to |1>.  Along it the trace distance is s, the Hilbert--Schmidt
distance sqrt(2) s, the fidelity 1 - s, and the Bures angle arccos sqrt(1 - s),
whose slope is infinite at both ends, where the states are pure.
Writes figures/ch07_metrics.tex.
"""
import numpy as np
from scipy.linalg import sqrtm
import needham as N


def state(r):
    """Density matrix with Bloch vector r, |r| <= 1."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return 0.5 * (np.eye(2) + r[0] * X + r[1] * Y + r[2] * Z)


def fidelity(rho, sigma):
    """Uhlmann fidelity F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    s = sqrtm(rho)
    return np.real(np.trace(sqrtm(s @ sigma @ s))) ** 2


def trace_distance(rho, sigma):
    """D = (1/2) ||rho - sigma||_1, half the sum of |eigenvalues|."""
    return 0.5 * np.abs(np.linalg.eigvalsh(rho - sigma)).sum()


def hs_distance(rho, sigma):
    """Hilbert--Schmidt distance sqrt(Tr (rho - sigma)^2)."""
    return np.sqrt(np.real(np.trace((rho - sigma) @ (rho - sigma))))


def bures_angle(rho, sigma):
    """Bures angle arccos sqrt(F)."""
    return np.arccos(np.clip(np.sqrt(fidelity(rho, sigma)), 0, 1))


def main():
    rho0 = state([0, 0, 1])
    s = np.linspace(0, 1, 401)
    rhos = [state([0, 0, 1 - 2 * si]) for si in s]
    D = np.array([trace_distance(rho0, r) for r in rhos])
    HS = np.array([hs_distance(rho0, r) for r in rhos])
    F = np.array([fidelity(rho0, r) for r in rhos])
    A = np.array([bures_angle(rho0, r) for r in rhos])
    assert np.allclose(D, s) and np.allclose(HS, np.sqrt(2) * s) and np.allclose(F, 1 - s, atol=1e-7)
    assert np.allclose(A, np.arccos(np.sqrt(1 - s)), atol=1e-6)
    assert abs(A[200] - np.pi / 4) < 1e-6 and abs(A[-1] - np.pi / 2) < 1e-6   # halfway at 1/2 Id, pi/2 at |1>
    print("at s = 1/2: D = %.3f, HS = %.3f, F = %.3f, Bures angle = %.3f (pi/4); at s = 1: Bures angle %.4f (pi/2)"
          % (D[200], HS[200], F[200], A[200], A[-1]))

    b = []
    Pl = N.Plot((0.0, 1.0), (0.0, 1.65), 6.4, 4.6)
    b += Pl.axes("$s$", "", xticks=[(0.5, "$\\tfrac12$"), (1.0, "$1$")],
                 yticks=[(1.0, "$1$"), (np.sqrt(2), "$\\sqrt2$"), (np.pi / 2, "$\\pi/2$")])
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(0.5, 0.28), Pl.pt(0.5, 1.6)))
    b.append("\\node[note, anchor=south] at %s {$\\rho=\\tfrac12\\Id$};" % Pl.pt(0.5, 1.6))
    b.append(Pl.curve(s, F, "line width=1.0pt, ndGold, dash pattern=on 4pt off 2pt"))
    b.append(Pl.curve(s, HS, "line width=1.0pt, ndGreen"))
    b.append(Pl.curve(s, D, "line width=1.2pt, ndBlue"))
    b.append(Pl.curve(s, A, "line width=1.4pt, ndRed"))
    b.append("\\node[font=\\footnotesize, text=ndRed, anchor=south west] at %s {Bures angle $\\arccos\\sqrt F$};" % Pl.pt(1.02, np.pi / 2 - 0.06))
    b.append("\\node[font=\\footnotesize, text=ndGreen, anchor=north west] at %s {Hilbert--Schmidt $\\sqrt2\\,s$};" % Pl.pt(1.02, np.sqrt(2) + 0.06))
    b.append("\\node[font=\\footnotesize, text=ndBlue, anchor=west] at %s {trace distance $D=s$};" % Pl.pt(1.02, 1.0))
    b.append("\\node[font=\\footnotesize, text=ndGold!80!ndInk, anchor=south west] at %s {fidelity $F=1-s$};" % Pl.pt(0.02, 1.02))
    b.append("\\node[note, anchor=south west] (sl) at %s {infinite slope\\\\ at the pure state};" % Pl.pt(0.27, 0.015))
    b.append("\\draw[pointer] (sl.west) to[bend left=20] %s;" % Pl.pt(0.015, 0.14))
    N.write("ch07_metrics", b)
    print("written figures/ch07_metrics.tex")


if __name__ == "__main__":
    main()
