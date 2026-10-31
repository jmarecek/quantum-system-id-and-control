"""ch01_twoqubit_data.py -- Example E: the data of an identification experiment.
Two coupled qubits with local amplitude damping and dephasing; a few Pauli
expectation values in time, exact and estimated from a finite number of shots
(200 per time point, at 61 times in [0, 30]).
Writes figures/ch01_twoqubit_data.tex.
"""
import os, sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N
import tikzexport as T

I, X, Y, Z = T.pauli()
kron = np.kron
H = 0.5 * 1.0 * kron(Z, I) + 0.5 * 1.3 * kron(I, Z) + 0.35 * kron(X, X)
sm = np.array([[0, 1], [0, 0]], dtype=complex)
Ls = [(0.05, kron(sm, I)), (0.05, kron(I, sm)), (0.04, kron(Z, I)), (0.04, kron(I, Z))]
d = 4


def vec(A):
    return A.reshape(-1, order='F')


Id = np.eye(d)
A = -1j * (kron(Id, H) - kron(H.T, Id))
for g, L in Ls:
    LdL = L.conj().T @ L
    A += g * (kron(L.conj(), L) - 0.5 * kron(Id, LdL) - 0.5 * kron(LdL.T, Id))
psi0 = np.kron(np.array([1, 1]) / np.sqrt(2), np.array([1, 0])).astype(complex)
rho0 = np.outer(psi0, psi0.conj())


def expectation(O, t):
    rho = (expm(A * t) @ vec(rho0)).reshape(d, d, order='F')
    return np.real(np.trace(O @ rho))


def main():
    rng = np.random.RandomState(3)
    ts = np.linspace(0, 30, 61)
    fine = np.linspace(0, 30, 601)
    obs = [("X\\!\\otimes\\! I", kron(X, I), "ndBlue"), ("Z\\!\\otimes\\! I", kron(Z, I), "ndRed"),
           ("Y\\!\\otimes\\! Z", kron(Y, Z), "ndGold")]
    shots = 200
    P = N.Plot((0.0, 30.0), (-0.65, 1.05), 10.0, 5.0)
    b = P.axes("$t$", "", xticks=[(k, "$%d$" % k) for k in (10, 20, 30)],
               yticks=[(-0.5, "$-0.5$"), (0, "$0$"), (0.5, "$0.5$"), (1, "$1$")])
    b.append("\\draw[faint] %s -- %s;" % (P.pt(0, 0), P.pt(30, 0)))
    curves = {}
    for name, O, col in obs:
        exact, est = [], []
        for t in ts:
            e = expectation(O, t)
            exact.append(e)
            p = (1 + e) / 2
            est.append(2 * rng.binomial(shots, p) / shots - 1)
        curves[name] = np.array([expectation(O, t) for t in fine])
        assert np.allclose(curves[name][::10], exact)
        b.append(P.curve(fine, curves[name], "line width=1.0pt, %s" % col))
        b += ["\\fill[%s, fill opacity=0.8] %s circle (1.1pt);" % (col, P.pt(t, y)) for t, y in zip(ts, est)]
        err = np.abs(np.array(est) - np.array(exact))
        assert err.max() < 4.5 / np.sqrt(shots)                  # the dots scatter by about 1/sqrt(200)
    xi, zi, yz = (curves[n] for n, _, _ in obs)
    assert abs(xi[0] - 1) < 1e-9 and abs(zi[-1] - 0.69) < 0.02           # the labels below rely on these values
    b.append("\\node[note, anchor=west, text=ndBlue] (x) at %s {$\\langle X\\!\\otimes\\! I\\rangle$, from $1$};" % P.pt(2.2, 0.98))
    b.append("\\draw[pointer] (x.west) to[bend right=20] %s;" % P.pt(0.5, 0.93))
    b.append("\\node[note, anchor=south, text=ndRed] (z) at %s {$\\langle Z\\!\\otimes\\! I\\rangle$, settling at $%.2f$};" % (P.pt(23.0, 0.80), zi[-1]))
    k = np.argmin(np.where((fine > 5) & (fine < 8), yz, 1.0))
    b.append("\\node[note, anchor=west, text=ndGold] (y) at %s {$\\langle Y\\!\\otimes\\! Z\\rangle$};" % P.pt(10.0, -0.52))
    b.append("\\draw[pointer] (y.west) to[bend left=20] %s;" % P.pt(fine[k] + 0.3, yz[k] - 0.03))
    b.append("\\node[note, anchor=north east] at %s {curves: the exact values; dots: estimates\\\\ from $%d$ shots at each of $%d$ times};" % (P.pt(30.0, -0.32), shots, len(ts)))
    N.write("ch01_twoqubit_data", b)
    print("written figures/ch01_twoqubit_data.tex")


if __name__ == "__main__":
    main()
