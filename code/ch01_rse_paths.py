"""ch01_rse_paths.py -- Example D: sample paths of the random Schroedinger equation
for a driven qubit (each path unitary) and the purity of their average.

H = u sigma_x + H_n(t), with u = 0.5 and H_n a white-noise combination of the
three Pauli matrices (strength 0.35); 200 paths from |0> over [0, 8], of which
five are drawn, with the average <sigma_z> and the purity of the average state.
Writes figures/ch01_rse_paths.tex.
"""
import os, sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N
import tikzexport as T


def main():
    rng = np.random.RandomState(7)
    I, X, Y, Z = T.pauli()
    T_final, n_steps = 8.0, 400
    dt = T_final / n_steps
    t = np.linspace(0, T_final, n_steps + 1)
    u = 0.5                                  # constant drive along x
    sigma = 0.35                             # noise strength of the random Hermitian term
    paths, rhos = [], np.zeros((n_steps + 1, 2, 2), dtype=complex)
    M = 200
    for k in range(M):
        psi = np.array([1, 0], dtype=complex)
        zs = [1.0]
        rhos[0] += np.outer(psi, psi.conj()) / M
        xi = rng.normal(size=(n_steps, 3)) * sigma / np.sqrt(dt)   # white-noise coefficients of the noise Hamiltonian
        for n in range(n_steps):
            Hn = xi[n, 0] * X + xi[n, 1] * Y + xi[n, 2] * Z
            psi = expm(-1j * (u * X + Hn) * dt) @ psi
            zs.append(np.real(psi.conj() @ Z @ psi))
            rhos[n + 1] += np.outer(psi, psi.conj()) / M
        if k < 5:
            paths.append(np.array(zs))
    purity = np.real(np.einsum('tij,tji->t', rhos, rhos))
    zavg = np.real(np.einsum('tij,ji->t', rhos, Z))
    assert abs(purity[-1] - 0.5) < 0.02 and abs(purity[0] - 1) < 1e-9    # from a pure state to (nearly) the maximally mixed one
    assert abs(zavg[-1]) < 0.1

    b = []
    P = N.Plot((0.0, 8.0), (-1.05, 1.05), 9.6, 5.0)
    b += P.axes("$t$", "", xticks=[(k, "$%d$" % k) for k in (2, 4, 6, 8)],
                yticks=[(-1, "$-1$"), (0, "$0$"), (0.5, "$\\tfrac12$"), (1, "$1$")])
    b.append("\\draw[faint] %s -- %s;" % (P.pt(0, 0), P.pt(8, 0)))
    b.append("\\draw[dashedink] %s -- %s;" % (P.pt(0, 0.5), P.pt(8, 0.5)))
    for p in paths:
        b.append(P.curve(t, p, "line width=0.45pt, ndBlue!45"))
    b.append(P.curve(t, zavg, "line width=1.3pt, ndRed"))
    b.append(P.curve(t, purity, "line width=1.3pt, ndGold"))
    b.append("\\node[note, anchor=west, text=ndGold] (pu) at %s {the purity $\\Tr\\rho^2$ of the\\\\ average state: from $1$\\\\ down to $\\tfrac12$};" % P.pt(8.45, 0.82))
    b.append("\\draw[pointer] (pu.south west) to[bend left=20] %s;" % P.pt(7.7, purity[-8] + 0.03))
    b.append("\\node[note, anchor=west, text=ndRed] (av) at %s {the average\\\\ $\\Tr(\\rho\\,\\sigma_z)$ over\\\\ $%d$ paths};" % (P.pt(8.45, 0.05), M))
    b.append("\\draw[pointer] (av.west) to[bend right=10] %s;" % P.pt(7.75, zavg[-6]))
    k = int(np.argmin([abs(p[-10]) for p in paths]))
    b.append("\\node[note, anchor=west, text=ndBlue] (sp) at %s {five sample paths\\\\ of $\\langle\\sigma_z\\rangle$, each\\\\ a unitary evolution};" % P.pt(8.45, -0.55))
    b.append("\\draw[pointer] (sp.west) to[bend left=15] %s;" % P.pt(7.6, paths[k][-17]))
    N.write("ch01_rse_paths", b)
    print("written figures/ch01_rse_paths.tex  final purity %.3f" % purity[-1])


if __name__ == "__main__":
    main()
