"""ch01_rse_paths.py -- Example C: sample paths of the random Schroedinger equation
for a driven qubit (each path unitary) and the purity of their average.
Writes figures/ch01_rse_paths.tex."""
import os, sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

rng = np.random.RandomState(7)
I, X, Y, Z = T.pauli()
T_final, N = 8.0, 400; dt = T_final / N; t = np.linspace(0, T_final, N + 1)
u = 0.5                                  # constant drive along x
sigma = 0.35                             # noise strength of the random Hermitian term
paths, rhos = [], np.zeros((N + 1, 2, 2), dtype=complex)
M = 200
for k in range(M):
    psi = np.array([1, 0], dtype=complex); zs = [1.0]; rhos[0] += np.outer(psi, psi.conj()) / M
    xi = rng.normal(size=(N, 3)) * sigma / np.sqrt(dt)   # white-noise coefficients of the noise Hamiltonian
    for n in range(N):
        Hn = xi[n, 0] * X + xi[n, 1] * Y + xi[n, 2] * Z
        psi = expm(-1j * (u * X + Hn) * dt) @ psi
        zs.append(np.real(psi.conj() @ Z @ psi)); rhos[n + 1] += np.outer(psi, psi.conj()) / M
    if k < 5: paths.append(np.array(zs))
purity = np.real(np.einsum('tij,tji->t', rhos, rhos))
zavg = np.real(np.einsum('tij,ji->t', rhos, Z))
series = [dict(x=t, y=p, label=("sample paths" if i == 0 else ""), style="thin, opacity=0.6") for i, p in enumerate(paths)]
series += [dict(x=t, y=zavg, label=r"average $\Tr(\rho\sigma_z)$", style="very thick"),
           dict(x=t, y=purity, label=r"purity $\Tr\rho^2$", style="very thick, dashed")]
body = T.pgfplots_axis(series, xlabel="$t$", ylabel="", width="0.85\\linewidth", height="0.42\\linewidth", legend_pos="south west")
T.write_tikz("ch01_rse_paths", body)
print("written figures/ch01_rse_paths.tex  final purity %.3f" % purity[-1])
