"""ch01_twoqubit_data.py -- Example D: the data of an identification experiment.
Two coupled qubits with local amplitude damping and dephasing; a few Pauli
expectation values in time, exact and estimated from a finite number of shots.
Writes figures/ch01_twoqubit_data.tex."""
import os, sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

rng = np.random.RandomState(3)
I, X, Y, Z = T.pauli()
kron = np.kron
H = 0.5 * 1.0 * kron(Z, I) + 0.5 * 1.3 * kron(I, Z) + 0.35 * kron(X, X)
sm = np.array([[0, 1], [0, 0]], dtype=complex)
Ls = [(0.05, kron(sm, I)), (0.05, kron(I, sm)), (0.04, kron(Z, I)), (0.04, kron(I, Z))]
d = 4
def vec(A): return A.reshape(-1, order='F')
Id = np.eye(d)
A = -1j * (kron(Id, H) - kron(H.T, Id))
for g, L in Ls:
    LdL = L.conj().T @ L
    A += g * (kron(L.conj(), L) - 0.5 * kron(Id, LdL) - 0.5 * kron(LdL.T, Id))
psi0 = np.kron(np.array([1, 1]) / np.sqrt(2), np.array([1, 0])).astype(complex)
rho0 = np.outer(psi0, psi0.conj())
ts = np.linspace(0, 30, 61)
obs = {r"$\langle X\!\otimes\! I\rangle$": kron(X, I), r"$\langle Z\!\otimes\! I\rangle$": kron(Z, I), r"$\langle Y\!\otimes\! Z\rangle$": kron(Y, Z)}
series = []
shots = 200
for name, O in obs.items():
    exact, est = [], []
    for t in ts:
        rho = (expm(A * t) @ vec(rho0)).reshape(d, d, order='F')
        e = np.real(np.trace(O @ rho)); exact.append(e)
        p = (1 + e) / 2; est.append(2 * rng.binomial(shots, p) / shots - 1)
    series.append(dict(x=ts, y=exact, label=name))
    series.append(dict(x=ts, y=est, label="", style="only marks, mark=*, mark size=0.8pt"))
body = T.pgfplots_axis(series, xlabel="$t$", ylabel="expectation value", width="0.85\\linewidth", height="0.42\\linewidth", legend_pos="north east")
T.write_tikz("ch01_twoqubit_data", body)
print("written figures/ch01_twoqubit_data.tex")
