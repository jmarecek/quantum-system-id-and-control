"""ch01_penalty.py -- Example E: hard versus soft constraints on the velocity.
A qubit with a cheap direction (x rotations) and a penalised one (y rotations):
minimum-energy paths from |0> to a target for increasing penalty q approach the
path that uses only the allowed direction plus its commutators.
Writes figures/ch01_penalty.tex."""
import os, sys
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

I, X, Y, Z = T.pauli()
N = 24; Tf = 1.0; dt = Tf / N
def U_of(a):        # a = (ax[0..N-1], ay[0..N-1]); drift Z rotation always present
    U = np.eye(2, dtype=complex)
    for k in range(N):
        U = expm(-1j * dt * (0.8 * Z + a[k] * X + a[N + k] * Y)) @ U
    return U
Vt = expm(-1j * (np.pi / 2) * (X + 0.6 * Y) / np.sqrt(1 + 0.36))   # target gate
def cost(a, q):
    U = U_of(a); infid = 1 - abs(np.trace(Vt.conj().T @ U)) ** 2 / 4
    energy = dt * (np.sum(a[:N] ** 2) + q * np.sum(a[N:] ** 2))
    return 200 * infid + energy
a0 = np.concatenate([np.full(N, 1.5), np.full(N, 0.9)])
qs = [1, 4, 16, 64]; energies = []; ys = []
for q in qs:
    res = minimize(cost, a0, args=(q,), method='L-BFGS-B', options=dict(maxiter=400))
    a0 = res.x; a = res.x
    energies.append(dt * np.sum(a[:N] ** 2)); ys.append(a[N:])
tgrid = np.arange(N) * dt
series = [dict(x=tgrid, y=ys[i], label="$q=%d$" % q, const=True) for i, q in enumerate(qs)]
body = T.pgfplots_axis(series, xlabel="$t$", ylabel=r"penalised control $a_y(t)$", width="0.85\\linewidth", height="0.4\\linewidth", legend_pos="north east")
T.write_tikz("ch01_penalty", body)
print("written figures/ch01_penalty.tex  energies of the cheap control:", ["%.2f" % e for e in energies])
