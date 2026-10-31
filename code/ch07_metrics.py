"""ch07_metrics.py -- three distances between qubit states along a path.

We fix rho_0 = |0><0| and move rho(s) along the straight segment of Bloch
vectors from (0,0,1) through the maximally mixed state to (0,0,-1).
Output: figures/ch07_metrics.tex
"""
import sys, os
import numpy as np
from scipy.linalg import sqrtm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T


def state(r):
    """Density matrix with Bloch vector r (|r| <= 1)."""
    I, X, Y, Z = T.pauli()
    return 0.5 * (I + r[0] * X + r[1] * Y + r[2] * Z)


def fidelity(rho, sigma):
    """Uhlmann fidelity F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    s = sqrtm(rho)
    return np.real(np.trace(sqrtm(s @ sigma @ s))) ** 2


def trace_distance(rho, sigma):
    """D = (1/2) ||rho - sigma||_1 = half the sum of |eigenvalues|."""
    return 0.5 * np.abs(np.linalg.eigvalsh(rho - sigma)).sum()


def hs_distance(rho, sigma):
    return np.sqrt(np.real(np.trace((rho - sigma) @ (rho - sigma))))


def bures_angle(rho, sigma):
    return np.arccos(np.clip(np.sqrt(fidelity(rho, sigma)), 0, 1))


if __name__ == "__main__":
    rho0 = state([0, 0, 1])
    s = np.linspace(0, 1, 201)
    D, HS, A, F = [], [], [], []
    for si in s:
        rho = state([0, 0, 1 - 2 * si])
        D.append(trace_distance(rho0, rho)); HS.append(hs_distance(rho0, rho))
        A.append(bures_angle(rho0, rho)); F.append(fidelity(rho0, rho))
    body = T.pgfplots_axis([
        dict(x=s, y=D, label="trace distance $D$"),
        dict(x=s, y=HS, label="Hilbert--Schmidt $\\|\\rho_0-\\rho\\|_2$", style="dashed"),
        dict(x=s, y=A, label="Bures angle $\\arccos\\sqrt{F}$"),
        dict(x=s, y=F, label="fidelity $F$", style="dotted"),
    ], xlabel="$s$ (Bloch vector $(0,0,1-2s)$)", ylabel="", legend_pos="north west",
        width="0.8\\linewidth", height="0.45\\linewidth")
    T.write_tikz("ch07_metrics", body)
    print("wrote figures/ch07_metrics.tex")
