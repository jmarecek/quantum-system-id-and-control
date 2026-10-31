"""ch12_grape.py -- a minimal GRAPE for a qubit state transfer.

Model:  H(t) = (Delta/2) sigma_z + (u(t)/2) sigma_x,  u piecewise constant on N slices,
goal:   maximise F = |<1|psi(T)>|^2  (transfer |0> -> |1>) with a small energy penalty.
Writes  figures/ch12_grape_conv.tex  (fidelity vs iteration)
        figures/ch12_grape_pulse.tex (initial guess and optimised pulse).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm
import tikzexport as T

I2, X, Y, Z = T.pauli()
Delta, N, Ttot, eps, penalty = 0.5, 40, 6.0, 0.8, 1e-3
dt = Ttot / N
psi0 = np.array([1, 0], dtype=complex)
target = np.array([0, 1], dtype=complex)


def propagators(u):
    return [expm(-1j * dt * (0.5 * Delta * Z + 0.5 * u[n] * X)) for n in range(N)]


def grape_gradient(u):
    """Forward/backward sweep: returns (fidelity, dF/du_n)."""
    U = propagators(u)
    psi = [psi0]                                  # forward states psi_n
    for n in range(N):
        psi.append(U[n] @ psi[-1])
    overlap = np.vdot(target, psi[-1])            # <1|psi(T)>
    chi = [None] * (N + 1)
    chi[N] = overlap * target                     # backward state chi_N
    for n in range(N, 0, -1):
        chi[n - 1] = U[n - 1].conj().T @ chi[n]
    grad = np.zeros(N)
    for n in range(N):                            # dF/du_n = 2 dt Im <chi_n| H1 |psi_n>
        grad[n] = 2 * dt * np.imag(np.vdot(chi[n + 1], 0.5 * X @ psi[n + 1]))
    return abs(overlap) ** 2, grad


def main():
    rng = np.random.RandomState(1)
    u_guess = 0.3 * np.ones(N) + 0.05 * rng.randn(N)
    u = u_guess.copy()
    hist = []
    for it in range(150):
        F, g = grape_gradient(u)
        hist.append(F)
        u = u + eps * (g - penalty * u)           # gradient ascent step
    F, _ = grape_gradient(u)
    hist.append(F)
    print("final fidelity %.6f" % F)
    its = np.arange(len(hist))
    T.write_tikz("ch12_grape_conv", T.pgfplots_axis(
        [dict(x=its, y=1 - np.array(hist), label="$1-F$")],
        xlabel="iteration", ylabel="infidelity $1-F$", legend_pos="north east",
        axis_options="ymode=log"))
    t = np.arange(N + 1) * dt
    T.write_tikz("ch12_grape_pulse", T.pgfplots_axis(
        [dict(x=t, y=np.append(u_guess, u_guess[-1]), label="initial guess", const=True, style="dashed"),
         dict(x=t, y=np.append(u, u[-1]), label="GRAPE, 150 iterations", const=True)],
        xlabel="$t$", ylabel="$u(t)$", legend_pos="south east"))


if __name__ == "__main__":
    main()
