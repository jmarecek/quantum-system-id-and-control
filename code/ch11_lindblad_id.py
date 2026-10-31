"""ch11_lindblad_id.py -- identify a qubit Lindbladian from noisy time series.

The GKSL equation is written as a linear system on the coherence vector
x_j = Tr(F_j rho), F = (I, X, Y, Z)/sqrt(2):  dx/dt = A x.  We simulate
trajectories from several initial states, corrupt the expectation values
with shot noise, fit the discrete propagator A_d = exp(A dt) by least
squares and recover A = log(A_d)/dt.  Writes figures/ch11_lindblad_id.tex.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm, logm
import tikzexport as T

I2, X, Y, Z = T.pauli()
F = [P / np.sqrt(2) for P in (I2, X, Y, Z)]      # orthonormal basis, Tr(F_j F_k) = delta_jk


def lindbladian(rho, omega, gamma1, gamma_phi):
    """GKSL generator of a driven-free qubit with relaxation and dephasing."""
    H = 0.5 * omega * Z
    sm = np.array([[0, 1], [0, 0]], dtype=complex)          # sigma_-
    D = gamma1 * (sm @ rho @ sm.conj().T - 0.5 * (sm.conj().T @ sm @ rho + rho @ sm.conj().T @ sm))
    D += 0.5 * gamma_phi * (Z @ rho @ Z - rho)
    return -1j * (H @ rho - rho @ H) + D


def generator_matrix(omega, gamma1, gamma_phi):
    """Matrix A of dx/dt = A x, x_j = Tr(F_j rho), by applying L to the basis."""
    A = np.zeros((4, 4))
    for k in range(4):
        LFk = lindbladian(F[k], omega, gamma1, gamma_phi)
        for j in range(4):
            A[j, k] = np.real(np.trace(F[j] @ LFk))
    return A


def coherence_vector(rho):
    return np.real(np.array([np.trace(Fj @ rho) for Fj in F]))


def identify(A_true, dt=0.05, steps=80, shots=10000, seed=0):
    rng = np.random.RandomState(seed)
    Ad_true = expm(A_true * dt)
    inits = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1]) / np.sqrt(2),
             np.array([1, 1j]) / np.sqrt(2)]
    X0, X1 = [], []
    for psi in inits:
        x = coherence_vector(np.outer(psi, np.conj(psi)))
        traj = [x]
        for _ in range(steps):
            traj.append(Ad_true @ traj[-1])
        traj = np.array(traj)
        noisy = traj + rng.normal(scale=1.0 / np.sqrt(shots), size=traj.shape)
        noisy[:, 0] = traj[:, 0]                 # Tr(rho) = 1 is known exactly
        X0.append(noisy[:-1]); X1.append(noisy[1:])
    X0 = np.vstack(X0).T; X1 = np.vstack(X1).T   # columns = samples
    Ad_hat = X1 @ np.linalg.pinv(X0)             # least squares: X1 ~ Ad X0
    A_hat = np.real(logm(Ad_hat)) / dt           # principal logarithm
    return A_hat


def main():
    A_true = generator_matrix(omega=2.0, gamma1=0.3, gamma_phi=0.2)
    A_hat = identify(A_true)
    err = np.linalg.norm(A_hat - A_true) / np.linalg.norm(A_true)
    print("relative error of A_hat: %.3g" % err)
    lab = "xtick={0,1,2,3}, ytick={0,1,2,3}, xticklabels={$I$,$X$,$Y$,$Z$}, yticklabels={$I$,$X$,$Y$,$Z$}, point meta min=-2.2, point meta max=2.2"
    left = T.pgfplots_matrix(A_true, xlabel="exact $A$", width="0.36\\linewidth",
                             colorbar=False, axis_options=lab)
    right = T.pgfplots_matrix(A_hat, xlabel="recovered $\\hat A$ (rel.\\ error %.1e)" % err,
                              width="0.36\\linewidth", colorbar=True, axis_options=lab)
    body = left + "\n\\begin{scope}[xshift=0.42\\linewidth]\n" + right + "\n\\end{scope}"
    T.write_tikz("ch11_lindblad_id", body)


if __name__ == "__main__":
    main()
