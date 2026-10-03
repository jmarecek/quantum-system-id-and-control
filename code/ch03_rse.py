"""ch03_rse.py -- the random Schroedinger equation for one qubit.

Sample paths of  i d/dt psi = (H_0 + H_1(t, omega)) psi  with
H_0 = (omega/2) sigma_z + (u/2) sigma_x and a random dephasing term
H_1 = (1/2) b(t, omega) sigma_z, b an Ornstein-Uhlenbeck process.
Each path stays pure; the ensemble average is a mixed state.
Writes figures/ch03_rse.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N

I, X, Y, Z = T.pauli()


def ou_path(rng, n_steps, dt, tau=0.5, sigma=1.0):
    """Ornstein-Uhlenbeck process with correlation time tau and std sigma."""
    b = np.zeros(n_steps)
    a = np.exp(-dt / tau)
    for k in range(1, n_steps):
        b[k] = a * b[k - 1] + sigma * np.sqrt(1 - a * a) * rng.randn()
    return b


def sample_path(rng, times, omega=1.0, u=1.0):
    dt = times[1] - times[0]
    b = ou_path(rng, len(times), dt)
    psi = np.array([1.0, 0.0], dtype=complex)
    out = [psi]
    for k in range(1, len(times)):
        H = 0.5 * omega * Z + 0.5 * u * X + 0.5 * b[k] * Z
        psi = expm(-1j * H * dt) @ psi
        out.append(psi)
    return np.array(out)


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    times = np.linspace(0, 10, 400)
    n_paths = 300
    paths = [sample_path(rng, times) for _ in range(n_paths)]
    assert all(np.allclose(np.linalg.norm(p, axis=1), 1) for p in paths)      # every path stays pure
    rho_hat = np.mean([np.einsum("ti,tj->tij", p, p.conj()) for p in paths], axis=0)
    purity = np.real(np.einsum("tij,tji->t", rho_hat, rho_hat))
    x_hat = np.real(np.einsum("tij,ji->t", rho_hat, X))
    b = []
    # left panel: sample paths and the ensemble mean of <sigma_x>
    Pl = N.Plot((0, 10), (-1.05, 1.1), 8.0, 4.4)
    b += Pl.axes("$t$", "$\\langle\\sigma_x\\rangle$", yticks=[(-1, "$-1$"), (0, "$0$"), (1, "$1$")], yzero=0.0)
    cols = ["ndGold", "ndGreen", "ndRose!60!ndRed"]
    for k in range(3):
        xs = np.real(np.einsum("ti,ij,tj->t", paths[k].conj(), X, paths[k]))
        b.append(Pl.curve(times, xs, "line width=0.7pt, %s" % cols[k]))
    b.append(Pl.curve(times, x_hat, "line width=2.0pt, ndBlue"))
    b.append("\\node[note, anchor=south west] (s) at %s {sample paths: each stays pure,\\\\ but they disagree with each other};" % Pl.pt(4.2, 1.06))
    b.append("\\node[note, anchor=north west, text=ndBlue] (m) at %s {ensemble mean over %d paths:\\\\ damped};" % (Pl.pt(5.8, -0.55), n_paths))
    b.append("\\draw[pointer] (m.north west) to[bend left=15] %s;" % Pl.pt(5.2, x_hat[np.searchsorted(times, 5.2)] - 0.03))
    b.append("\\node[font=\\small] at %s {(a) $\\langle\\sigma_x\\rangle$ along sample paths};" % N.pt(Pl.P(5, -1.05) + np.array([0, -0.65])))
    # right panel: purity of the averaged state
    Q = N.Plot((0, 10), (0.4, 1.05), 5.2, 4.4, origin=(10.0, 0.0))
    b += Q.axes("$t$", "$\\Tr\\hat\\rho^2$", yticks=[(0.5, "$\\tfrac12$"), (1, "$1$")])
    b.append("\\draw[dashedink] %s -- %s;" % (Q.pt(0, 0.5), Q.pt(10, 0.5)))
    b.append("\\draw[dashedink] %s -- %s;" % (Q.pt(0, 1.0), Q.pt(10, 1.0)))
    b.append(Q.curve(times, purity, "line width=1.8pt, ndRed"))
    b.append("\\node[note, anchor=south west] at %s {each path: purity $1$};" % Q.pt(4.2, 1.0))
    b.append("\\node[note, anchor=north west, text=ndRed] at %s {the average decays\\\\ towards $\\tfrac12$: mixed};" % Q.pt(4.2, 0.92))
    b.append("\\node[font=\\small] at %s {(b) purity of $\\hat\\rho=\\Expect\\rho$};" % N.pt(Q.P(5, 0.4) + np.array([0, -0.65])))
    N.write("ch03_rse", b)
    print("written figures/ch03_rse.tex; final purity %.3f" % purity[-1])
