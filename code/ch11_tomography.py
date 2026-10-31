"""ch11_tomography.py -- Monte-Carlo qubit state tomography.

Compares linear inversion with projected least squares (projection onto the
Bloch ball, i.e. onto the set of density matrices) as a function of the
number of shots per Pauli setting.  Writes figures/ch11_tomography.tex.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tikzexport as T

I2, X, Y, Z = T.pauli()
PAULIS = [X, Y, Z]


def sample_bloch(rho, shots, rng):
    """Estimate the Bloch vector of rho from `shots` projective measurements
    of each of X, Y, Z (finite statistics: binomial outcomes)."""
    r_hat = np.zeros(3)
    for k, P in enumerate(PAULIS):
        p_plus = 0.5 * (1.0 + np.real(np.trace(rho @ P)))   # Prob(outcome +1)
        n_plus = rng.binomial(shots, p_plus)
        r_hat[k] = (2.0 * n_plus - shots) / shots               # empirical <P>
    return r_hat


def linear_inversion(r_hat):
    """rho = (I + r.sigma)/2 -- may fail to be positive semidefinite."""
    return 0.5 * (I2 + r_hat[0] * X + r_hat[1] * Y + r_hat[2] * Z)


def project_to_states(r_hat):
    """Projected least squares: nearest density matrix in Frobenius norm.
    For a qubit this is the radial projection onto the Bloch ball."""
    n = np.linalg.norm(r_hat)
    r = r_hat if n <= 1.0 else r_hat / n
    return linear_inversion(r)


def fidelity_pure(psi, rho):
    return np.real(np.conj(psi) @ rho @ psi)


def main(seed=1, trials=400):
    rng = np.random.RandomState(seed)
    theta, phi = 0.4 * np.pi, 0.3            # a fixed pure target state
    psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    rho = np.outer(psi, np.conj(psi))
    shots_grid = np.unique(np.logspace(1, 4, 13).astype(int))
    inf_lin, inf_proj = [], []
    for shots in shots_grid:
        e_lin = e_proj = 0.0
        for _ in range(trials):
            r_hat = sample_bloch(rho, shots, rng)
            e_lin += 1.0 - fidelity_pure(psi, linear_inversion(r_hat))
            e_proj += 1.0 - fidelity_pure(psi, project_to_states(r_hat))
        inf_lin.append(max(e_lin / trials, 1e-7))
        inf_proj.append(max(e_proj / trials, 1e-7))
    guide = 1.5 / shots_grid
    body = T.pgfplots_axis(
        [dict(x=shots_grid, y=inf_lin, label="linear inversion", style="mark=o"),
         dict(x=shots_grid, y=inf_proj, label="projected least squares", style="mark=square"),
         dict(x=shots_grid, y=guide, label="$\\propto 1/N$", style="dashed")],
        xlabel="shots per Pauli setting $N$", ylabel="mean infidelity $1-F$",
        legend_pos="south west", axis_options="xmode=log, ymode=log")
    T.write_tikz("ch11_tomography", body)


if __name__ == "__main__":
    main()
