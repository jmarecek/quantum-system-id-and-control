"""ch06_shots.py -- shots, estimates and Fisher information for Chapter 6.

The helpers below are shared by the ch06_*.py scripts.  The model is the qubit
of Chapter 5 (ch05_spectrum): omega = 3, gamma_1 = 0.2, gamma_phi = 0.4, so
that Gamma = 0.5, prepared in the tilted state X0 and measured through a Pauli
observable with outcomes +-1.  An expectation value y is estimated from N shots
as the mean of N outcomes, i.e. 1 - 2 Binomial(N, (1 - y)/2) / N, and the
Fisher information of N shots of each of the expectation values y_k(theta) is
    F(theta) = N sum_k grad y_k grad y_k^T / (1 - y_k^2).

The script itself (main) checks the 1/sqrt(N) law for the decay rate gamma_1:
  (a) <sigma_z>(t) of the example at 12 times in [1, 15], each estimated
      from N = 100 shots (error bars of one standard deviation), with the
      least-squares fit of gamma_1;
  (b) the root-mean-square error of the fitted gamma_1 over 300 repetitions
      against N (dots), with the Cramer-Rao bound sqrt(1/F) (line), which
      falls with slope -1/2 on logarithmic axes.
Writes figures/ch06_shots.tex.
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N
from ch05_spectrum import qubit_A, X0, CX, CZ, CN, OMEGA, GAMMA1, GPHI, GAMMA, TAU

THETA_TRUE = np.array([OMEGA, GAMMA1, GPHI])            # (omega, gamma_1, gamma_phi)


def tilted(phi):
    """Bloch vector at angle phi from the south pole, in the x-z plane (phi = 0: |1>)."""
    return np.array([np.sin(phi), 0.0, -np.cos(phi), 1.0])


def expect(theta, C, x0, t):
    """Expectation values C e^{A(theta) t} x0 at the times t; theta = (omega, gamma_1, gamma_phi)."""
    A = qubit_A(theta[0], 0.0, theta[1], theta[2])
    return np.array([(C @ expm(A * s) @ x0).item() for s in np.atleast_1d(t)])


def sample_mean(y, shots, rng):
    """Estimates of the expectation values y (of +-1 observables) from `shots` shots each."""
    p_minus = (1.0 - np.asarray(y)) / 2.0
    return 1.0 - 2.0 * rng.binomial(shots, p_minus) / shots


def fisher(model, theta, shots=1, h=1e-6):
    """Fisher information of `shots` shots on each of the +-1 expectation values model(theta)."""
    theta = np.asarray(theta, dtype=float)
    y = model(theta)
    J = np.empty((len(y), len(theta)))
    for i in range(len(theta)):
        d = np.zeros_like(theta)
        d[i] = h * max(1.0, abs(theta[i]))
        J[:, i] = (model(theta + d) - model(theta - d)) / (2 * d[i])
    w = shots / np.clip(1.0 - y ** 2, 1e-12, None)
    return J.T @ (w[:, None] * J)


def crb(F):
    """Cramer-Rao bound: the inverse of the Fisher information (the smallest covariance)."""
    return np.linalg.inv(F)


def fit_gamma1(times, zhat, x0=X0):
    """Least-squares estimate of gamma_1 from estimates of <sigma_z>(t) = 1 - (1 - z0) e^{-gamma_1 t}."""
    from scipy.optimize import minimize_scalar
    z0 = x0[2]
    loss = lambda g: np.sum((1 - (1 - z0) * np.exp(-g * times) - zhat) ** 2)
    return minimize_scalar(loss, bounds=(1e-3, 5.0), method="bounded").x


def main():
    rng = np.random.default_rng(1)
    times = np.linspace(1.0, 15.0, 12)
    model = lambda th: expect(np.array([OMEGA, th[0], GPHI]), CZ, X0, times)
    z_true = model([GAMMA1])
    b = []
    # ---- (a) one data set of N = 100 shots per time, and its fit
    n_a = 100
    zhat = sample_mean(z_true, n_a, rng)
    g_fit = fit_gamma1(times, zhat)
    sd = np.sqrt(np.clip(1 - zhat ** 2, 0, None) / n_a)
    tt = np.linspace(0, 16, 300)
    z0 = X0[2]
    Pl = N.Plot((0.0, 16.0), (-0.6, 1.1), 7.0, 4.6, origin=(0.0, 0.0))
    b += Pl.axes("$t$", "$\\langle\\sigma_z\\rangle$", xticks=[(5, "$5$"), (10, "$10$"), (15, "$15$")],
                 yticks=[(-0.5, "$-0.5$"), (0, "$0$"), (0.5, "$0.5$"), (1, "$1$")])
    b.append("\\draw[faint] %s -- %s;" % (Pl.pt(0, 0), Pl.pt(16, 0)))
    b.append(Pl.curve(tt, 1 - (1 - z0) * np.exp(-GAMMA1 * tt), "dashedink"))
    b.append(Pl.curve(tt, 1 - (1 - z0) * np.exp(-g_fit * tt), "line width=1.1pt, ndRed"))
    for t, z, e in zip(times, zhat, sd):
        b.append("\\draw[line width=0.7pt, ndInk] %s -- %s;" % (Pl.pt(t, z - e), Pl.pt(t, z + e)))
        b.append("\\draw[line width=0.7pt, ndInk] %s -- %s %s -- %s;" % (Pl.pt(t - 0.15, z - e), Pl.pt(t + 0.15, z - e), Pl.pt(t - 0.15, z + e), Pl.pt(t + 0.15, z + e)))
        b.append("\\node[dot] at %s {};" % Pl.pt(t, z))
    assert abs(g_fit - GAMMA1) < 0.05
    b.append("\\node[note, anchor=north west] at %s {estimates from $N=%d$ shots,\\\\ with one standard deviation};" % (Pl.pt(4.6, -0.12), n_a))
    b.append("\\node[note, anchor=south east, text=ndRed] (f) at %s {fitted: $\\hat\\gamma_1=%.3f$};" % (Pl.pt(15.6, 0.32), g_fit))
    b.append("\\draw[pointer] (f.north) to[bend right=15] %s;" % Pl.pt(11.5, 1 - (1 - z0) * np.exp(-g_fit * 11.5) - 0.03))
    b.append("\\node[note, anchor=south west] (tr) at %s {true: $\\gamma_1=%.1f$};" % (Pl.pt(1.0, 0.75), GAMMA1))
    b.append("\\draw[pointer] (tr.east) to[bend left=15] %s;" % Pl.pt(7.0, 1 - (1 - z0) * np.exp(-GAMMA1 * 7.0) + 0.02))
    b.append("\\node[font=\\small] at %s {(a) data from $%d$ shots per time, and the fit};" % (N.pt(Pl.P(8.0, -0.6) + np.array([0.0, -0.95])), n_a))
    # ---- (b) the error against the number of shots
    shot_grid = np.logspace(1, 5, 9).astype(int)
    rmse, bound = [], []
    for n_shots in shot_grid:
        errs = [fit_gamma1(times, sample_mean(z_true, n_shots, rng)) - GAMMA1 for _ in range(300)]
        rmse.append(np.sqrt(np.mean(np.square(errs))))
        bound.append(np.sqrt(crb(fisher(model, [GAMMA1], n_shots))[0, 0]))
        print("N = %6d  RMSE = %.5f  Cramer-Rao = %.5f" % (n_shots, rmse[-1], bound[-1]))
    rmse, bound = np.array(rmse), np.array(bound)
    assert np.all(np.abs(np.log(rmse[2:] / bound[2:])) < 0.3)          # least squares is nearly efficient
    assert np.allclose(np.diff(np.log10(bound)) / np.diff(np.log10(shot_grid)), -0.5, atol=1e-6)
    lx, ly = np.log10(shot_grid), np.log10(rmse)
    Q = N.Plot((0.8, 5.2), (-4.0, -0.8), 6.6, 4.6, origin=(9.4, 0.0))
    b += Q.axes("shots $N$ per time", "error of $\\hat\\gamma_1$", xticks=[(k, "$10^{%d}$" % k) for k in range(1, 6)],
                yticks=[(k, "$10^{%d}$" % k) for k in range(-4, 0)])
    b.append(Q.curve(lx, np.log10(bound), "line width=1.1pt, ndBlue"))
    b += ["\\node[dot, fill=ndRed] at %s {};" % Q.pt(x, y) for x, y in zip(lx, ly)]
    # a slope triangle: one decade in N, half a decade in the error
    x0, y0 = 3.5, np.log10(bound[0]) - 0.5 * (3.5 - lx[0]) - 0.38
    b.append("\\draw[ink, line width=0.5pt] %s -- %s -- %s -- cycle;" % (Q.pt(x0, y0), Q.pt(x0 + 1, y0 - 0.5), Q.pt(x0, y0 - 0.5)))
    b.append("\\node[font=\\scriptsize, anchor=east] at %s {$\\tfrac12$};" % Q.pt(x0, y0 - 0.25))
    b.append("\\node[font=\\scriptsize, anchor=north] at %s {$1$};" % Q.pt(x0 + 0.5, y0 - 0.5))
    b.append("\\node[note, anchor=south west, text=ndBlue] (c) at %s {the Cram\\'er--Rao bound\\\\ $1/\\sqrt{F}$, falling as $N^{-1/2}$};" % Q.pt(2.75, -1.3))
    b.append("\\draw[pointer] (c.south west) to[bend right=15] %s;" % Q.pt(2.6, np.log10(bound[0]) - 0.5 * (2.6 - lx[0]) + 0.05))
    b.append("\\node[note, anchor=south west, text=ndRed] at %s {red dots: the error of\\\\ least squares over\\\\ $300$ repetitions};" % Q.pt(0.95, -3.95))
    b.append("\\node[font=\\small] at %s {(b) the error against the number of shots};" % N.pt(Q.P(3.0, -4.0) + np.array([0.0, -0.95])))
    N.write("ch06_shots", b)
    print("written figures/ch06_shots.tex; fitted gamma_1 from 100 shots: %.4f" % g_fit)


if __name__ == "__main__":
    main()
