"""ch01_penalty.py -- Example F: hard versus soft constraints on the velocity.
A qubit with drift 0.8 sigma_z, a cheap direction (x rotations) and a penalised
one (y rotations): minimum-energy controls for a target gate, piecewise
constant on 24 steps of [0, 1], for increasing penalty q.  As q grows the
penalised control a_y is used less, and the cheap control a_x, together with
its commutator with the drift, does the work.
Writes figures/ch01_penalty.tex.
"""
import os, sys
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N
import tikzexport as T

I, X, Y, Z = T.pauli()
NSTEP = 24
Tf = 1.0
dt = Tf / NSTEP
Vt = expm(-1j * (np.pi / 2) * (X + 0.6 * Y) / np.sqrt(1 + 0.36))   # target gate


def U_of(a):        # a = (ax[0..N-1], ay[0..N-1]); drift Z rotation always present
    U = np.eye(2, dtype=complex)
    for k in range(NSTEP):
        U = expm(-1j * dt * (0.8 * Z + a[k] * X + a[NSTEP + k] * Y)) @ U
    return U


def infidelity(a):
    return 1 - abs(np.trace(Vt.conj().T @ U_of(a))) ** 2 / 4


def cost(a, q):
    energy = dt * (np.sum(a[:NSTEP] ** 2) + q * np.sum(a[NSTEP:] ** 2))
    return 200 * infidelity(a) + energy


def main():
    a0 = np.concatenate([np.full(NSTEP, 1.5), np.full(NSTEP, 0.9)])
    qs = [1, 4, 16, 64]
    ex, ey, ys = [], [], []
    for q in qs:
        res = minimize(cost, a0, args=(q,), method='L-BFGS-B', options=dict(maxiter=400))
        a0 = a = res.x
        assert infidelity(a) < 2e-3                                   # every control reaches the gate
        ex.append(dt * np.sum(a[:NSTEP] ** 2))
        ey.append(dt * np.sum(a[NSTEP:] ** 2))
        ys.append(a[NSTEP:])
    assert np.all(np.diff(ey) < 0) and np.all(np.diff(ex) > 0)       # less of a_y, more of a_x, as q grows

    b = []
    P = N.Plot((0.0, 1.0), (-0.5, 1.6), 8.0, 4.8)
    b += P.axes("$t$", "$a_y(t)$", xticks=[(0.5, "$0.5$"), (1, "$1$")],
                yticks=[(-0.5, "$-0.5$"), (0.5, "$0.5$"), (1, "$1$"), (1.5, "$1.5$")])
    b.append("\\draw[faint] %s -- %s;" % (P.pt(0, 0), P.pt(1, 0)))
    b.append("\\node[font=\\scriptsize, anchor=east] at %s {$0$};" % P.pt(-0.01, 0))
    shades = ["ndBlue!40", "ndBlue!55", "ndBlue!80", "ndInk"]
    for q, y, e, col in zip(qs, ys, ey, shades):
        steps = []
        for k, v in enumerate(y):
            steps += [P.P(k * dt, v), P.P((k + 1) * dt, v)]
        b.append(N.polyline(steps, "line width=1.1pt, %s" % col))
        b.append("\\node[font=\\footnotesize, anchor=west, text=%s] at %s {$q=%d$:\\ $\\int\\! a_y^2\\,\\mathrm{d}t=%.3g$};"
                 % (col if col != "ndBlue!40" else "ndBlue!60", P.pt(1.03, y[-1]), q, e))
    b.append("\\node[note, anchor=north west] at %s {the penalised control dies away as $q$ grows;\\\\ the cheap one, $a_x$, works harder: $\\int\\! a_x^2\\,\\mathrm{d}t$ rises\\\\ from $%.1f$ to $%.1f$, its commutator with the drift\\\\ supplying the $y$ rotations};"
             % (P.pt(0.03, -0.74), ex[0], ex[-1]))
    N.write("ch01_penalty", b)
    print("written figures/ch01_penalty.tex  energies of the cheap control:", ["%.2f" % e for e in ex],
          " of the penalised one:", ["%.4f" % e for e in ey])


if __name__ == "__main__":
    main()
