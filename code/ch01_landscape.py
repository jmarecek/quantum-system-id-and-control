"""ch01_landscape.py -- a toy control landscape with local traps.

Two-segment piecewise-constant control (u1, u2) of the three-level system of
Control_Technical (H0 = diag(0, 0.515916, 1), V couples 1-2 and 2-3).  The
objective is the fidelity  |Tr(U_target^dagger U(u1,u2))|/3  for a fixed
random target.  The heat map shows the nonconvex landscape that motivates
global (polynomial) optimisation in Lecture 12.
Run from the course root:  python3 code/ch01_landscape.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm
import tikzexport as T

H0 = np.diag([0.0, 0.515916, 1.0])
V = np.array([[0, 1 / np.sqrt(2), 0], [1 / np.sqrt(2), 0, 1], [0, 1, 0]])
T_final, segments = 3.0, 2
dt = T_final / segments


def propagator(u):
    """U(T) for piecewise-constant control amplitudes u = (u_1, ..., u_m)."""
    U = np.eye(3, dtype=complex)
    for uk in u:                       # later segments multiply from the left
        U = expm(-1j * (H0 + uk * V) * dt) @ U
    return U


def fidelity(U, U_target):
    return abs(np.trace(U_target.conj().T @ U)) / 3.0


rng = np.random.RandomState(1)
u_star = rng.uniform(-2, 2, size=segments)      # the target is reachable by construction
U_target = propagator(u_star)

grid = np.linspace(-4, 4, 81)
F = np.array([[fidelity(propagator((u1, u2)), U_target) for u1 in grid] for u2 in grid])

body = T.pgfplots_matrix(F, xlabel="$u_1$", ylabel="$u_2$", width="0.55\\linewidth",
                         axis_options="xtick={0,20,40,60,80}, xticklabels={-4,-2,0,2,4}, "
                                      "ytick={0,20,40,60,80}, yticklabels={-4,-2,0,2,4}, "
                                      "colorbar style={title=$F$, font=\\footnotesize}")
T.write_tikz("ch01_landscape", body)
print("wrote ch01_landscape.tex; max fidelity on grid %.3f at u*=%s" % (F.max(), np.round(u_star, 2)))
