"""ch09_landscape.py -- a toy quantum control landscape.

Two piecewise-constant pulses of duration dt each act on a qubit,
    U(u1,u2) = exp(-i(Delta sz + u2 sx)dt/2) exp(-i(Delta sz + u1 sx)dt/2),
and the objective is the transfer fidelity F = |<down|U|up>|^2.
We tabulate F on a grid, locate its critical points numerically, and export
  figures/ch09_landscape.tex   (heat map with critical points marked).
"""
import sys, os
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

I2, SX, SY, SZ = T.pauli()
Delta, dt = 0.6, 1.6
up, down = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)


def fidelity(u1, u2):
    U1 = expm(-0.5j * (Delta * SZ + u1 * SX) * dt)
    U2 = expm(-0.5j * (Delta * SZ + u2 * SX) * dt)
    return abs(down.conj() @ U2 @ U1 @ up) ** 2


def gradient(u1, u2, h=1e-5):
    """Central finite-difference gradient of the fidelity."""
    return np.array([(fidelity(u1 + h, u2) - fidelity(u1 - h, u2)) / (2 * h),
                     (fidelity(u1, u2 + h) - fidelity(u1, u2 - h)) / (2 * h)])


grid = np.linspace(-3.0, 3.0, 61)
F = np.array([[fidelity(a, b) for a in grid] for b in grid])   # rows = u2

# critical points: interior grid points where F is a local max/min/saddle
crit = []
for j in range(1, len(grid) - 1):
    for i in range(1, len(grid) - 1):
        nb = F[j-1:j+2, i-1:i+2]
        c = F[j, i]
        if c >= nb.max() - 1e-12:
            crit.append((grid[i], grid[j], "max"))
        elif c <= nb.min() + 1e-12:
            crit.append((grid[i], grid[j], "min"))

body = T.pgfplots_matrix(F, xlabel="$u_1$", ylabel="$u_2$", width="0.55\\linewidth",
                         axis_options="xtick={0,15,30,45,60}, xticklabels={-3,-1.5,0,1.5,3}, "
                                      "ytick={0,15,30,45,60}, yticklabels={-3,-1.5,0,1.5,3}, "
                                      "title={transfer fidelity $F(u_1,u_2)$}")
marks = []
for (a, b, kind) in crit:
    i, j = np.argmin(abs(grid - a)), np.argmin(abs(grid - b))
    style = "white, mark=*" if kind == "max" else "red, mark=x"
    marks.append("\\addplot[only marks, %s, mark size=2pt] coordinates {(%d,%d)};" % (style, i, j))
body = body.replace("\\end{axis}", "\n".join(marks) + "\n\\end{axis}")
T.write_tikz("ch09_landscape", body)
print("critical points found:", len(crit), " max F on grid = %.4f" % F.max())
