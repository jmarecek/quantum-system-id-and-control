"""ch01_trap_slice.py -- Question B: a one-parameter slice through the control
landscape of the three-level system with a trap: fidelity along a line through the
trap and the global optimum.  Writes figures/ch01_trap_slice.tex."""
import os, sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

# three-level ladder, drift H0 = diag(0,1,2.2) (anharmonic), control couples 0-1 and 1-2 with different strengths
H0 = np.diag([0.0, 1.0, 2.2]).astype(complex)
Hc = np.array([[0, 1, 0], [1, 0, 1.3], [0, 1.3, 0]], dtype=complex)
target = np.array([0, 0, 1], dtype=complex)  # transfer 0 -> 2
def fidelity(u1, u2, dt=1.2):
    psi = np.array([1, 0, 0], dtype=complex)
    for u in (u1, u2):
        psi = expm(-1j * (H0 + u * Hc) * dt) @ psi
    return abs(target.conj() @ psi) ** 2
g = np.linspace(-3, 3, 121)
F = np.array([[fidelity(a, b) for a in g] for b in g])
j, i = np.unravel_index(np.argmax(F), F.shape); best = (g[i], g[j])
# a local maximum that is not global: grid points above all eight neighbours with a clearly lower value
loc = np.ones_like(F, dtype=bool)
for dj in (-1, 0, 1):
    for di in (-1, 0, 1):
        if di == 0 and dj == 0: continue
        loc[1:-1, 1:-1] &= F[1:-1, 1:-1] >= np.roll(np.roll(F, dj, 0), di, 1)[1:-1, 1:-1]
loc[0, :] = loc[-1, :] = loc[:, 0] = loc[:, -1] = False
cand = np.argwhere(loc & (F < 0.85 * F.max()) & (F > 0.3 * F.max()))
dist = [abs(g[i0] - best[0]) + abs(g[j0] - best[1]) for j0, i0 in cand]
j2, i2 = cand[int(np.argmax(dist))]; trap = (g[i2], g[j2])
s = np.linspace(-0.3, 1.3, 400)
line = [fidelity(trap[0] + t * (best[0] - trap[0]), trap[1] + t * (best[1] - trap[1])) for t in s]
body = T.pgfplots_axis([dict(x=s, y=line, label=r"$F$ along the segment trap $\rightarrow$ optimum")],
                       xlabel=r"position on the segment ($0=$ trap, $1=$ global optimum)", ylabel="fidelity $F$",
                       width="0.8\\linewidth", height="0.4\\linewidth", legend_pos="north west",
                       axis_options="ymin=0, ymax=1.05")
body = body.replace(r"\end{axis}", r"\addplot[only marks, mark=*, qocD] coordinates {(0,%.4f)} node[above right, font=\scriptsize] {local trap};" % line[np.argmin(np.abs(s))] +
                    "\n" + r"\addplot[only marks, mark=*, qocC] coordinates {(1,%.4f)} node[above left, font=\scriptsize] {global optimum};" % line[np.argmin(np.abs(s - 1))] +
                    "\n" + r"\end{axis}")
T.write_tikz("ch01_trap_slice", body)
print("written figures/ch01_trap_slice.tex  F(trap)=%.3f F(opt)=%.3f" % (F[j2, i2], F[j, i]))
