"""ch01_trap_slice.py -- Question C: a one-parameter slice through the control
landscape of a three-level system with a trap: fidelity along the straight
segment from a local maximum (the trap) to the global optimum.

The system is a ladder with drift H0 = diag(0, 1, 2.2), control coupling
0-1 and 1-2 with strengths 1 and 1.3, and two piecewise-constant segments
of length 1.2; the target is the transfer 0 -> 2.  The landscape is sampled
on a 121 x 121 grid of [-3, 3]^2; the trap is the grid local maximum with
0.3 F_max < F < 0.85 F_max farthest from the global optimum.
Writes figures/ch01_trap_slice.tex.
"""
import os, sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

# three-level ladder, drift H0 = diag(0,1,2.2) (anharmonic), control couples 0-1 and 1-2 with different strengths
H0 = np.diag([0.0, 1.0, 2.2]).astype(complex)
Hc = np.array([[0, 1, 0], [1, 0, 1.3], [0, 1.3, 0]], dtype=complex)
target = np.array([0, 0, 1], dtype=complex)  # transfer 0 -> 2


def fidelity(u1, u2, dt=1.2):
    psi = np.array([1, 0, 0], dtype=complex)
    for u in (u1, u2):
        psi = expm(-1j * (H0 + u * Hc) * dt) @ psi
    return abs(target.conj() @ psi) ** 2


def main():
    g = np.linspace(-3, 3, 121)
    F = np.array([[fidelity(a, b) for a in g] for b in g])
    j, i = np.unravel_index(np.argmax(F), F.shape)
    best = (g[i], g[j])
    # a local maximum that is not global: grid points above all eight neighbours with a clearly lower value
    loc = np.ones_like(F, dtype=bool)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            loc[1:-1, 1:-1] &= F[1:-1, 1:-1] >= np.roll(np.roll(F, dj, 0), di, 1)[1:-1, 1:-1]
    loc[0, :] = loc[-1, :] = loc[:, 0] = loc[:, -1] = False
    cand = np.argwhere(loc & (F < 0.85 * F.max()) & (F > 0.3 * F.max()))
    dist = [abs(g[i0] - best[0]) + abs(g[j0] - best[1]) for j0, i0 in cand]
    j2, i2 = cand[int(np.argmax(dist))]
    trap = (g[i2], g[j2])
    s = np.linspace(-0.3, 1.3, 400)
    line = np.array([fidelity(trap[0] + t * (best[0] - trap[0]), trap[1] + t * (best[1] - trap[1])) for t in s])
    f0, f1 = fidelity(*trap), fidelity(*best)
    inner = (s > 0.05) & (s < 0.95)
    smin, fmin = s[inner][np.argmin(line[inner])], line[inner].min()
    assert f1 > 0.95 and 0.6 < f0 < 0.7                         # a trap well below the optimum
    assert fmin < 0.05                                          # ... separated from it by a deep valley
    near = np.abs(s) < 0.1                                      # the trap is (to grid accuracy) a local maximum along the segment
    assert line[near].max() - f0 < 1e-3 and line[np.argmin(np.abs(s + 0.1))] < f0 and line[np.argmin(np.abs(s - 0.1))] < f0

    b = []
    P = N.Plot((-0.3, 1.3), (0.0, 1.05), 9.0, 4.6)
    b += P.axes("", "$F$", xticks=[(0, "$0$"), (0.5, "$\\tfrac12$"), (1, "$1$")],
                yticks=[(0.5, "$0.5$"), (1, "$1$")])
    b.append("\\node[font=\\small, anchor=north] at %s {position on the segment from the trap ($0$) to the optimum ($1$)};"
             % N.pt(P.P(0.5, 0.0) + np.array([0.0, -0.55])))
    b.append("\\fill[ndSky, fill opacity=0.45] %s rectangle %s;" % (P.pt(0.0, 0.0), P.pt(1.0, 1.0)))
    b.append(P.curve(s, line, "line width=1.1pt, ndBlue"))
    b.append("\\node[dot, fill=ndRed, minimum size=4.5pt] at %s {};" % P.pt(0.0, f0))
    b.append("\\node[dot, fill=ndGreen, minimum size=4.5pt] at %s {};" % P.pt(1.0, f1))
    b.append("\\node[note, anchor=east, text=ndRed] (tr) at %s {a trap: a local maximum,\\\\ $F=%.2f$};" % (P.pt(-0.08, 0.86), f0))
    b.append("\\draw[pointer] (tr.south) to[bend right=20] %s;" % P.pt(-0.01, f0 + 0.03))
    b.append("\\node[note, anchor=west, text=ndGreen] (op) at %s {the global optimum,\\\\ $F=%.2f$};" % (P.pt(1.08, 0.72), f1))
    b.append("\\draw[pointer] (op.north) to[bend right=25] %s;" % P.pt(1.02, f1 - 0.01))
    b.append("\\node[note, anchor=south] (va) at %s {in between, $F$ falls to $%.2f$:\\\\ a gradient method started\\\\ at the trap stays there};" % (P.pt(0.29, 0.72), fmin))
    b.append("\\draw[pointer] (va.south) to[bend right=10] %s;" % P.pt(smin + 0.01, fmin + 0.03))
    N.write("ch01_trap_slice", b)
    print("written figures/ch01_trap_slice.tex  F(trap)=%.3f F(opt)=%.3f  min between %.3f at %.2f" % (F[j2, i2], F[j, i], fmin, smin))


if __name__ == "__main__":
    main()
