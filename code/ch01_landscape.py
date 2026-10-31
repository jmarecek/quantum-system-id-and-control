"""ch01_landscape.py -- a toy control landscape with local traps.

Two-segment piecewise-constant control (u1, u2) of the three-level system of
Control_Technical (H0 = diag(0, 0.515916, 1), V couples 1-2 and 2-3).  The
objective is the fidelity  |Tr(U_target^dagger U(u1,u2))|/3  for a fixed
random target.  The map of contours shows the nonconvex landscape that motivates
global (polynomial) optimisation in Lecture 12.
Run from the course root:  python3 code/ch01_landscape.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm
import needham as N

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


LEVELS = [0.2, 0.4, 0.6, 0.8, 0.95]
BANDS = ["ndPaper", "ndSky!30", "ndSky!80", "ndBlue!28", "ndBlue!45", "ndGold!55"]


def bands(Pl, xs, ys, F):
    """Run-length rectangles of the colour bands of F (rows follow ys, columns xs)."""
    out = []
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    idx = np.searchsorted(LEVELS, F)
    for i, y in enumerate(ys):
        j = 0
        while j < len(xs):
            k = j
            while k + 1 < len(xs) and idx[i, k + 1] == idx[i, j]:
                k += 1
            x0, x1 = max(xs[j] - dx / 2, xs[0]), min(xs[k] + dx / 2, xs[-1])
            y0, y1 = max(y - dy / 2, ys[0]), min(y + dy / 2, ys[-1])
            out.append("\\fill[%s] %s rectangle %s;" % (BANDS[idx[i, j]], Pl.pt(x0, y0), Pl.pt(x1, y1)))
            j = k + 1
    return out


def contour_lines(Pl, xs, ys, F):
    from skimage import measure
    out = []
    for lev in LEVELS[:-1]:
        for path in measure.find_contours(F, lev):
            pts = [(np.interp(cj, np.arange(len(xs)), xs), np.interp(ci, np.arange(len(ys)), ys)) for ci, cj in path]
            out.append(N.polyline([Pl.P(a, c) for a, c in pts], "line width=0.4pt, ndInk!50"))
    return out


def star(p, r=0.16):
    """A five-pointed star (no TikZ shape library is loaded)."""
    a = np.pi / 2 + np.arange(10) * np.pi / 5
    rr = np.where(np.arange(10) % 2 == 0, r, 0.42 * r)
    pts = [np.asarray(p) + q * np.array([np.cos(t), np.sin(t)]) for q, t in zip(rr, a)]
    return "\\fill[ndGold, draw=ndInk, line width=0.3pt] %s -- cycle;" % " -- ".join(N.pt(q) for q in pts)


def local_maxima(F, grid, U_target):
    """The local maxima of F in the square, refined from the grid by Nelder--Mead."""
    from scipy.optimize import minimize
    found = []
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid) - 1):
            if F[i, j] == F[i - 1:i + 2, j - 1:j + 2].max():
                r = minimize(lambda p: -fidelity(propagator(p), U_target), [grid[j], grid[i]], method="Nelder-Mead",
                             options=dict(xatol=1e-8, fatol=1e-12))
                if np.all(np.abs(r.x) < 4) and not any(np.linalg.norm(r.x - q) < 1e-3 for q, _ in found):
                    found.append((r.x, -r.fun))
    return sorted(found, key=lambda t: -t[1])


rng = np.random.RandomState(1)
u_star = rng.uniform(-2, 2, size=segments)      # the target is reachable by construction
U_target = propagator(u_star)

grid = np.linspace(-4, 4, 161)
F = np.array([[fidelity(propagator((u1, u2)), U_target) for u1 in grid] for u2 in grid])
maxima = local_maxima(F, grid, U_target)
(best, f_best), traps = maxima[0], maxima[1:]
assert np.allclose(best, u_star, atol=1e-4) and abs(f_best - 1) < 1e-9
assert len(traps) >= 10 and traps[0][1] > 0.85                  # many traps, the best of them nearly as good
ring = [0.01 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
assert all(fidelity(propagator(p + d), U_target) < f for p, f in maxima for d in ring)   # maxima, not saddles

b = []
Pl = N.Plot((-4.0, 4.0), (-4.0, 4.0), 5.6, 5.6)
b += bands(Pl, grid, grid, F)
b += contour_lines(Pl, grid, grid, F)
b.append("\\draw[ink, line width=0.5pt] %s rectangle %s;" % (Pl.pt(-4, -4), Pl.pt(4, 4)))
for v in (-4, -2, 0, 2, 4):
    b.append("\\draw[faint] %s -- ++(0,-0.08) node[below, font=\\scriptsize, text=ndInk] {$%d$};" % (Pl.pt(v, -4), v))
    b.append("\\draw[faint] %s -- ++(-0.08,0) node[left, font=\\scriptsize, text=ndInk] {$%d$};" % (Pl.pt(-4, v), v))
b.append("\\node at %s {$u_1$};" % N.pt(Pl.P(0, -4) + np.array([0, -0.62])))
b.append("\\node at %s {$u_2$};" % N.pt(Pl.P(-4, 0) + np.array([-0.7, 0])))
for p, f in traps:
    b.append("\\fill[ndRed, draw=white, line width=0.3pt] %s circle (1.9pt);" % Pl.pt(*p))
for p, f in traps[:3]:
    b.append("\\node[font=\\scriptsize, text=ndRed, fill=white, fill opacity=0.8, text opacity=1, inner sep=0.6pt, above=1.5pt] at %s {$%.2f$};"
             % (Pl.pt(*p), f))
b.append(star(Pl.P(*best)))
# notes and the key of the bands, to the right
x0 = Pl.P(4, 0)[0] + 0.4
b.append("\\node[note, anchor=north west, align=left] at (%.2f,%.2f) {star: the target,\\\\ $F=1$ at $(%.2f,%.2f)$};"
         % (x0, Pl.P(0, 4)[1], *best))
b.append("\\node[note, anchor=north west, align=left, text=ndRed] at (%.2f,%.2f) {red: %d local maxima,\\\\ traps for a gradient\\\\"
         " method; the best\\\\ three reach $F=%.2f$,\\\\ $%.2f$ and $%.2f$};" % (x0, Pl.P(0, 2.3)[1], len(traps), *[f for _, f in traps[:3]]))
edges = [0.0] + LEVELS + [1.0]
for k, col in enumerate(BANDS):
    y = Pl.P(0, -4)[1] + 0.32 * k
    b.append("\\fill[%s, draw=ndInk!50, line width=0.3pt] (%.2f,%.2f) rectangle ++(0.42,0.28);" % (col, x0, y))
    b.append("\\node[font=\\scriptsize, anchor=west] at (%.2f,%.2f) {$%s\\le F<%s$};"
             % (x0 + 0.48, y + 0.14, ("%g" % edges[k]), ("%g" % edges[k + 1]) if k < len(BANDS) - 1 else "1"))
N.write("ch01_landscape", b)
print("wrote ch01_landscape.tex; max fidelity on grid %.3f at u*=%s; %d traps, best %s"
      % (F.max(), np.round(u_star, 2), len(traps), np.round([f for _, f in traps[:3]], 3)))
