"""ch02_pwc.py -- piecewise-constant control of a qubit.

A control u(t) that is constant on N intervals steers the qubit; the
propagator is the (reversed) product of matrix exponentials.
Writes figures/ch02_pwc.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N

I, X, Y, Z = T.pauli()
H0 = 0.5 * Z                 # drift (qubit frequency omega = 1)
H1 = 0.5 * X                 # control Hamiltonian


def propagate(u, dt):
    """Return the list U(t_0), U(t_1), ..., U(t_N) for piecewise-constant u."""
    U = np.eye(2, dtype=complex)
    Us = [U]
    for u_n in u:                                   # forward in time ...
        U = expm(-1j * (H0 + u_n * H1) * dt) @ U    # ... new factor on the LEFT
        Us.append(U)
    return Us


def figure(u, Us, dt, sx=1.6, su=1.1, sp=2.4, gap=1.0, hi=6):
    """Needham-style drawing: the step control on top, the population below."""
    n, psi0 = len(u), np.array([1.0, 0.0], dtype=complex)
    Tf = n * dt
    y0 = -gap - sp                                   # baseline of the lower panel
    b = []
    # --- upper panel: the control, one pale bar per piece
    for k, uk in enumerate(u):
        x0, x1 = sx * k * dt, sx * (k + 1) * dt
        col = "ndGold!55" if k == hi else "ndSky"
        b.append("\\fill[%s] (%.3f,0) rectangle (%.3f,%.3f);" % (col, x0, x1, su * uk))
    steps = []
    for k, uk in enumerate(u):
        steps += [(sx * k * dt, su * uk), (sx * (k + 1) * dt, su * uk)]
    b.append(N.polyline([(0, 0)] + steps + [(sx * Tf, 0)], "ink"))
    b.append("\\draw[faint, ->, >=stealth] (0,0) -- (%.3f,0) node[right] {$t$};" % (sx * Tf + 0.4))
    b.append("\\draw[faint, ->, >=stealth] (0,%.3f) -- (0,%.3f) node[left] {$u(t)$};"
             % (su * min(u.min(), 0) - 0.2, su * u.max() + 0.5))
    xh = sx * (hi + 0.5) * dt
    b.append("\\node[note, anchor=south] (p) at (%.3f,%.3f) {one piece, one factor:\\\\"
             "$U\\leftarrow\\ee^{-\\imath H(u^{(k)})\\Delta t}\\,U$};" % (xh + 1.4, su * u.max() + 0.35))
    b.append("\\draw[pointer] (p.south west) to[bend right=15] (%.3f,%.3f);" % (xh, su * u[hi] + 0.05))
    # --- lower panel: the population of |1>, exact within each piece
    b.append("\\draw[faint, ->, >=stealth] (0,%.3f) -- (%.3f,%.3f) node[right] {$t$};" % (y0, sx * Tf + 0.4, y0))
    b.append("\\draw[faint, ->, >=stealth] (0,%.3f) -- (0,%.3f);" % (y0, y0 + sp + 0.35))
    b.append("\\draw[dashedink] (0,%.3f) -- (%.3f,%.3f) node[right, font=\\footnotesize] {$1$};"
             % (y0 + sp, sx * Tf, y0 + sp))
    curve = []
    for k, uk in enumerate(u):
        for s in np.linspace(0, dt, 12, endpoint=(k == n - 1)):
            psi = expm(-1j * (H0 + uk * H1) * s) @ Us[k] @ psi0
            curve.append((sx * (k * dt + s), y0 + sp * abs(psi[1]) ** 2))
    b.append("\\fill[ndRose, fill opacity=0.6] (0,%.3f) -- %s -- (%.3f,%.3f) -- cycle;"
             % (y0, " -- ".join(N.pt(p) for p in curve), sx * Tf, y0))
    b.append(N.polyline(curve, "line width=1.3pt, ndRed"))
    for k, U in enumerate(Us):
        b.append("\\fill[ndInk] (%.3f,%.3f) circle (1.3pt);" % (sx * k * dt, y0 + sp * abs((U @ psi0)[1]) ** 2))
    b.append("\\node[left, text=ndRed] at (0,%.3f) {$|\\braket{1|\\psi(t)}|^2$};" % (y0 + 0.6 * sp))
    b.append("\\node[note, anchor=west] (d) at (%.3f,%.3f) {dots: the product of\\\\ exponentials at $t_0,\\dots,t_n$};"
             % (sx * Tf + 0.3, y0 + 0.35 * sp))
    kd = n - 3
    b.append("\\draw[pointer] (d.west) to[bend left=20] (%.3f,%.3f);"
             % (sx * kd * dt, y0 + sp * abs((Us[kd] @ psi0)[1]) ** 2 + 0.06))
    return b


if __name__ == "__main__":
    N_, T_final = 20, 6.0
    dt = T_final / N_
    rng = np.random.RandomState(3)
    u = 1.2 * np.sin(np.linspace(0, np.pi, N_)) + 0.3 * rng.randn(N_)   # some pulse
    Us = propagate(u, dt)
    N.write("ch02_pwc", figure(u, Us, dt))
    print("written figures/ch02_pwc.tex")
