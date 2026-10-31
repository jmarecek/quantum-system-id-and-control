"""ch01_bangbang.py -- Question B: a resonant pulse versus the bang-bang
(time-optimal) solution for a qubit with drift and a bounded control.
H = (omega/2) sigma_z + u(t) sigma_x, |u| <= u0, from |0> to |1>.

Both pulses respect the bound.  The resonant pulse u = u0 cos(omega t) is run
until the population of |1> is largest.  The bang-bang pulse alternates
between +u0 and -u0; its arc lengths are chosen to reach |1> exactly in the
least total time (constrained minimisation over the arc lengths, for one to
five arcs: one or two arcs cannot reach |1>, and more than three do not help).
Writes figures/ch01_bangbang_controls.tex and figures/ch01_bangbang_populations.tex.
"""
import os, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T
import needham as N

I, X, Y, Z = T.pauli()
omega, u0 = 1.0, 0.4
psi0 = np.array([1, 0], dtype=complex)


def step(u, dt):
    """exp(-i (omega/2 Z + u X) dt), in closed form."""
    h = np.array([u, 0.5 * omega])
    a = np.linalg.norm(h)
    return np.cos(a * dt) * I - 1j * np.sin(a * dt) * (h[0] * X + h[1] * Z) / a


def arcs_final(durs, sign=1):
    psi = psi0
    for k, d in enumerate(durs):
        psi = step(sign * (-1) ** k * u0, d) @ psi
    return abs(psi[1]) ** 2


def fastest(k, rng, starts=60):
    """Least total time of k alternating arcs (+u0 first) that reach |1>, or None."""
    best = None
    for _ in range(starts):
        r = minimize(np.sum, rng.uniform(0.2, 3.0, k), method="SLSQP", bounds=[(0, 10)] * k,
                     constraints=[{"type": "eq", "fun": lambda d: arcs_final(d) - 1}],
                     options=dict(maxiter=500, ftol=1e-12))
        if abs(arcs_final(r.x) - 1) < 1e-9 and (best is None or r.fun < best[0] - 1e-9):
            best = (r.fun, r.x)
    return best


def trajectory(u_of_t, T_end, n=1500):
    t = np.linspace(0, T_end, n + 1)
    psi, pop = psi0, [0.0]
    for a, b in zip(t[:-1], t[1:]):
        psi = step(u_of_t(0.5 * (a + b)), b - a) @ psi
        pop.append(abs(psi[1]) ** 2)
    return t, np.array(pop)


def main():
    rng = np.random.default_rng(0)
    sols = {k: fastest(k, rng) for k in range(1, 6)}
    assert sols[1] is None and sols[2] is None                 # one or two arcs cannot reach |1>
    T_star, durs = sols[3]
    assert all(abs(sols[k][0] - T_star) < 1e-4 for k in (4, 5))  # more arcs do not help
    switches = np.cumsum(durs)[:-1]
    bang = lambda t: u0 * (-1) ** int(np.searchsorted(switches, t))
    # the resonant pulse, inside the same bound, run to its largest population of |1>
    t_r, p_r = trajectory(lambda t: u0 * np.cos(omega * t), 10.0, 4000)
    i_r = np.argmax(p_r)
    T_res, P_res = t_r[i_r], p_r[i_r]
    t_b, p_b = trajectory(bang, T_star, 2000)
    assert abs(p_b[-1] - 1) < 1e-6 and T_star < T_res and P_res < 1
    print("bang-bang: arcs %s, T* = %.3f; resonant u0 cos(wt): max population %.4f at t = %.2f (RWA pi/u0 = %.2f)"
          % (np.round(durs, 3), T_star, P_res, T_res, np.pi / u0))

    tmax = 9.4
    # ---- the two controls
    b = []
    Pl = N.Plot((0.0, tmax), (-0.55, 0.55), 5.6, 3.4)
    b += Pl.axes("$t$", "$u(t)$", yzero=0.0,
                 yticks=[(u0, "$u_0$"), (-u0, "$-u_0$")])
    for k in (4, 8):                                     # tick labels above the axis, clear of both pulses
        p = Pl.P(k, 0.0)
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[anchor=south, font=\\scriptsize, text=ndInk] {$%d$};"
                 % (p[0], p[1] - 0.06, p[0], p[1] + 0.06, k))
    b.append("\\fill[ndRose!45] %s rectangle %s;" % (Pl.pt(0, u0), Pl.pt(tmax, 0.55)))
    b.append("\\fill[ndRose!45] %s rectangle %s;" % (Pl.pt(0, -0.55), Pl.pt(tmax, -u0)))
    b.append("\\node[note, text=ndRed!80!ndInk] at %s {forbidden: $\\abs{u}>u_0$};" % Pl.pt(0.5 * tmax, 0.475))
    tt = np.linspace(0, T_res, 400)
    b.append(Pl.curve(tt, u0 * np.cos(omega * tt), "line width=1.1pt, ndBlue"))
    edges = np.concatenate([[0.0], switches, [T_star]])
    pts = []
    for k in range(len(durs)):
        v = u0 * (-1) ** k
        pts += [(edges[k], v), (edges[k + 1], v)]
    b.append(N.polyline([Pl.P(*p) for p in pts], "line width=1.4pt, ndRed"))
    for s in switches:
        b.append("\\draw[ndRed, line width=0.5pt, densely dotted] %s -- %s;" % (Pl.pt(s, u0), Pl.pt(s, -u0)))
    b.append("\\node[note, text=ndRed] at %s {bang-bang: $u=\\pm u_0$, two switches};" % Pl.pt(3.9, -0.475))
    b.append("\\node[note, anchor=west, text=ndBlue] at %s {resonant:\\\\ $u_0\\cos\\omega t$};"
             % Pl.pt(T_res + 0.15, u0 * np.cos(omega * T_res)))
    N.write("ch01_bangbang_controls", b)

    # ---- the populations they produce
    b = []
    Pl = N.Plot((0.0, tmax), (0.0, 1.0), 5.6, 3.4)
    b += Pl.axes("$t$", "$\\abs{\\langle 1|\\psi(t)\\rangle}^2$", xticks=[(k, "$%d$" % k) for k in (2, 4, 6, 8)],
                 yticks=[(0.5, "$\\tfrac12$"), (1, "$1$")])
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(0, 1), Pl.pt(tmax, 1)))
    b.append(Pl.curve(t_r[: i_r + 1], p_r[: i_r + 1], "line width=1.1pt, ndBlue"))
    b.append(Pl.curve(t_b, p_b, "line width=1.4pt, ndRed"))
    for t_end, col in [(T_star, "ndRed"), (T_res, "ndBlue")]:
        b.append("\\draw[%s, line width=0.5pt, densely dotted] %s -- %s;" % (col, Pl.pt(t_end, 0), Pl.pt(t_end, 1)))
    b.append("\\node[dot, fill=ndRed] at %s {};" % Pl.pt(T_star, 1.0))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % Pl.pt(T_res, P_res))
    b.append("\\node[note, anchor=south east, text=ndRed] at %s {bang-bang:\\\\ $\\ket1$ at $t^\\star=%.2f$};" % (Pl.pt(T_star - 0.1, 1.02), T_star))
    b.append("\\node[note, anchor=north west, text=ndBlue] at %s {resonant:\\\\ $%.3f$ at\\\\ $t=%.2f$};" % (Pl.pt(T_res + 0.12, 0.66), P_res, T_res))
    N.write("ch01_bangbang_populations", b)
    print("written figures/ch01_bangbang_*.tex")


if __name__ == "__main__":
    main()
