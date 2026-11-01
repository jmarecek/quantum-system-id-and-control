"""ch09_hamiltonian.py -- what the maximisation of the Pontryagin Hamiltonian picks.

The control enters H_P through the switching function b = H_x, and the bound
is U = [-u0, u0] with u0 = 1.5.
(a) Cost of time (or none), H_P = c + b u, affine in u: the maximum lies at an
    end, u = u0 sgn b (drawn for b = 0.7 and, faintly, b = -0.7).
(b) Cost of energy, F0 = u^2/2 with lambda_0 = -1, H_P = c + b u - u^2/2,
    concave: the maximum is u = b, clipped at the bound (drawn for b = 0.8,
    interior, and b = 1.9, clipped at u0).
(c) b = 0 on an interval: H_P does not depend on u, every u is a maximum,
    and the principle says nothing; the arc is singular.
(d) The rule of (b) at work.  The driven qubit of Chapter 4 (omega = 1,
    Delta = -1), transferred from |0> to |1> in the fixed time T = 3.8 with
    |u| <= 1.5 and the least energy (1/2) int u^2 dt.  The control was
    computed by SLSQP on 120 constant steps, starting from the bang-bang
    control of fig:ch09:timeoptimal stretched to T; it is a local optimum, and it is an
    extremal: with the Bloch costate p(T) = (mu_x, mu_y, 0) of the endpoint
    constraint carried back along the trajectory, the switching function is
    H_x(t) = p . (x x r), and u = clip(H_x, -u0, u0) on every step, to 3e-5,
    once mu is fitted on the steps that do not touch the bound.  The control
    lies on the bound at the beginning and the end (43 of the 120 steps) and
    follows H_x in between; with T = 4 it would not touch the bound at all,
    and as T decreases to t* = 3.485 it becomes the bang-bang control.
Writes figures/ch09_hamiltonian.tex.
"""
import numpy as np
from scipy.optimize import minimize
import needham as N

U0 = 1.5
DELTA = -1.0
TE = 3.8
NSTEP = 120
TSTAR = 2 * np.pi / np.hypot(U0, DELTA)


def rot(n, th):
    n = np.asarray(n, float) / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def path(us, T=TE):
    dt = T / len(us)
    r, Rs = [np.array([0, 0, 1.0])], []
    for u in us:
        R = rot([u, 0, DELTA], np.hypot(u, DELTA) * dt)
        Rs.append(R)
        r.append(R @ r[-1])
    return np.array(r), Rs


def end_jacobian(us, T=TE, h=1e-7):
    """d r(T)/d u_k: the suffix product of the later steps applied to the derivative of step k."""
    dt = T / len(us)
    r, Rs = path(us, T)
    J = np.zeros((3, len(us)))
    M = np.eye(3)
    for k in range(len(us) - 1, -1, -1):
        dR = (rot([us[k] + h, 0, DELTA], np.hypot(us[k] + h, DELTA) * dt)
              - rot([us[k] - h, 0, DELTA], np.hypot(us[k] - h, DELTA) * dt)) / (2 * h)
        J[:, k] = M @ (dR @ r[k])
        M = M @ Rs[k]
    return J


def energy_optimal(T=TE, n=NSTEP):
    """Least-energy transfer |0> -> |1> in the time T with |u| <= U0, by SLSQP from the stretched bang-bang control."""
    dt = T / n
    t1 = (np.pi - np.arccos((DELTA / U0) ** 2)) / np.hypot(U0, DELTA)
    x0 = np.where((np.arange(n) + 0.5) * dt < t1 * T / TSTAR, -U0, U0)
    cons = [{"type": "eq", "fun": lambda us: path(us, T)[0][-1][:2], "jac": lambda us: end_jacobian(us, T)[:2]}]
    r = minimize(lambda us: 0.5 * np.sum(us ** 2) * dt, x0, jac=lambda us: us * dt, constraints=cons,
                 bounds=[(-U0, U0)] * n, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14})
    assert r.success
    return r.x


def switching(us, T=TE):
    """H_x on each step for the costates p(T) = e_x and e_y, by the midpoint of the step; and the fitted mu."""
    n = len(us)
    dt = T / n
    r, Rs = path(us, T)
    S = []
    for pT in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])):
        p = [None] * (n + 1)
        p[n] = pT
        for k in range(n - 1, -1, -1):
            p[k] = Rs[k].T @ p[k + 1]
        s = []
        for k in range(n):
            Rh = rot([us[k], 0, DELTA], np.hypot(us[k], DELTA) * dt / 2)
            s.append((Rh.T @ p[k + 1]) @ np.cross([1, 0, 0], Rh @ r[k]))
        S.append(s)
    S = np.array(S).T
    free = np.abs(us) < U0 - 1e-6
    mu = np.linalg.lstsq(S[free], us[free], rcond=None)[0]
    return S @ mu, mu, free


def checks():
    u = np.linspace(-U0, U0, 3001)
    for b in (0.7, -0.7):                                       # affine: the end with the sign of b
        assert np.isclose(u[np.argmax(b * u)], U0 * np.sign(b))
    for b in (0.8, 1.9):                                        # quadratic: b, clipped
        assert abs(u[np.argmax(b * u - u ** 2 / 2)] - np.clip(b, -U0, U0)) < 1e-3
    v = np.random.default_rng(2).uniform(-U0, U0, NSTEP)              # the Jacobian of the end point
    J = end_jacobian(v)
    for k in (0, 37, 119):
        e = np.zeros(NSTEP)
        e[k] = 1e-6
        assert np.allclose((path(v + e)[0][-1] - path(v - e)[0][-1]) / 2e-6, J[:, k], atol=1e-6)
    us = energy_optimal()
    r, _ = path(us)
    assert np.allclose(r[-1], [0, 0, -1], atol=1e-7)
    Hx, mu, free = switching(us)
    assert np.abs(us - np.clip(Hx, -U0, U0)).max() < 1e-4        # the maximum principle, step by step
    assert (~free).sum() == 43 and np.abs(Hx).max() > U0
    assert np.all(~free[:5]) and np.all(~free[-5:])             # on the bound at both ends
    # with T = 4 the least-energy control stays inside the bound
    u4 = energy_optimal(4.0)
    assert np.abs(u4).max() < U0 - 1e-3
    return us, Hx, free


def small_plot(x0, name, curves, mark, note, cap, ylim=(-2.4, 2.4)):
    """One panel of H_P(u) over U = [-u0, u0]."""
    pl = N.Plot((-U0, U0), ylim, 3.3, 2.6, origin=(x0, 4.3))
    b = ["\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$u$};" % (pl.pt(-U0 - 0.25, 0), pl.pt(U0 + 0.3, 0)),
         "\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$\\HP$};" % (pl.pt(0, ylim[0]), pl.pt(0, ylim[1] + 0.25))]
    b.append("\\fill[ndSky, fill opacity=0.5] %s rectangle %s;" % (pl.pt(-U0, ylim[0]), pl.pt(U0, ylim[1])))
    for v, lab in ((-U0, "$-u_0$"), (U0, "$u_0$")):
        b.append("\\draw[faint] %s -- %s;" % (pl.pt(v, ylim[0]), pl.pt(v, ylim[1])))
        b.append("\\node[font=\\scriptsize, anchor=north] at %s {%s};" % (pl.pt(v, ylim[0]), lab))
    b.append("\\begin{scope}")
    b.append("\\clip %s rectangle %s;" % (pl.pt(-U0 - 0.2, ylim[0]), pl.pt(U0 + 0.2, ylim[1])))
    u = np.linspace(-U0, U0, 121)
    for f, sty in curves:
        b.append(pl.curve(u, f(u), sty))
    b.append("\\end{scope}")
    for (x, y, col) in mark:
        b.append("\\node[dot, fill=%s] at %s {};" % (col, pl.pt(x, y)))
    b += note(pl)
    b.append("\\node[font=\\small, anchor=north, align=center] at %s {%s};" % (pl.pt(0, ylim[0] - 0.75), cap))
    return b


def main():
    us, Hx, free = checks()
    b = []
    gold, faint = "line width=1.3pt, ndBlue", "line width=0.8pt, ndBlue!40"
    # ---- (a) affine
    b += small_plot(0.6, "a", [(lambda u: -0.7 * u, faint), (lambda u: 0.7 * u, gold)],
                    [(U0, 0.7 * U0, "ndRed"), (-U0, 0.7 * U0, "ndRed!45")],
                    lambda pl: ["\\node[note, anchor=north west, align=left] at %s {$b>0$: $u=u_0$\\\\ $b<0$: $u=-u_0$};" % pl.pt(-0.62, -1.02),
                                "\\node[note, anchor=south, text=ndRed] at %s {maximum\\\\ at an end};" % pl.pt(0.45, 1.15)],
                    "(a) cost of time:\\\\ $\\HP=c+bu$, bang-bang")
    # ---- (b) quadratic
    b += small_plot(5.6, "b", [(lambda u: 0.8 * u - u ** 2 / 2, gold), (lambda u: 1.9 * u - u ** 2 / 2, "line width=1.3pt, ndGreen")],
                    [(0.8, 0.32, "ndRed"), (U0, 1.9 * U0 - U0 ** 2 / 2, "ndRed")],
                    lambda pl: ["\\node[note, anchor=north west, text=ndBlue] at %s {$b=0.8$: $u=b$};" % pl.pt(-1.45, 2.25),
                                "\\node[note, anchor=north west, text=ndGreen, align=left] at %s {$b=1.9$:\\\\ clipped, $u=u_0$};" % pl.pt(0.12, -0.55)],
                    "(b) cost of energy:\\\\ $\\HP=c+bu-\\tfrac12u^2$")
    # ---- (c) flat
    b += small_plot(10.6, "c", [(lambda u: 0.9 + 0 * u, "line width=2.2pt, ndRed")], [],
                    lambda pl: ["\\node[note, anchor=north] at %s {$b=0$: every $u$ is\\\\ a maximum; singular};" % pl.pt(0, 0.45)],
                    "(c) $b\\equiv0$ on an interval:\\\\ the principle is silent")

    # ---- (d) the energy-optimal transfer of the qubit
    dt = TE / NSTEP
    tm = (np.arange(NSTEP) + 0.5) * dt
    pl = N.Plot((0, TE), (-2.3, 2.3), 9.6, 2.6, origin=(2.4, 0.0))
    b += pl.axes(xlabel="{\\scriptsize $t$}", xticks=[(1, "$1$"), (2, "$2$"), (3, "$3$"), (TE, "$T$")],
                 yticks=[(-U0, "$-u_0$"), (U0, "$u_0$")], yzero=0)
    # shade the steps on the bound
    idx = np.where(~free)[0]
    for sg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
        b.append("\\fill[ndRose, fill opacity=0.7] %s rectangle %s;" % (pl.pt(sg[0] * dt, -2.3), pl.pt((sg[-1] + 1) * dt, 2.3)))
    for v in (-U0, U0):
        b.append("\\draw[dashedink] %s -- %s;" % (pl.pt(0, v), pl.pt(TE, v)))
    steps_t = np.repeat(np.arange(NSTEP + 1) * dt, 2)[1:-1]
    steps_u = np.repeat(us, 2)
    b.append(pl.curve(steps_t, steps_u, "line width=1.3pt, ndBlue"))
    keep = np.abs(Hx) < 2.3
    for sg in np.split(np.where(keep)[0], np.where(np.diff(np.where(keep)[0]) != 1)[0] + 1):
        b.append(pl.curve(tm[sg], Hx[sg], "line width=1.0pt, ndRed"))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=north] at %s {$u(t)$};" % pl.pt(3.3, 1.35))
    j = int(np.argmin(np.abs(tm - 0.25)))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at %s {$H_x(t)$};" % pl.pt(0.45, -2.05))
    b.append("\\node[note, anchor=west] at %s {on the bound where $\\abs{H_x}>u_0$ (shaded);\\\\ elsewhere $u=H_x$};" % pl.pt(TE + 0.2, 0.6))
    b.append("\\node[note, anchor=west] at %s {least energy, $T=3.8$, $\\abs u\\le1.5$};" % pl.pt(TE + 0.2, -0.8))
    b.append("\\node[font=\\small, anchor=north] at %s {(d) the rule of (b) in the energy-optimal transfer of our qubit};" % pl.pt(TE / 2 + 0.6, -2.75))
    N.write("ch09_hamiltonian", b)
    return us, free


if __name__ == "__main__":
    us, free = main()
    print("energy %.4f, steps on the bound %d of %d" % (0.5 * np.sum(us ** 2) * TE / NSTEP, (~free).sum(), NSTEP))
    print("written figures/ch09_hamiltonian.tex")
