"""ch02_speed.py -- fixed time versus minimum time: a quantum speed limit.

Qubit, no drift, drive H = (u_x sx + u_y sy)/2 with u_x^2 + u_y^2 <= M^2,
start |0>, target |1>.  The reachable states at time T form the cap of angular
radius M T about |0> (Figure ch02_reach), so the best fidelity at time T is
    F*(T) = cos^2((pi - M T)/2) = sin^2(M T / 2)   for M T <= pi,  and 1 after.
We check this against random bounded piecewise-constant controls: none beats
the bound, and the resonant pulse u_x = M attains it.
Writes figures/ch02_speed.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N

M = 1.0
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]])
SX, SY = 7.0 / 1.4, 3.6                    # page units per unit of M T / pi and of fidelity
rng = np.random.default_rng(3)


def best(T):
    return np.sin(min(M * T, np.pi) / 2) ** 2


def fidelity(controls, T):
    psi = np.array([1, 0], complex)
    dt = T / len(controls)
    for ux, uy in controls:
        psi = expm(-0.5j * (ux * sx + uy * sy) * dt) @ psi
    return abs(psi[1]) ** 2


def P(mt, f):
    return (SX * mt / np.pi, SY * f)


def main():
    b = []
    ts = np.linspace(0, 1.4 * np.pi, 200) / M
    curve = [P(M * t, best(t)) for t in ts]
    b.append("\\fill[ndSky, fill opacity=0.6] (0,0) -- %s -- (%.3f,0) -- cycle;"
             % (" -- ".join(N.pt(p) for p in curve), curve[-1][0]))
    b.append("\\fill[ndRose, fill opacity=0.45] (0,0) -- %s -- (%.3f,%.3f) -- (0,%.3f) -- cycle;"
             % (" -- ".join(N.pt(p) for p in curve), curve[-1][0], SY * 1.12, SY * 1.12))
    # random bounded controls
    pts = []
    for k in range(300):
        T = rng.uniform(0.05, 1.4 * np.pi) / M
        if k % 2:                                  # fully random directions and amplitudes
            a = rng.uniform(0, 2 * np.pi, 6)
            r = M * np.sqrt(rng.uniform(0, 1, 6))
        else:                                      # perturbed resonant pulses
            a = rng.normal(0, 0.5, 6)
            r = M * rng.uniform(0.7, 1.0, 6)
        f = fidelity(np.stack([r * np.cos(a), r * np.sin(a)], 1), T)
        assert f <= best(T) + 1e-9
        pts.append(P(M * T, f))
    for p in pts:
        b.append("\\fill[ndInk!45] %s circle (0.9pt);" % N.pt(p))
    # the bound, attained by the resonant pulse
    for t in (0.3, 1.1, 2.5):
        assert np.isclose(fidelity([(M, 0.0)] * 4, t), best(t))
    b.append(N.polyline(curve, "line width=1.6pt, ndBlue"))
    # axes
    b.append("\\draw[faint, ->, >=stealth] (0,0) -- (%.2f,0) node[below] {$T$};" % (SX * 1.45))
    b.append("\\draw[faint, ->, >=stealth] (0,0) -- (0,%.2f) node[left] {fidelity};" % (SY * 1.18))
    for f, lab in [(0, "0"), (0.5, "\\tfrac12"), (1, "1")]:
        b.append("\\draw[faint] (-0.08,%.2f) -- (0.08,%.2f) node[left, xshift=-3pt, font=\\scriptsize, text=ndInk] {$%s$};" % (SY * f, SY * f, lab))
    b.append("\\draw[dashedink] (0,%.2f) -- (%.2f,%.2f);" % (SY, SX * 1.4, SY))
    # minimum time, exact and with tolerance
    b.append("\\node[dot, fill=ndRed, minimum size=4.4pt] at %s {};" % N.pt(P(np.pi, 1)))
    b.append("\\draw[dashedink] %s -- (%.3f,0) node[below, font=\\scriptsize, text=ndInk] {$T_{\\min}=\\pi/M$};"
             % (N.pt(P(np.pi, 1)), SX))
    eps2 = 0.05
    te = 2 * np.arcsin(np.sqrt(1 - eps2)) / M
    b.append("\\draw[line width=0.6pt, ndRed!70, dash pattern=on 3pt off 2pt] (0,%.3f) -- %s;" % (SY * (1 - eps2), N.pt(P(M * te, 1 - eps2))))
    b.append("\\node[font=\\scriptsize, text=ndRed, left] at (0,%.3f) {$1-\\varepsilon^2$};" % (SY * (1 - eps2) - 0.18))
    b.append("\\node[dot, fill=ndRed!70] at %s {};" % N.pt(P(M * te, 1 - eps2)))
    # a fixed-time slice
    tf = 0.55 * np.pi / M
    b.append("\\draw[line width=0.8pt, ndGreen] (%.3f,0) -- %s;" % (P(M * tf, 0)[0], N.pt(P(M * tf, best(tf)))))
    b.append("\\node[dot, fill=ndGreen] at %s {};" % N.pt(P(M * tf, best(tf))))
    b.append("\\node[font=\\scriptsize, below, text=ndGreen!70!black] at (%.3f,0) {$T$ fixed};" % P(M * tf, 0)[0])
    # notes
    b.append("\\node[note, anchor=west, text=ndGreen!60!black] (f) at (%.2f,%.2f) {(1) fixed time: the best\\\\ fidelity at this $T$};" % (SX * 0.02, SY * 0.78))
    b.append("\\draw[pointer] (f.east) to[bend left=15] %s;" % N.pt(P(M * tf, best(tf))))
    b.append("\\node[note, anchor=west, text=ndRed] (m) at (%.2f,%.2f) {(3) minimum time: the first $T$\\\\ at which the target is reached};"
             % (SX * 1.47, SY * 0.85))
    b.append("\\draw[pointer] (m.north west) to[bend right=20] %s;" % N.pt(P(np.pi, 1) + np.array([0.08, 0.05])))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {\\emph{unreachable}: above\\\\ the speed limit $\\sin^2(MT/2)$};" % (SX * 0.12, SY * 1.03 + 0.25))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {grey: random controls\\\\ with $|u|\\le M$; none\\\\ beats the blue curve};" % (SX * 1.47, SY * 0.35))
    N.write("ch02_speed", b)
    print("written figures/ch02_speed.tex; T_min(eps) = %.3f vs T_min = %.3f" % (te, np.pi / M))


if __name__ == "__main__":
    main()
