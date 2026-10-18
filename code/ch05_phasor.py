"""ch05_phasor.py -- a mode is a shrinking, rotating arrow.

lambda = -Gamma + i omega is the transverse pole of the running example
(Gamma = 0.5, omega = 3).
(a) z(t) = e^{lambda t} in the complex plane: an inward spiral.  Position
    vectors at four times; at one of them the velocity dz/dt = lambda z, which
    is the vector turned through arg lambda (a little more than a right angle, so
    that it points slightly inwards) and stretched by |lambda|.
(b) the shadow of the arrow on the real axis, Re z(t) = e^{-Gamma t} cos(omega t),
    against t, between the envelopes +-e^{-Gamma t}; the same four times are marked.
Writes figures/ch05_phasor.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import GAMMA, OMEGA

LAM = -GAMMA + 1j * OMEGA
TIMES = [0.0, 0.35, 0.7, 1.05]
KV = 1                          # index of the time at which the velocity is drawn
VSCALE = 1 / 3                  # the velocity is drawn at a third of its length


def main():
    R = 2.0                                       # cm per unit in panel (a)
    P = lambda z: np.array([R * z.real, R * z.imag])
    ts = np.linspace(0, 6.5, 900)
    z = np.exp(LAM * ts)
    # the velocity is lambda z: compare with a finite difference of the curve
    h = 1e-6
    for t in TIMES:
        assert abs((np.exp(LAM * (t + h)) - np.exp(LAM * (t - h))) / (2 * h) - LAM * np.exp(LAM * t)) < 1e-6
    assert np.isclose(np.angle(LAM), np.pi - np.arctan(OMEGA / GAMMA))
    b = []
    # ---- (a) the complex plane
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[right, text=ndInk] {$\\mathrm{Re}$};" % (-1.25 * R, 1.3 * R))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[above, text=ndInk] {$\\mathrm{Im}$};" % (-1.2 * R, 1.25 * R))
    b.append("\\draw[faint] (0,0) circle (%.3f);" % R)
    b.append("\\node[font=\\scriptsize, below right, text=ndInk!70] at (%.2f,0) {$1$};" % R)
    b.append(N.polyline([P(w) for w in z], "ink"))
    zk = [np.exp(LAM * t) for t in TIMES]
    for k, (t, w) in enumerate(zip(TIMES, zk)):
        b.append("\\draw[vec, ndBlue] (0,0) -- %s;" % N.pt(P(w)))
        b.append("\\draw[dashedink, ndGold] %s -- (%.3f,0);" % (N.pt(P(w)), P(w)[0]))
        b.append("\\fill[ndGold] (%.3f,0) circle (1.9pt);" % P(w)[0])
        b.append("\\node[dot, fill=ndBlue] at %s {};" % N.pt(P(w)))
        pos = ["above right", "right", "above left", "below left"][k]
        b.append("\\node[font=\\scriptsize, text=ndBlue, %s] at %s {$t_%d$};" % (pos, N.pt(P(w)), k))
    # velocity at t_KV, and the angle arg(lambda) between position and velocity
    w = zk[KV]
    v = LAM * w * VSCALE
    b.append("\\draw[dashedink] %s -- %s;" % (N.pt(P(w)), N.pt(P(w * 1.42))))
    b.append("\\draw[vec, ndRed] %s -- %s;" % (N.pt(P(w)), N.pt(P(w + v))))
    a0, a1 = np.degrees(np.angle(w)), np.degrees(np.angle(w) + np.angle(LAM))
    rr = 0.42
    b.append("\\draw[line width=0.6pt, ndRed] ($%s+(%.2f:%.2f)$) arc (%.2f:%.2f:%.2f);"
             % (N.pt(P(w)), a0, rr, a0, a1, rr))
    am = np.radians((a0 + a1) / 2)
    b.append("\\node[font=\\scriptsize, text=ndRed] at %s {$\\arg\\lambda$};"
             % N.pt(P(w) + (rr + 0.38) * np.array([np.cos(am), np.sin(am)])))
    tip = P(w + v)
    b.append("\\node[font=\\small, text=ndRed, above] at %s {$\\lambda\\,\\ee^{\\lambda t_1}$};" % N.pt(tip + np.array([0.0, 0.02])))
    # notes on the left of the plane
    xl = -1.3 * R - 0.15
    b.append("\\node[note, anchor=east] (d) at (%.2f,%.2f) {the velocity (red) is\\\\ $\\lambda$ times the vector:\\\\ it is turned through\\\\ $\\arg\\lambda$ and stretched\\\\ by $\\abs{\\lambda}$ (drawn at $\\tfrac13$)};" % (xl, 1.3))
    b.append("\\draw[pointer] (d.east) to[bend left=25] %s;" % N.pt(P(w + 0.55 * v) + np.array([0.0, 0.08])))
    b.append("\\node[note, anchor=east] (r) at (%.2f,%.2f) {the vector turns\\\\ at the rate $\\omega=\\mathrm{Im}\\,\\lambda$;\\\\ its length $\\ee^{\\alpha t}$ decreases,\\\\ $\\alpha=\\mathrm{Re}\\,\\lambda<0$};" % (xl, -1.25))
    wr = np.exp(LAM * 2.6)
    b.append("\\draw[pointer] (r.east) to[bend right=20] %s;" % N.pt(P(wr)))
    b.append("\\node[note, text=ndGold!80!ndInk, anchor=north] at (%.2f,%.2f) {gold: the projection $\\mathrm{Re}\\,\\ee^{\\lambda t}$ on the real axis};" % (0.0, -1.22 * R))
    b.append("\\node[font=\\small] at (0,%.2f) {(a) $\\ee^{\\lambda t}$ in the complex plane, $\\lambda=-\\Gamma+\\imath\\omega$};" % (-1.22 * R - 0.75))
    # ---- (b) the real part against time
    x0b = 1.3 * R + 1.3
    Pl = N.Plot((0, 6.0), (-1.05, 1.05), 7.6, 4.2, origin=(x0b, -2.1))
    b += Pl.axes("$t$", "$\\mathrm{Re}\\,\\ee^{\\lambda t}$",
                 xticks=[(2 * np.pi / OMEGA, "$\\frac{2\\pi}{\\omega}$"), (4 * np.pi / OMEGA, "$\\frac{4\\pi}{\\omega}$")],
                 yticks=[(-1, "$-1$"), (1, "$1$")], yzero=0.0)
    tt = np.linspace(0, 6.0, 500)
    env = np.exp(-GAMMA * tt)
    b.append(Pl.curve(tt, env, "dashedink"))
    b.append(Pl.curve(tt, -env, "dashedink"))
    re = np.exp(LAM * tt).real
    assert np.allclose(re, np.exp(-GAMMA * tt) * np.cos(OMEGA * tt))
    b.append(Pl.curve(tt, re, "line width=1.3pt, ndGold!85!ndInk"))
    for k, t in enumerate(TIMES):
        b.append("\\fill[ndGold] %s circle (1.9pt);" % Pl.pt(t, np.exp(LAM * t).real))
        pos = ["above right", "above right", "below left", "below"][k]
        b.append("\\node[font=\\scriptsize, text=ndBlue, %s] at %s {$t_%d$};" % (pos, Pl.pt(t, np.exp(LAM * t).real), k))
    b.append("\\node[note, anchor=west] (e) at %s {envelope $\\pm\\ee^{\\alpha t}$:\\\\ the length of the vector};" % Pl.pt(2.6, 0.72))
    b.append("\\draw[pointer] (e.west) to[bend right=20] %s;" % Pl.pt(1.55, np.exp(-GAMMA * 1.55) + 0.02))
    b.append("\\node[note, anchor=west] (c) at %s {$\\ee^{\\alpha t}\\cos\\omega t$: one\\\\ oscillation per turn};" % Pl.pt(3.7, -0.62))
    b.append("\\draw[pointer] (c.west) to[bend left=20] %s;" % Pl.pt(3.15, np.exp(LAM * 3.15).real - 0.03))
    b.append("\\node[font=\\small] at %s {(b) its projection on the real axis, against time};"
             % N.pt(Pl.P(3.0, -1.05) + np.array([0, -0.98])))
    N.write("ch05_phasor", b)
    print("written figures/ch05_phasor.tex; arg(lambda) = %.1f deg, |lambda| = %.3f"
          % (np.degrees(np.angle(LAM)), abs(LAM)))


if __name__ == "__main__":
    main()
