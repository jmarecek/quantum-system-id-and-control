"""ch03_timescales.py -- the Markov approximation compares two timescales.

The bath correlation function C(tau) = exp(-|tau|/tau_B) cos(omega_B tau)
dies out within a few tau_B, while the system, oscillating at omega_S and
decohering on T_2, hardly moves in that time: tau_S >> tau_B.  Within the
window [0, tau_B] the system state is essentially frozen, which is what
allows rho_S(tau) to be replaced by rho_S(t).  Writes figures/ch03_timescales.tex.
"""
import numpy as np
import needham as N

TAU_B, OMEGA_B = 0.12, 16.0                # bath memory time and oscillation
OMEGA_S, T2 = 1.0, 7.0                     # system frequency and coherence time


def main():
    t = np.linspace(0, 10, 800)
    C = np.exp(-t / TAU_B) * np.cos(OMEGA_B * t)
    xs = np.cos(OMEGA_S * t) * np.exp(-t / T2)
    tau_s = 2 * np.pi / OMEGA_S
    assert tau_s / TAU_B > 20                                        # well separated scales
    i = np.searchsorted(t, 3 * TAU_B)
    assert abs(C[i:]).max() < 0.06 and 1 - xs[i] < 0.25              # bath forgot, system barely moved
    Pl = N.Plot((0, 10), (-1.05, 1.15), 11.0, 4.6)
    b = Pl.axes("$t$", "", yticks=[(-1, "$-1$"), (0, "$0$"), (1, "$1$")], yzero=0.0)
    b.append("\\fill[ndGold, fill opacity=0.25] %s rectangle %s;" % (Pl.pt(0, -1.05), Pl.pt(3 * TAU_B, 1.15)))
    b.append(Pl.curve(t, xs, "line width=1.6pt, ndBlue"))
    b.append(Pl.curve(t, C, "line width=1.2pt, ndRed"))
    b.append("\\node[note, anchor=south west, text=ndRed] (c) at %s {bath correlation $C(\\tau)$:\\\\ gone after a few $\\tau_B$};" % Pl.pt(1.1, 0.45))
    b.append("\\draw[pointer] (c.west) to[bend right=20] %s;" % Pl.pt(0.18, C[np.searchsorted(t, 0.18)] + 0.04))
    b.append("\\node[note, anchor=north west, text=ndBlue] (s) at %s {system, e.g.\\ $\\langle\\sigma_x\\rangle$:\\\\ changes on $\\tau_S=2\\pi/\\omega_S\\gg\\tau_B$};" % Pl.pt(5.6, -0.45))
    b.append("\\draw[pointer] (s.north west) to[bend left=15] %s;" % Pl.pt(4.4, xs[np.searchsorted(t, 4.4)] - 0.04))
    b.append("\\node[note, anchor=south] at %s {memory window\\\\ $\\lesssim3\\tau_B$};" % Pl.pt(1.5 * TAU_B, 1.17))
    # the system period
    y = -0.85
    b.append("\\draw[<->, >=stealth, line width=0.6pt, ndBlue] %s -- %s node[midway, below, font=\\scriptsize] {$\\tau_S$};"
             % (Pl.pt(0, y), Pl.pt(tau_s, y)))
    b.append("\\node[note, anchor=west] at %s {inside the window the system is\\\\ frozen: $\\rhoS(\\tau)\\approx\\rhoS(t)$};" % Pl.pt(6.1, 0.95))
    N.write("ch03_timescales", b)
    print("written figures/ch03_timescales.tex; tau_S/tau_B = %.1f" % (tau_s / TAU_B))


if __name__ == "__main__":
    main()
