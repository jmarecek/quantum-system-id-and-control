"""ch05_winding.py -- how a Fourier transform finds a frequency.

The signal is y(t) = e^{-Gamma t} cos(omega t), the shape of the <sigma_x> data
of the running example (Gamma = 0.5, omega = 3).  Multiplying it by e^{-i nu t}
winds it around the origin at the probe frequency nu: the curve y(t) e^{-i nu t}
in the complex plane.  Adding up all its points gives the arrow
    yhat(nu) = int_0^infty y(t) e^{-i nu t} dt
             = 1/2 [ 1/(Gamma + i(nu - omega)) + 1/(Gamma + i(nu + omega)) ].
Far from omega the wound curve is balanced around 0 and the arrow is short; at
nu = omega every turn of the signal is turned back by the probe, the curve stays
on one side, and the arrow is long.  The last panel plots |yhat(nu)| against
nu: the spectrum, with the three probes marked.  Writes figures/ch05_winding.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import GAMMA, OMEGA

PROBES = [(1.0, "ndBlue", "far from $\\omega$"), (2.5, "ndGold", "near $\\omega$"), (OMEGA, "ndRed", "at $\\omega$")]


def yhat(nu):
    return 0.5 * (1 / (GAMMA + 1j * (nu - OMEGA)) + 1 / (GAMMA + 1j * (nu + OMEGA)))


def main():
    t = np.linspace(0, 40, 400001)
    y = np.exp(-GAMMA * t) * np.cos(OMEGA * t)
    for nu, _, _ in PROBES:
        w = y * np.exp(-1j * nu * t)
        integral = np.sum((w[1:] + w[:-1]) / 2 * np.diff(t))
        assert abs(integral - yhat(nu)) < 1e-6
    w_on = y * np.exp(-1j * OMEGA * t)
    assert w_on.real.min() >= -1e-12                     # on resonance the curve stays in Re >= 0
    nus = np.linspace(0, 6, 601)
    mag = np.abs(yhat(nus))
    assert abs(nus[np.argmax(mag)] - OMEGA) < 0.05
    b = []
    R = 1.45                                             # cm per unit in the small planes
    tt = np.linspace(0, 10, 3000)
    yy = np.exp(-GAMMA * tt) * np.cos(OMEGA * tt)
    notes = ["the contributions cancel:\\\\ a short arrow", "partial cancellation", "the rotation is undone:\\\\ the curve stays on one side,\\\\ a long arrow"]
    for j, (nu, col, word) in enumerate(PROBES):
        cx = j * 3.65
        P = lambda z, cx=cx: np.array([cx + R * z.real, R * z.imag])
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (N.pt(P(-1.1 + 0j)), N.pt(P(1.25 + 0j))))
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (N.pt(P(-1.1j)), N.pt(P(1.15j))))
        wz = yy * np.exp(-1j * nu * tt)
        b.append(N.polyline([P(z) for z in wz], "line width=0.7pt, ndInk!80"))
        a = yhat(nu)
        b.append("\\draw[vec, %s, line width=1.6pt] %s -- %s;" % (col, N.pt(P(0j)), N.pt(P(a))))
        b.append("\\node[dot, fill=%s, minimum size=4.5pt] at %s {};" % (col, N.pt(P(a))))
        b.append("\\node[font=\\small, text=%s, anchor=north] at %s {$\\nu=%s$: %s};"
                 % (col, N.pt(P(-1.2j)), "%g" % nu if nu != OMEGA else "\\omega", word))
        b.append("\\node[note, anchor=north] at %s {%s};" % (N.pt(P(-1.62j)), notes[j]))
    b.append("\\node[note, anchor=south, align=center] at (3.65,1.85) {the signal $y(t)$ wound about $0$ at the trial frequency $\\nu$, that is, the curve $y(t)\\,\\ee^{-\\imath\\nu t}$;\\\\ the arrow is the sum of all its points: $\\hat y(\\nu)=\\int_0^\\infty y(t)\\,\\ee^{-\\imath\\nu t}\\,dt$};")
    # ---- the spectrum
    Pl = N.Plot((0, 6.0), (0, 1.1), 4.4, 3.4, origin=(10.9, -1.7))
    b += Pl.axes("$\\nu$", "$\\abs{\\hat y(\\nu)}$", xticks=[(OMEGA, "$\\omega$"), (6, "$6$")],
                 yticks=[(0.5, "$0.5$"), (1, "$1$")])
    b.append(Pl.curve(nus, mag, "ink"))
    for nu, col, _ in PROBES:
        b.append("\\draw[dashedink, %s] %s -- %s;" % (col, Pl.pt(nu, 0), Pl.pt(nu, abs(yhat(nu)))))
        b.append("\\node[dot, fill=%s, minimum size=4.5pt] at %s {};" % (col, Pl.pt(nu, abs(yhat(nu)))))
    b.append("\\node[note, anchor=west] at %s {the length of the\\\\ arrow for each\\\\ trial $\\nu$: the\\\\ spectrum};" % Pl.pt(3.55, 0.75))
    b.append("\\node[note, anchor=north] at %s {a peak at $\\omega$, its\\\\ width governed by $\\Gamma$};" % Pl.pt(3.0, -0.3))
    b.append("\\node[font=\\small] at (3.65,-3.75) {(a) the signal wound at three trial frequencies};")
    b.append("\\node[font=\\small] at %s {(b) the spectrum};" % N.pt(np.array([Pl.P(3.0, 0)[0], -3.75])))
    N.write("ch05_winding", b)
    print("written figures/ch05_winding.tex; |yhat| at the probes: %s"
          % ", ".join("%.3f" % abs(yhat(nu)) for nu, _, _ in PROBES))


if __name__ == "__main__":
    main()
