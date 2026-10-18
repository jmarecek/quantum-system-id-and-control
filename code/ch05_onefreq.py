"""ch05_onefreq.py -- what one driving frequency can and cannot tell.

The weak-drive response of the qubit of the chapter has the transfer function
    G(s) = b / ((s + Gamma)^2 + omega^2),   b = omega = 3,  Gamma = 0.5
(ch05_response).  Suppose that only the form of G is known.  A sinusoidal
drive at one frequency nu_0 yields the single complex number g_0 = G(i nu_0),
gain and phase, which gives two real equations for the three unknowns
(b, Gamma, omega).  Solving,
    Gamma = b Im(1/g_0) / (2 nu_0),   omega^2 = b Re(1/g_0) - Gamma^2 + nu_0^2,
a one-parameter family of resonances agrees with the measurement, and their
poles -Gamma + i omega fill a curve in the lambda-plane.  A second frequency
nu_1 gives a second curve, and the two curves meet at the true pole only.
  (a) the gain |G(i nu)| of members of the family, all through the measured
      point at nu_0; only one of them passes through the point at nu_1;
  (b) the two curves of poles in the lambda-plane.
Writes figures/ch05_onefreq.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import OMEGA, GAMMA

NU0, NU1 = 2.6, 3.4
B_TRUE = OMEGA


def G(s, b, gam, om):
    return b / ((s + gam) ** 2 + om ** 2)


def family(nu, bs):
    """(b, Gamma, omega) of the resonances that agree with the measurement at nu."""
    q = 1 / G(1j * nu, B_TRUE, GAMMA, OMEGA)
    out = []
    for b in bs:
        gam = b * q.imag / (2 * nu)
        w2 = b * q.real - gam ** 2 + nu ** 2
        if gam > 0 and w2 > 0:
            out.append((b, gam, np.sqrt(w2)))
    return out


def main():
    g0, g1 = G(1j * NU0, B_TRUE, GAMMA, OMEGA), G(1j * NU1, B_TRUE, GAMMA, OMEGA)
    cands = family(NU0, [1.0, 2.0, 3.0, 4.5, 6.0])
    for b, gam, om in cands:                                   # every candidate fits nu_0 exactly ...
        assert abs(G(1j * NU0, b, gam, om) - g0) < 1e-12
        fits1 = abs(G(1j * NU1, b, gam, om) - g1) < 1e-9       # ... but only the true one fits nu_1
        assert fits1 == (abs(b - B_TRUE) < 1e-12)
    loc0 = family(NU0, np.linspace(0.3, 8.5, 300))
    loc1 = family(NU1, np.linspace(0.3, 8.5, 300))
    # the two curves of poles cross only at the true pole
    p0 = np.array([-g + 1j * w for _, g, w in loc0])
    p1 = np.array([-g + 1j * w for _, g, w in loc1])
    d = np.abs(p0[:, None] - p1[None, :])
    i, j = np.unravel_index(np.argmin(d), d.shape)
    assert abs(p0[i] - (-GAMMA + 1j * OMEGA)) < 0.03 and abs(p1[j] - (-GAMMA + 1j * OMEGA)) < 0.03

    b = []
    # ---- (a) gain curves
    nu = np.linspace(0, 6, 400)
    Pl = N.Plot((0, 6), (0, 2.2), 7.4, 4.6, origin=(0.0, 0.0))
    b += Pl.axes("$\\nu$", "$\\abs{\\TF(\\imath\\nu)}$", xticks=[(NU0, "$\\nu_0$"), (NU1, "$\\nu_1$"), (6, "$6$")],
                 yticks=[(1, "$1$"), (2, "$2$")])
    for bb, gam, om in cands:
        if abs(bb - B_TRUE) > 1e-12:
            b.append(Pl.curve(nu, np.abs(G(1j * nu, bb, gam, om)), "line width=0.7pt, ndGold"))
    b.append(Pl.curve(nu, np.abs(G(1j * nu, B_TRUE, GAMMA, OMEGA)), "line width=1.4pt, ndBlue"))
    for v, g, col in [(NU0, g0, "ndRed"), (NU1, g1, "ndGreen")]:
        b.append("\\draw[dashedink] %s -- %s;" % (Pl.pt(v, 0), Pl.pt(v, abs(g))))
        b.append("\\node[dot, fill=%s, minimum size=5pt] at %s {};" % (col, Pl.pt(v, abs(g))))
    b.append("\\node[note, anchor=north west] (f) at %s {gold: other resonances with the\\\\ same gain and phase at $\\nu_0$};" % Pl.pt(0.15, 2.25))
    b.append("\\draw[pointer] (f.south) to[bend right=20] %s;" % Pl.pt(1.0, np.abs(G(1j * 1.0, *cands[-1])) + 0.02))
    b.append("\\node[note, anchor=west, text=ndBlue] (t) at %s {the true response};" % Pl.pt(4.3, 1.3))
    b.append("\\draw[pointer] (t.west) to[bend right=15] %s;" % Pl.pt(3.55, abs(G(3.55j, B_TRUE, GAMMA, OMEGA))))
    b.append("\\node[note, anchor=east, text=ndRed] (m0) at %s {measured\\\\ at $\\nu_0$};" % Pl.pt(1.95, 1.4))
    b.append("\\draw[pointer] (m0.east) to[bend left=15] %s;" % Pl.pt(NU0 - 0.05, abs(g0) + 0.05))
    b.append("\\node[note, anchor=west, text=ndGreen] (m1) at %s {measured at $\\nu_1$:\\\\ only one curve fits};" % Pl.pt(4.3, 0.78))
    b.append("\\draw[pointer] (m1.west) to[bend left=15] %s;" % Pl.pt(NU1 + 0.06, abs(g1) - 0.04))
    b.append("\\node[font=\\small] at %s {(a) the gain of resonances that fit the data};" % N.pt(Pl.P(3.0, 0) + np.array([0.0, -0.95])))
    # ---- (b) the lambda-plane
    sx, sy, ox, oy = 2.6, 1.9, 13.2, -2.85
    P = lambda z: (ox + sx * z.real, oy + sy * z.imag)
    b.append("\\fill[ndSky, fill opacity=0.45] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (*P(-1.45 + 1.55j), *P(0 + 4.0j)))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,%.2f) -- (%.2f,%.2f) node[right, text=ndInk, font=\\small] {$\\mathrm{Re}\\,\\lambda$};" % (*P(-1.5 + 1.55j), *P(0.4 + 1.55j)))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,%.2f) -- (%.2f,%.2f) node[above, text=ndInk, font=\\small] {$\\mathrm{Im}\\,\\lambda$};" % (*P(0 + 1.55j), *P(0 + 4.1j)))
    for v in (-1.0, -0.5):
        b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f) node[below, font=\\scriptsize, text=ndInk] {$%g$};" % (*P(v + 1.6j), *P(v + 1.5j), v))
    for v in (2, 3):
        b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f) node[right, font=\\scriptsize, text=ndInk] {$%d$};" % (*P(-0.05 + v * 1j), *P(0.05 + v * 1j), v))
    for pts, col in [(p0, "ndRed"), (p1, "ndGreen")]:
        keep = (pts.real > -1.45) & (pts.imag > 1.6) & (pts.imag < 4.0)
        b.append(N.polyline([P(z) for z in pts[keep]], "line width=1.2pt, %s" % col))
    for bb, gam, om in cands:
        if abs(bb - B_TRUE) > 1e-12:
            b.append("\\fill[ndGold] (%.2f,%.2f) circle (1.8pt);" % P(-gam + 1j * om))
    b.append("\\node[dot, fill=ndInk, minimum size=5pt] (tp) at (%.2f,%.2f) {};" % P(-GAMMA + 1j * OMEGA))
    b.append("\\node[font=\\scriptsize, anchor=south] (tl) at (%.2f,%.2f) {$-\\Gamma+\\imath\\omega$};" % P(-0.22 + 3.62j))
    b.append("\\draw[pointer] (tl.south) to[bend left=15] (tp);")
    b.append("\\node[note, anchor=west, text=ndRed] at (%.2f,%.2f) {poles that fit\\\\ the data at $\\nu_0$};" % P(-1.42 + 3.72j))
    b.append("\\node[note, anchor=west, text=ndGreen] at (%.2f,%.2f) {poles that fit\\\\ the data at $\\nu_1$};" % P(-1.05 + 2.05j))
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {gold dots: the poles\\\\ of the gold curves in (a)};" % P(-0.72 + 1.35j))
    b.append("\\node[font=\\small] at (%.2f,%.2f) {(b) the poles they imply};" % P(-0.6 + 0.85j))
    N.write("ch05_onefreq", b)
    print("written figures/ch05_onefreq.tex; curves cross at", np.round(p0[i], 3))


if __name__ == "__main__":
    main()
