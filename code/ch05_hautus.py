"""ch05_hautus.py -- the Hautus test: a mode that the readout cannot see.

The damped qubit of the chapter (ch05_spectrum: omega = 3, gamma1 = 0.2,
gphi = 0.4, no drive) starts from the tilted state X0.  Its Bloch vector does
two things at once: it spins in the equatorial plane and shrinks there (the
transverse modes -Gamma +- i omega), and it climbs along z towards the north
pole (the modes 0 and -gamma1).  The right eigenvectors v of the transverse
modes lie in the (x, y)-plane, so  C v = 0  for the readout sigma_z; the
eigenvectors of 0 and -gamma1 have no x-component, so  C v = 0  for sigma_x.
  (a) the trajectory on the Bloch ball, its shadow in the equatorial plane
      (the spinning) and on the z-axis (the climbing), and the two readouts;
  (b) for each readout, the pole map -- seen poles filled, cancelled poles as
      a pole with a zero on top -- and the data it produces.
Writes figures/ch05_hautus.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import qubit_A, X0, CX, CZ, OMEGA, GAMMA, GAMMA1, outputs

V = N.View(az=28.0, el=17.0, R=2.15, shift=(0.0, 0.0))


def hautus_rank(A, C, lam):
    return np.linalg.matrix_rank(np.vstack([lam * np.eye(len(A)) - A, C]), 1e-9)


def ball_panel(A):
    """(a) the trajectory, its two shadows, and the two readout directions."""
    t = np.linspace(0, 25, 1500)
    xyz = outputs(A, np.eye(4)[:3], X0, t)
    b = []
    # equatorial disc, washed
    s = np.linspace(0, 2 * np.pi, 160)
    disc = [V.P((np.cos(a), np.sin(a), 0.0)) for a in s]
    b.append("\\fill[ndSky, fill opacity=0.55] %s -- cycle;" % " -- ".join(N.pt(p) for p in disc))
    b += V.sphere(shade=False)
    b += V.great_circle([0, 0, 1], front="faint", back="faint, densely dotted")
    b.append("\\draw[faint] %s -- %s;" % (V.pt((0, 0, 0)), V.pt((1.0, 0, 0))))
    for v, lab, pos in [((0, 1.35, 0), "y", "right")]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s, text=ndInk] {$%s$};"
                 % (V.pt((0, 0, 0)), V.pt(v), pos, lab))
    b.append("\\draw[faint] %s -- %s;" % (V.pt((0, 0, -1.0)), V.pt((0, 0, 1.0))))
    # the two shadows: spinning in the plane (blue), climbing along z (red)
    b.append(N.polyline([V.P((p[0], p[1], 0.0)) for p in xyz[:700]], "line width=1.1pt, ndBlue"))
    zc = xyz[:, 2]
    b.append("\\draw[line width=2.4pt, ndRed, ->, >=stealth] %s -- %s;"
             % (V.pt((0, 0, zc[0])), V.pt((0, 0, zc[-1] - 0.03))))
    # the trajectory itself
    b += V.curve(xyz, "line width=0.9pt, ndInk", "line width=0.6pt, ndInk, densely dotted")
    b.append("\\node[dot] at %s {};" % V.pt(xyz[0]))
    b.append("\\node[font=\\scriptsize, anchor=north east] at %s {$\\vec x_0$};" % V.pt(xyz[0]))
    b.append("\\draw[ndInk!50, densely dotted] %s -- %s;" % (V.pt(xyz[0]), V.pt((xyz[0, 0], xyz[0, 1], 0))))
    b.append("\\draw[ndInk!50, densely dotted] %s -- %s;" % (V.pt(xyz[0]), V.pt((0, 0, xyz[0, 2]))))
    # readout directions: the x-axis and z-axis, continued outside the ball
    b.append("\\draw[vec, ndBlue] %s -- %s node[pos=1.0, below, xshift=8pt, yshift=-1pt, fill=white, fill opacity=0.8, text opacity=1, inner sep=1pt, font=\\small, text=ndBlue] {$x$: measure $\\sigma_x$};"
             % (V.pt((1.0, 0, 0)), V.pt((1.75, 0, 0))))
    b.append("\\draw[vec, ndRed] %s -- %s node[pos=1.0, right, font=\\small, text=ndRed] {$z$: measure $\\sigma_z$};"
             % (V.pt((0, 0, 1.02)), V.pt((0, 0, 1.45))))
    # notes
    shadow = np.array([V.P((q[0], q[1], 0.0)) for q in xyz[:400]])
    i = 60 + np.argmin(shadow[60:, 0])
    pin_spin = shadow[i]                                     # left end of the first loop
    pin_climb = V.P((0, 0, 0.82))
    b.append("\\node[note, anchor=east] (s) at (-2.45,-0.55) {rotation in the plane,\\\\ perpendicular to $z$:\\\\ "
             "not seen by $\\sigma_z$\\\\ ($\\mathbf C\\vec v=0$)};")
    b.append("\\draw[pointer] (s.east) to[bend left=12] %s;" % N.pt(pin_spin))
    b.append("\\node[note, anchor=east] (c) at (-2.45,1.55) {rise along $z$,\\\\ perpendicular to $x$:\\\\ "
             "not seen by $\\sigma_x$};")
    b.append("\\draw[pointer] (c.east) to[bend left=15] %s;" % N.pt(pin_climb + np.array([-0.06, 0.0])))
    b.append("\\node[font=\\small] at (-0.9,-3.6) {(a) one trajectory, two motions};")
    return b


def pole_map(o, seen, hidden, col):
    """Small lambda-plane at page origin o (centre of the imaginary axis); Re stretched."""
    sx, sy = 1.6, 0.36
    P = lambda z: o + np.array([sx * z.real, sy * z.imag])
    b = ["\\fill[ndSky, fill opacity=0.35] %s rectangle %s;" % (N.pt(P(-0.75 - 3.6j)), N.pt(P(0 + 3.6j))),
         "\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$\\mathrm{Re}$};"
         % (N.pt(P(-0.8 + 0j)), N.pt(P(0.35 + 0j))),
         "\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$\\mathrm{Im}$};"
         % (N.pt(P(-3.7j)), N.pt(P(3.9j)))]
    for z in hidden:
        p = P(z)
        b.append("\\draw[line width=0.6pt, ndInk!55, fill=white] %s circle (3.0pt);" % N.pt(p))
        b.append("\\draw[line width=0.6pt, ndInk!55] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
                 % (p[0] - 0.075, p[1] - 0.075, p[0] + 0.075, p[1] + 0.075,
                    p[0] - 0.075, p[1] + 0.075, p[0] + 0.075, p[1] - 0.075))
    for z in seen:
        b.append("\\fill[%s] %s circle (3.0pt);" % (col, N.pt(P(z))))
    return b


def readout_row(A, C, name, col, y0, seen, hidden, note, note_up):
    b = pole_map(np.array([5.3, y0]), seen, hidden, col)
    b.append("\\node[font=\\small, text=%s, anchor=east] at (3.95,%.2f) {measure %s};" % (col, y0, name))
    t = np.linspace(0, 12, 400)
    y = outputs(A, C, X0, t)[:, 0]
    Pl = N.Plot((0, 12), (-1.0, 1.0), 4.0, 2.1, origin=(6.75, y0 - 1.05))
    b += Pl.axes("$t$", "", yticks=[(-1, "$-1$"), (1, "$1$")], yzero=0.0)
    b.append(Pl.curve(t, y, "line width=1.2pt, %s" % col))
    yn, anc = (0.45, "south east") if note_up else (-0.25, "north east")
    b.append("\\node[note, anchor=%s] at %s {%s};" % (anc, N.pt(Pl.P(12.0, yn)), note))
    return b


def main():
    A = qubit_A()
    lam, Vr = np.linalg.eig(A)
    trans = np.abs(lam.imag) > 1e-9
    # Hautus: C v = 0 exactly for the hidden modes, rank drops exactly there
    assert np.allclose(CZ @ Vr[:, trans], 0) and np.all(np.abs(CZ @ Vr[:, ~trans]) > 0.5)
    assert np.allclose(CX @ Vr[:, ~trans], 0) and np.all(np.abs(CX @ Vr[:, trans]) > 0.5)
    for l, tr in zip(lam, trans):
        assert hautus_rank(A, CZ, l) == (3 if tr else 4)
        assert hautus_rank(A, CX, l) == (4 if tr else 3)
    pair = np.array([-GAMMA + 1j * OMEGA, -GAMMA - 1j * OMEGA])
    slow = np.array([0.0 + 0j, -GAMMA1 + 0j])
    b = ball_panel(A)
    b += readout_row(A, CX, "$\\sigma_x$", "ndBlue", 1.55, pair, slow,
                     "shows $-\\Gamma\\pm\\imath\\omega$:\\\\ oscillates", True)
    b += readout_row(A, CZ, "$\\sigma_z$", "ndRed", -1.55, slow, pair,
                     "shows $0$, $-\\gamma_1$:\\\\ relaxes only", False)
    b.append("\\node[note, anchor=west] at (3.3,-3.05) {filled: a pole present in the data;\\ "
             "\\tikz[baseline=-0.6ex]{\\draw[line width=0.6pt, ndInk!55] (0,0) circle (3pt) "
             "(-0.075,-0.075) -- (0.075,0.075) (-0.075,0.075) -- (0.075,-0.075);}: cancelled, $\\mathbf C\\vec v_i=0$};")
    b.append("\\node[font=\\small] at (7.7,-3.6) {(b) the poles seen by each measurement, and the data};")
    N.write("ch05_hautus", b)
    print("written figures/ch05_hautus.tex")


if __name__ == "__main__":
    main()
