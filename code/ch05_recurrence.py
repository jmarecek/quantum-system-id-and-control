"""ch05_recurrence.py -- two numbers predict the future of the data.

The samples y_k = <sigma_x>(k tau) of the running example (ch05_spectrum) are
    y_k = x_0 r^k cos(omega tau k),   r = e^{-Gamma tau},
and every three consecutive samples obey the same linear recurrence
    y_{k+2} = a_1 y_{k+1} + a_2 y_k,    a_1 = 2 r cos(omega tau),  a_2 = -r^2.
The roots of the characteristic polynomial z^2 - a_1 z - a_2 are
z = r e^{+- i omega tau} = e^{lambda tau} with lambda = -Gamma +- i omega.
  (a) the samples as stems, with the stencil drawn at three places;
  (b) the two roots in the complex z-plane, inside the unit circle.
Writes figures/ch05_recurrence.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import qubit_A, outputs, X0, CX, OMEGA, GAMMA, TAU

KMAX = 15


def main():
    A = qubit_A()
    k = np.arange(KMAX + 1)
    y = outputs(A, CX, X0, TAU * k)[:, 0]
    r = np.exp(-GAMMA * TAU)
    a1, a2 = 2 * r * np.cos(OMEGA * TAU), -r ** 2
    assert np.allclose(y[2:], a1 * y[1:-1] + a2 * y[:-2])
    roots = np.roots([1.0, -a1, -a2])
    lam = np.array([-GAMMA + 1j * OMEGA, -GAMMA - 1j * OMEGA])
    assert np.allclose(np.sort_complex(roots), np.sort_complex(np.exp(lam * TAU)))

    b = []
    # ---- (a) stems and stencils
    t = np.linspace(0, KMAX, 600)
    yc = outputs(A, CX, X0, TAU * t)[:, 0]
    Pl = N.Plot((-0.5, KMAX + 0.5), (-0.62, 1.0), 10.0, 4.4, origin=(0.0, 0.0))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk] {$k$};" % (Pl.pt(-0.5, 0), Pl.pt(KMAX + 0.9, 0)))
    b.append(Pl.curve(t, yc, "line width=0.5pt, ndInk!30"))
    for i in range(KMAX + 1):
        b.append("\\draw[line width=1.0pt, ndInk!70] %s -- %s;" % (Pl.pt(i, 0), Pl.pt(i, y[i])))
        b.append("\\fill[ndInk] %s circle (1.7pt);" % Pl.pt(i, y[i]))
    for i in (0, 5, 10, 15):
        p = Pl.P(i, 0)
        lab_pos = "above" if y[i] < 0 else "below"
        b.append("\\node[font=\\scriptsize, %s] at (%.3f,%.3f) {$%d$};" % (lab_pos, p[0], p[1], i))
    stencils = [(0, "ndBlue"), (3, "ndGreen"), (7, "ndRed")]
    for k0, col in stencils:
        src0, src1, tgt = Pl.P(k0, y[k0]), Pl.P(k0 + 1, y[k0 + 1]), Pl.P(k0 + 2, y[k0 + 2])
        d, m = tgt - src0, src1 - (src0 + tgt) / 2
        side = "bend right=50" if (-d[1] * m[0] + d[0] * m[1]) > 0 else "bend left=50"   # away from y_{k+1}
        sty = "->, >=stealth, line width=0.9pt, %s, shorten <=2.5pt, shorten >=3.8pt" % col
        b.append("\\draw[%s] (%.3f,%.3f) to[%s] (%.3f,%.3f);" % (sty, src0[0], src0[1], side, tgt[0], tgt[1]))
        b.append("\\draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);" % (sty, src1[0], src1[1], tgt[0], tgt[1]))
        b.append("\\draw[line width=0.9pt, %s] (%.3f,%.3f) circle (3.6pt);" % (col, tgt[0], tgt[1]))
    # the stencil as a key: y_k and y_{k+1} feed y_{k+2} with the weights a_2 and a_1
    kx, ky, dx = 3.9, 3.85, 1.25
    pts = [np.array([kx + i * dx, ky]) for i in range(3)]
    sty = "->, >=stealth, line width=0.9pt, ndInk, shorten <=2.5pt, shorten >=3.8pt"
    for p_, lab in zip(pts, ("y_k", "y_{k+1}", "y_{k+2}")):
        b.append("\\fill[ndInk] %s circle (1.7pt);" % N.pt(p_))
        b.append("\\node[font=\\scriptsize, below] at %s {$%s$};" % (N.pt(p_), lab))
    b.append("\\draw[ink] %s circle (3.6pt);" % N.pt(pts[2]))
    b.append("\\draw[%s] %s -- %s node[midway, above=0pt, font=\\scriptsize] {$\\times%.2f$};" % (sty, N.pt(pts[1]), N.pt(pts[2]), a1))
    b.append("\\draw[%s] %s to[bend left=60] node[midway, above, font=\\scriptsize] {$\\times(%.2f)$} %s;" % (sty, N.pt(pts[0]), a2, N.pt(pts[2])))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {the same two coefficients for every $k$:\\\\ $y_{k+2}=%.2f\\,y_{k+1}%.2f\\,y_k$};"
             % (kx + 2 * dx + 0.35, ky + 0.12, a1, a2))
    b.append("\\node[note, anchor=north] at %s {stems: the samples $y_k=\\langle\\sigma_x\\rangle(k\\tau)$, $\\tau=%.2f$;\\\\ grey: the signal between the samples};"
             % (Pl.pt(11.4, -0.22), TAU))
    b.append("\\node[font=\\small] at (%.2f,-0.5) {(a) each sample is determined by the two before it};" % Pl.P(7.5, 0)[0])
    # ---- (b) the roots in the z-plane
    c, R = np.array([13.5, 2.75]), 1.4
    Z = lambda z: c + R * np.array([z.real, z.imag])
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$\\mathrm{Re}\\,z$};" % (N.pt(Z(-1.2)), N.pt(Z(1.3))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$\\mathrm{Im}\\,z$};" % (N.pt(Z(-1.2j)), N.pt(Z(1.3j))))
    b.append("\\draw[ink] %s circle (%.3f);" % (N.pt(c), R))
    b.append("\\fill[ndSky, fill opacity=0.5] %s circle (%.3f);" % (N.pt(c), R * r))
    b.append("\\draw[dashedink] %s circle (%.3f);" % (N.pt(c), R * r))
    th = OMEGA * TAU
    arc = [Z(0.42 * np.exp(1j * s)) for s in np.linspace(0, th, 30)]
    b.append(N.polyline(arc, "->, >=stealth, line width=0.6pt, ndInk"))
    b.append("\\node[font=\\scriptsize] at %s {$\\omega\\tau$};" % N.pt(Z(0.58 * np.exp(1j * th / 2))))
    b.append("\\draw[line width=0.6pt, ndInk!60] %s -- %s;" % (N.pt(c), N.pt(Z(roots[np.argmax(roots.imag)]))))
    for z in roots:
        b.append("\\fill[ndBlue] %s circle (2.6pt);" % N.pt(Z(z)))
    zt = roots[np.argmax(roots.imag)]
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=south west] at %s {$\\ee^{\\lambda\\tau}$};" % N.pt(Z(zt)))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=north west] at %s {$\\ee^{\\bar\\lambda\\tau}$};" % N.pt(Z(np.conj(zt))))
    b.append("\\node[note, anchor=north] at %s {roots of $z^2-%.2f\\,z+%.2f$:\\\\ modulus $\\ee^{-\\Gamma\\tau}$ (dashed), argument $\\omega\\tau$;\\\\ the poles under the map $z=\\ee^{\\lambda\\tau}$};"
             % (N.pt(c + np.array([0, -R - 0.2])), a1, -a2))
    b.append("\\node[font=\\small] at (%.2f,-0.5) {(b) the coefficients determine the poles};" % c[0])
    N.write("ch05_recurrence", b)
    print("written figures/ch05_recurrence.tex; a1 = %.4f, a2 = %.4f" % (a1, a2))


if __name__ == "__main__":
    main()
