"""ch02_magnus.py -- truncated Dyson series leave the group; Magnus does not.

The group U(1) = unit circle in the complex plane, with Lie algebra the
imaginary axis (drawn as the tangent line at 1).  For H = 1 the exact
propagator is e^{-i theta}, theta = t.  The Dyson truncations
  D_1 = 1 - i theta,  D_2 = D_1 - theta^2/2,  D_3 = D_2 + i theta^3/6
are polynomials and lie off the circle (|D_K| != 1), whereas every truncation
of the Magnus series is an exponential of an element of iR, hence on the circle.
Writes figures/ch02_magnus.tex.
"""
import numpy as np
from math import factorial
import needham as N

R = 2.4                                    # page radius of the unit circle
THETA = 1.25


def dyson(K, th):
    return sum((-1j * th) ** k / factorial(k) for k in range(K + 1))


def P(z):
    return (R * z.real, R * z.imag)


def main():
    b = []
    b.append("\\fill[ndSky, fill opacity=0.35] (0,0) circle (%.3f);" % R)
    b.append("\\draw[ink] (0,0) circle (%.3f);" % R)
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[below] {$\\mathrm{Re}$};" % (-R - 0.3, R + 1.4))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[left] {$\\mathrm{Im}$};" % (-R - 1.5, R + 0.3))
    # the Lie algebra, drawn as the tangent line at 1
    b.append("\\draw[line width=0.6pt, ndGold] (%.3f,%.3f) -- (%.3f,%.3f) node[above] {$1+\\imath\\R$};"
             % (R, -R - 1.3, R, 1.0))
    # exact path and the Dyson truncations as curves in theta
    th = np.linspace(0, THETA, 80)
    b.append(N.polyline([P(np.exp(-1j * t)) for t in th], "line width=2.2pt, ndBlue"))
    styles = {1: "line width=0.9pt, ndRed", 2: "line width=0.9pt, ndRed!70, dash pattern=on 4pt off 1.5pt",
              3: "line width=0.9pt, ndRed!45, dash pattern=on 1.5pt off 1.5pt"}
    for K in (1, 2, 3):
        b.append(N.polyline([P(dyson(K, t)) for t in th], styles[K]))
        z = dyson(K, THETA)
        b.append("\\node[dot, fill=ndRed] at %s {};" % N.pt(P(z)))
        pos = {1: "right", 2: "below", 3: "above left"}[K]
        b.append("\\node[%s, text=ndRed, font=\\footnotesize] at %s {$D_%d$};" % (pos, N.pt(P(z)), K))
    zx = np.exp(-1j * THETA)
    b.append("\\node[dot, fill=ndBlue] at %s {};" % N.pt(P(zx)))
    b.append("\\node[below left, text=ndBlue, xshift=-3pt] at %s {$\\ee^{-\\imath\\theta}$};" % N.pt(P(zx)))
    b.append("\\node[dot, label={[font=\\footnotesize]above right:$1$}] at (%.3f,0) {};" % R)
    # annotations
    mods = [abs(dyson(K, THETA)) for K in (1, 2, 3)]
    b.append("\\node[note, anchor=west] (d) at (%.2f,%.2f) {Dyson, truncated:\\\\ a polynomial,\\\\"
             "\\emph{off} the circle:\\\\ $|D_1|=%.2f$, $|D_2|=%.2f$,\\\\ $|D_3|=%.2f$};"
             % (R + 0.9, -1.4, mods[0], mods[1], mods[2]))
    b.append("\\draw[pointer] (d.north west) to[bend right=15] %s;" % N.pt(P(dyson(1, 0.8))))
    b.append("\\node[note, anchor=east] (m) at (%.2f,%.2f) {Magnus, truncated:\\\\"
             "$\\ee^{\\Omega_1+\\dots+\\Omega_K}$ with $\\Omega_k\\in\\imath\\R$\\\\ stays \\emph{on} the circle};"
             % (-1.2, -R - 0.75))
    b.append("\\draw[pointer] (m.east) to[bend right=20] %s;" % N.pt(P(np.exp(-1j * 0.95))))
    b.append("\\node[note, anchor=south east] at (%.2f,%.2f) {the group $\\Un[1]$\\\\ (unit circle)};" % (-0.72 * R, 0.72 * R))
    b.append("\\node[note, anchor=west] (t) at (%.2f,%.2f) {$D_1=1-\\imath\\theta$ runs along\\\\ the tangent line: the\\\\ Lie algebra direction};"
             % (R + 0.9, 1.2))
    b.append("\\draw[pointer] (t.west) to[bend right=10] (%.3f,%.3f);" % (R + 0.05, 0.6))
    N.write("ch02_magnus", b)
    print("written figures/ch02_magnus.tex; |D_K| =", ["%.3f" % m for m in mods])


if __name__ == "__main__":
    main()
