"""ch02_mixtures.py -- mixed states inside the Bloch ball.

The (x, z) cross-section of the Bloch ball.  A mixed state rho with Bloch
vector r is a convex combination of two pure states at the ends of any chord
through r:  r = p a + (1 - p) b  with |a| = |b| = 1.  Two different chords give
two different ensembles with the same density operator.  Writes
figures/ch02_mixtures.tex.
"""
import numpy as np
import needham as N

R = 2.5
r = np.array([0.30, 0.28])                 # (x, z) of the mixed state


def chord(r, angle):
    """End points a, b of the chord through r in direction angle, and weight p of a."""
    d = np.array([np.cos(angle), np.sin(angle)])
    # |r + s d| = 1  ->  s^2 + 2 (r.d) s + |r|^2 - 1 = 0
    rd, rr = r @ d, r @ r
    disc = np.sqrt(rd ** 2 - rr + 1)
    sa, sb = -rd + disc, -rd - disc
    a, b = r + sa * d, r + sb * d
    p = -sb / (sa - sb)                    # r = p a + (1 - p) b
    assert np.allclose(p * a + (1 - p) * b, r)
    return a, b, p


def page(v):
    return (R * v[0], R * v[1])


def main():
    b = []
    b.append("\\fill[ndSky, fill opacity=0.5] (0,0) circle (%.3f);" % R)
    b.append("\\draw[line width=1.6pt, ndBlue] (0,0) circle (%.3f);" % R)
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[below] {$x$};" % (-R - 0.3, R + 0.5))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[left] {$z$};" % (-R - 0.3, R + 0.5))
    # landmarks
    for v, lab, pos in [((0, 1), "\\ket0", "above right"), ((0, -1), "\\ket1", "below right"),
                        ((1, 0), "\\ket+", "above right"), ((-1, 0), "\\ket-", "left")]:
        b.append("\\node[dot, label={[font=\\footnotesize]%s:$%s$}] at %s {};" % (pos, lab, N.pt(page(v))))
    b.append("\\node[dot, fill=ndInk!60, label={[font=\\footnotesize]below left:$\\tfrac12\\Id$}] at (0,0) {};")
    # two chords through r
    out = []
    for ang, col in [(np.radians(100.0), "ndRed"), (np.radians(20.0), "ndGreen")]:
        a, c, p = chord(r, ang)
        b.append("\\draw[line width=1.1pt, %s] %s -- %s;" % (col, N.pt(page(a)), N.pt(page(c))))
        b.append("\\node[dot, fill=%s] at %s {};" % (col, N.pt(page(a))))
        b.append("\\node[dot, fill=%s] at %s {};" % (col, N.pt(page(c))))
        out.append((a, c, p, col))
    b.append("\\node[dot, minimum size=4.6pt, label={[font=\\small]above left:$\\rho$}] at %s {};" % N.pt(page(r)))
    (a1, c1, p1, _), (a2, c2, p2, _) = out
    b.append("\\node[note, anchor=west] (n1) at (%.2f,%.2f) {pure states:\\\\ the sphere $\\norm{\\mathbf r}=1$};" % (R + 0.6, 1.9))
    b.append("\\draw[pointer] (n1.west) to[bend right=15] %s;" % N.pt(page((np.cos(0.9), np.sin(0.9)))))
    b.append("\\node[note, anchor=west] (n2) at (%.2f,%.2f) {mixed states: inside;\\\\"
             "$\\Tr\\rho^2=\\tfrac12(1+\\norm{\\mathbf r}^2)$};" % (R + 0.6, -0.4))
    b.append("\\node[note, anchor=west] (n3) at (%.2f,%.2f) {\\textcolor{ndRed}{$\\rho=%.2f\\,\\rho_a+%.2f\\,\\rho_b$}\\\\"
             "\\textcolor{ndGreen!80!black}{$\\rho=%.2f\\,\\rho_{a'}+%.2f\\,\\rho_{b'}$}\\\\"
             "same $\\rho$, two different ensembles:\\\\ any chord through $\\rho$ will do};"
             % (R + 0.6, -2.3, p1, 1 - p1, p2, 1 - p2))
    b.append("\\draw[pointer] (n3.west) to[bend left=20] %s;" % N.pt(page(r + np.array([0.04, -0.03]))))
    labs = [(a1, "\\rho_a", "above"), (c1, "\\rho_b", "below"), (a2, "\\rho_{a'}", "right"), (c2, "\\rho_{b'}", "left")]
    for v, lab, pos in labs:
        b.append("\\node[%s, font=\\footnotesize] at %s {$%s$};" % (pos, N.pt(page(v)), lab))
    N.write("ch02_mixtures", b)
    print("written figures/ch02_mixtures.tex; p =", round(p1, 3), round(p2, 3))


if __name__ == "__main__":
    main()
