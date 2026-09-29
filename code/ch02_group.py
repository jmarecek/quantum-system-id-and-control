"""ch02_group.py -- a Lie group, its Lie algebra, and the exponential map.

The group G is drawn as a sphere (a picture, not SU(2) itself); the Lie
algebra g is the tangent plane at the identity 1.  A vector X in g is the
initial velocity of the one-parameter subgroup t -> e^{tX}, which we draw as
the great circle through 1 in the direction of X, with marks at t = 1, 2, 3.
Writes figures/ch02_group.tex.
"""
import numpy as np
import needham as N

V = N.View(az=30.0, el=24.0, R=2.4)
ONE = np.array([0.0, 0.0, 1.0])           # the identity, drawn at the "north pole"


def geodesic(direction, t):
    """Great circle through ONE with unit initial velocity `direction`."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return np.outer(np.cos(t), ONE) + np.outer(np.sin(t), d)


def main():
    b = V.sphere()
    b += V.great_circle([0, 0, 1], "faint", "faint, densely dotted")        # an "equator"
    # tangent plane at the identity: the Lie algebra
    s = 0.8
    corners = [ONE + s * np.array(c) for c in [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]]
    b.append("\\filldraw[fill=ndPaper, fill opacity=0.85, draw=ndInk!70, line width=0.6pt] %s -- cycle;"
             % " -- ".join(V.pt(c) for c in corners))
    b.append("\\node[anchor=north east] at %s {$\\mathfrak g=T_{\\Id}G$};" % V.pt(corners[0] + np.array([0.0, 0.0, 0])))
    # two Lie algebra elements and their one-parameter subgroups
    SCALE = 0.42                        # the page length of X in the tangent plane
    for d, col, name, lab_pos in [((1.0, -0.35, 0.0), "ndBlue", "X", "below left"),
                                  ((-0.25, 1.0, 0.0), "ndRed", "Y", "above")]:
        d = np.array(d) / np.linalg.norm(d)
        t = np.linspace(0, 1.25, 80)
        b += V.curve(geodesic(d, t), "line width=1.1pt, %s!75" % col, None)
        b.append("\\draw[vec, %s] %s -- %s node[%s] {$%s$};"
                 % (col, V.pt(ONE), V.pt(ONE + SCALE * d), lab_pos, name))
        for k in (1, 2, 3):
            tk = k * SCALE
            p = geodesic(d, np.array([tk]))[0]
            if V.visible(p):
                b.append("\\fill[%s] %s circle (1.5pt);" % (col, V.pt(p)))
        pend = geodesic(d, np.array([3 * SCALE]))[0]
        if name == "X":
            b.append("\\node[%s, below right] at %s {$\\ee^{3X}$};" % (col, V.pt(pend)))
        else:
            b.append("\\node[%s, above right] at %s {$\\ee^{3Y}$};" % (col, V.pt(pend)))
    b.append("\\node[dot, label={[label distance=1pt]above:$\\Id$}] at %s {};" % V.pt(ONE))
    b.append("\\node at %s {$G$};" % V.pt(np.array([0.55, -0.55, -0.65])))
    # annotations
    b.append("\\node[note, anchor=west] (n1) at (3.7,2.5) {the Lie algebra: all\\\\ velocities at $\\Id$};")
    b.append("\\draw[pointer] (n1.west) to[bend right=15] %s;" % V.pt(corners[2] + np.array([0.0, -0.2, 0])))
    b.append("\\node[note, anchor=west] (n2) at (3.7,0.5) {$t\\mapsto\\ee^{tX}$: set off from $\\Id$\\\\"
             "with velocity $X$ and keep\\\\ going ``straight''};")
    p_mid = geodesic(np.array((1.0, -0.35, 0.0)) / np.linalg.norm((1.0, -0.35, 0.0)), np.array([0.75]))[0]
    b.append("\\draw[pointer] (n2.west) to[bend left=15] %s;" % V.pt(p_mid))
    b.append("\\node[note, anchor=west] (n3) at (3.7,-1.5) {dots: equal steps of $t$;\\\\"
             "$\\ee^{sX}\\ee^{tX}=\\ee^{(s+t)X}$};")
    N.write("ch02_group", b)
    print("written figures/ch02_group.tex")


if __name__ == "__main__":
    main()
