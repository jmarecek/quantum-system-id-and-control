"""ch02_rotation.py -- a one-parameter group and its generator.

J = [[0, -1], [1, 0]] generates the rotations of the plane: e^{tJ} is the
rotation by the angle t.  The curve t -> e^{tJ} x runs around a circle with
velocity J e^{tJ} x, always perpendicular to the position.  Equal steps in t
give equal steps along the circle, because e^{sJ} e^{tJ} = e^{(s+t)J}.
Writes figures/ch02_rotation.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N

J = np.array([[0.0, -1.0], [1.0, 0.0]])
R = 2.6                                    # page radius of the circle
X0 = np.array([1.0, 0.0])
TMAX = 2.1


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def main():
    for t in (0.3, 1.0, TMAX):
        assert np.allclose(expm(t * J), rot(t))
    b = []
    b.append("\\fill[ndSky, fill opacity=0.35] (0,0) circle (%.3f);" % R)
    b.append("\\draw[faint] (0,0) circle (%.3f);" % R)
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[above] {$x_1$};" % (-R - 0.4, R + 0.5))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[left] {$x_2$};" % (-R - 0.4, R + 0.6))
    t = np.linspace(0, TMAX, 120)
    arc = [R * (expm(s * J) @ X0) for s in t]
    b.append(N.polyline(arc, "line width=1.8pt, ndBlue"))
    for k in range(1, 6):                    # equal steps of t = 0.4
        p = R * (expm(0.4 * k * J) @ X0)
        b.append("\\fill[ndBlue] %s circle (1.6pt);" % N.pt(p))
    # the position and the velocity at the end of the arc
    x = expm(TMAX * J) @ X0
    v = J @ x
    b.append("\\draw[vec, ndInk] (0,0) -- %s node[midway, below left] {$\\ee^{tJ}x$};" % N.pt(R * x))
    b.append("\\draw[vec, ndRed] %s -- %s node[left, text=ndRed] {$J\\,\\ee^{tJ}x$};"
             % (N.pt(R * x), N.pt(R * x + 1.3 * v)))
    # right angle mark between position and velocity
    c = R * x
    u1, u2 = -0.25 * x, 0.25 * v
    b.append("\\draw[ink, line width=0.5pt] %s -- %s -- %s;" % (N.pt(c + u1), N.pt(c + u1 + u2), N.pt(c + u2)))
    b.append("\\draw[vec, ndInk] (0,0) -- %s node[below left] {$x$};" % N.pt(R * X0))
    b.append("\\draw[vec, ndRed] %s -- %s node[right, text=ndRed] {$Jx$};" % (N.pt(R * X0), N.pt(R * X0 + 1.3 * J @ X0)))
    # angle t
    ta = np.linspace(0, TMAX, 40)
    b.append(N.polyline([0.6 * np.array([np.cos(s), np.sin(s)]) for s in ta], "ink, line width=0.5pt"))
    b.append("\\node at (%.3f,%.3f) {$t$};" % (0.85 * np.cos(TMAX / 2), 0.85 * np.sin(TMAX / 2)))
    # annotations
    b.append("\\node[note, anchor=west] (a) at (%.2f,2.2) {$J=\\begin{pmatrix}0&-1\\\\1&0\\end{pmatrix}$,"
             " \\ $\\ee^{tJ}=\\begin{pmatrix}\\cos t&-\\sin t\\\\ \\sin t&\\cos t\\end{pmatrix}$:\\\\"
             "the exponential of $J$ is a rotation};" % (R + 1.3))
    b.append("\\node[note, anchor=west] (b) at (%.2f,0.2) {the velocity $J\\,\\ee^{tJ}x$ is always\\\\"
             "perpendicular to the position:\\\\ the curve stays on the circle};" % (R + 1.3))
    b.append("\\draw[pointer] (b.west) to[bend right=15] %s;" % N.pt(R * X0 + 0.9 * J @ X0 + np.array([0.08, 0])))
    b.append("\\node[note, anchor=west] (c) at (%.2f,-1.8) {dots: equal steps of $t$,\\\\"
             "since $\\ee^{sJ}\\ee^{tJ}=\\ee^{(s+t)J}$};" % (R + 1.3))
    b.append("\\draw[pointer] (c.west) to[bend right=10] %s;" % N.pt(R * (expm(0.4 * J) @ X0)))
    N.write("ch02_rotation", b)
    print("written figures/ch02_rotation.tex")


if __name__ == "__main__":
    main()
