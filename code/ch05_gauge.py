"""ch05_gauge.py -- one transfer function, many realisations.

For any invertible T, the realisation (T A T^-1, C T^-1, T x0) produces the
same outputs C e^{At} x0 as (A, C, x0): the data see only the transfer
function.  Two choices of T acting on the Bloch block of the running example
(readout sigma_n), the affine coordinate 1 left alone:
  * T_1, a rotation of the Bloch ball (by 50 degrees about the y-axis): a
    change of frame, i.e. a unitary change of basis; T_1 A T_1^-1 is again a
    Lindbladian and e^{T_1 A T_1^-1 t} maps the Bloch ball into itself;
  * T_2, a shear and dilation of the Bloch ball: T_2 A T_2^-1 reproduces the
    data too, but e^{T_2 A T_2^-1 t} maps part of the Bloch ball outside it,
    so it is not a quantum channel and T_2 A T_2^-1 is not a Lindbladian.
The figure shows the three realisations as coloured matrices, the image of
the Bloch ball after t* = pi / (2 omega), and the common output.
Writes figures/ch05_gauge.tex.
"""
import numpy as np
from scipy.linalg import expm
from scipy.spatial import ConvexHull
import needham as N
from ch05_spectrum import OMEGA, X0, CN, qubit_A, outputs

TSTAR = np.pi / (2 * OMEGA)
CELL = 0.55


def blockdiag(M):
    T = np.eye(4)
    T[:3, :3] = M
    return T


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def ball_image(A, t, n=40):
    """Points of the image of the unit sphere under x -> e^{At} x (affine coordinate 1)."""
    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2))
    pts = np.stack([np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w), np.ones_like(u)], -1).reshape(-1, 4)
    E = expm(A * t)
    return (pts @ E.T)[:, :3]


def cells(M, x, y, vmax, b, fmt="%.1f"):
    """Matrix M as a grid of cells with top-left corner (x, y): blue >0, red <0."""
    M = np.atleast_2d(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            x0, y0 = x + j * CELL, y - i * CELL
            if abs(v) > 5e-3:
                col = "ndBlue" if v > 0 else "ndRed"
                b.append("\\fill[%s!%d] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, int(12 + 55 * np.sqrt(min(abs(v) / vmax, 1))), x0, y0, x0 + CELL, y0 - CELL))
                b.append("\\node[font=\\fontsize{5.5}{5.5}\\selectfont, text=ndInk] at (%.3f,%.3f) {$%s$};" % (x0 + CELL / 2, y0 - CELL / 2, (fmt % v).replace("-", "{-}")))
    b.append("\\draw[ink, line width=0.6pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x, y, x + M.shape[1] * CELL, y - M.shape[0] * CELL))


def main():
    A = qubit_A()
    T1 = blockdiag(rot_y(np.radians(50)))
    T2 = blockdiag(np.array([[1.25, 0.35, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]]))
    reals = [(A, CN, X0)]
    for T in (T1, T2):
        Ti = np.linalg.inv(T)
        reals.append((T @ A @ Ti, CN @ Ti, T @ X0))
    t = np.linspace(0, 12, 481)
    ys = [outputs(Ak, Ck, xk, t)[:, 0] for Ak, Ck, xk in reals]
    for y in ys[1:]:
        assert np.allclose(y, ys[0], atol=1e-12)               # identical data
    radii = [np.linalg.norm(ball_image(Ak, TSTAR), axis=1).max() for Ak, _, _ in reals]
    assert radii[0] <= 1 + 1e-9 and radii[1] <= 1 + 1e-9        # channels: the ball maps into itself
    assert radii[2] > 1.2                                         # T_2: the image leaves the ball
    for Ak, _, _ in reals[:2]:                                    # ... at every time, not just t*
        assert max(np.linalg.norm(ball_image(Ak, s, 24), axis=1).max() for s in np.linspace(0, 6, 25)) <= 1 + 1e-9
    assert np.linalg.norm(reals[2][2][:3]) > 1                    # even T_2 x0 is not a Bloch vector

    vmax = max(np.abs(Ak).max() for Ak, _, _ in reals)
    b = []
    cols = [0.0, 3.75, 7.5]
    titles = ["$(\\mathbf A,\\ \\mathbf C,\\ \\vec x_0)$",
              "$\\mathbf T_1$: a rotation",
              "$\\mathbf T_2$: a shear and dilation"]
    ytop = 3.4
    V = None
    for k, ((Ak, Ck, xk), xc) in enumerate(zip(reals, cols)):
        b.append("\\node[font=\\small, anchor=south west] at (%.3f,%.3f) {%s};" % (xc - 0.05, ytop + 0.12, titles[k]))
        cells(Ak, xc, ytop, vmax, b)
        cells(xk[:, None], xc + 4 * CELL + 0.22, ytop, vmax, b)
        cells(Ck, xc, ytop - 4 * CELL - 0.22, vmax, b)
        b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {$\\mathbf C$};" % (xc - 0.04, ytop - 4.5 * CELL - 0.22))
        if k == 0:
            b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {$\\mathbf A$};" % (xc - 0.04, ytop - 2 * CELL))
            b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$\\vec x_0$};" % (xc + 4.5 * CELL + 0.22, ytop - 4 * CELL - 0.05))
        # the Bloch ball and its image at t*
        V = N.View(az=100.0, el=20.0, R=1.0, shift=(xc + 1.25, -1.0))
        b.append("\\fill[ndSky!45] %s circle (%.3f);" % (N.pt(V.shift), V.R))
        b += V.great_circle([0, 0, 1])
        P = ball_image(Ak, TSTAR)
        P2 = np.array([V.P(p) for p in P])
        hull = ConvexHull(P2)
        col = "ndRed" if k == 2 else "ndBlue"
        b.append("\\filldraw[%s, line width=0.8pt, fill=%s!25, fill opacity=0.8] %s -- cycle;" % (col, col, " -- ".join(N.pt(P2[i]) for i in hull.vertices)))
        b.append("\\draw[ink] %s circle (%.3f);" % (N.pt(V.shift), V.R))
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {shaded: the image of the\\\\ Bloch ball after $t^*=\\pi/(2\\omega)$};" % (cols[0] + 1.25, -2.1))
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {a quantum channel: the\\\\ ball is mapped into itself};" % (cols[1] + 1.25, -2.1))
    b.append("\\node[note, anchor=north, text=ndRed] at (%.3f,%.3f) {not a channel: part of the\\\\ ball is carried outside it};" % (cols[2] + 1.25, -2.1))
    # ---- the common output
    Pl = N.Plot((0, 12), (-0.45, 0.9), 4.6, 3.0, origin=(11.6, 0.3))
    b += Pl.axes("$t$", "", xticks=[(5, "$5$"), (10, "$10$")], yticks=[(0, "$0$"), (0.5, "$0.5$")], yzero=0.0)
    b.append(Pl.curve(t, ys[0], "line width=1.1pt, ndInk"))
    for y, col, off, mark in [(ys[1], "ndBlue", 0.0, "circle"), (ys[2], "ndRed", 0.5, "rectangle")]:
        for s in np.arange(off, 12.01, 1.0):
            v = np.interp(s, t, y)
            p = Pl.P(s, v)
            if mark == "circle":
                b.append("\\draw[%s, line width=0.7pt, fill=white] (%.3f,%.3f) circle (1.7pt);" % (col, p[0], p[1]))
            else:
                b.append("\\draw[%s, line width=0.7pt, fill=white] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, p[0] - 0.055, p[1] - 0.055, p[0] + 0.055, p[1] + 0.055))
    b.append("\\node[font=\\small, anchor=south] at %s {$\\langle\\sigma_n\\rangle(t)$};" % Pl.pt(6, 0.98))
    b.append("\\node[note, anchor=north] at %s {the same curve for all three:\\\\ line $(\\mathbf A,\\mathbf C,\\vec x_0)$,\\\\ circles $\\mathbf T_1$, squares $\\mathbf T_2$};" % Pl.pt(6, -0.5))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) three realisations and their images of the Bloch ball};" % (5.3, -3.35))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) the data};" % (13.9, -3.35))
    N.write("ch05_gauge", b)
    print("written figures/ch05_gauge.tex; max radius of the image at t*: %s" % np.round(radii, 3))


if __name__ == "__main__":
    main()
