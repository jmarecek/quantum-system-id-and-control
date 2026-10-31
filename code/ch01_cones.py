"""ch01_cones.py -- the three fundamental cones of conic optimisation, as TikZ.

Writes figures/ch01_cones.tex with three panels:
  (a) the nonnegative orthant R^2_+,
  (b) the Lorentz (second-order, ice-cream) cone  t >= ||x||_2  in R^3, cut at t = 1,
  (c) a spectrahedron (slice of the PSD cone): the elliptope of 3x3 correlation
      matrices [[1,x,y],[x,1,z],[y,z,1]] >= 0, whose boundary is the cubic
      surface 1 + 2xyz - x^2 - y^2 - z^2 = 0.  It has four corners, the
      rank-one matrices (+-1, +-1, +-1) with xyz = 1, and contains the six
      segments joining them; elsewhere it is curved.
Surfaces are sampled with numpy and drawn in orthographic projection
(needham.View): front parts solid, hidden parts dotted.
Run from the course root:  python3 code/ch01_cones.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import ConvexHull
import needham as N


def split(V, pts, front, front_style, back_style):
    """Polylines of pts (3D), solid where `front` holds and in back_style elsewhere."""
    out = []
    for m, style in [(front, front_style), (~front, back_style)]:
        if style is None:
            continue
        idx = np.where(m)[0]
        for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(seg) > 1:
                out.append(N.polyline([V.P(p) for p in pts[seg]], style))
    return out


def hull(points2d):
    h = ConvexHull(points2d)
    return [points2d[i] for i in h.vertices]


def orthant(b, ox):
    s = 2.3
    b.append("\\fill[ndSky, fill opacity=0.55] (%.3f,0) rectangle (%.3f,%.3f);" % (ox, ox + s, s))
    b.append("\\draw[faint, ->, >=stealth] (%.3f,0) -- (%.3f,0) node[right, text=ndInk] {$x_1$};" % (ox - 0.5, ox + s + 0.4))
    b.append("\\draw[faint, ->, >=stealth] (%.3f,-0.5) -- (%.3f,%.3f) node[above, text=ndInk] {$x_2$};" % (ox, ox, s + 0.4))
    b.append("\\draw[ink] (%.3f,%.3f) -- (%.3f,0) -- (%.3f,0);" % (ox, s, ox, ox + s))
    b.append("\\node at (%.3f,%.3f) {$\\R^2_+$};" % (ox + 1.15, 1.45))
    b.append("\\node[note] at (%.3f,%.3f) {two half-planes:\\\\ flat sides,\\\\ one corner};" % (ox + 1.2, 0.75))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,-0.75) {(a) the orthant $\\R^2_+$};" % (ox + 1.0))


def lorentz(b, ox):
    V = N.View(az=30.0, el=20.0, R=1.75, shift=(ox, 0.0))
    d = V.d
    phi = np.linspace(0, 2 * np.pi, 241)
    rim = np.stack([np.cos(phi), np.sin(phi), np.ones_like(phi)], 1)
    apex = np.zeros(3)
    outline = hull(np.array([V.P(apex)] + [V.P(p) for p in rim]))
    b.append("\\fill[ndSky, fill opacity=0.55] %s -- cycle;" % " -- ".join(N.pt(p) for p in outline))
    b.append("\\fill[ndSky, fill opacity=0.6] %s -- cycle;" % " -- ".join(V.pt(p) for p in rim[:-1]))
    # generators: the outer surface faces the viewer where cos(phi - az) cos(el) > sin(el)
    az, el = np.radians(30.0), np.radians(20.0)
    for p in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        if np.cos(p - az) * np.cos(el) > np.sin(el) + 0.05:
            b.append("\\draw[faint] %s -- %s;" % (V.pt(apex), V.pt([np.cos(p), np.sin(p), 1.0])))
    for s in (+1, -1):                                         # the two silhouette generators
        p = az + s * np.arccos(np.tan(el))
        g = np.array([np.cos(p), np.sin(p), 1.0])
        assert abs(np.array([np.cos(p), np.sin(p), -1.0]) @ d) < 1e-9
        b.append("\\draw[ink] %s -- %s;" % (V.pt(apex), V.pt(g)))
    b.append(N.polyline([V.P(p) for p in rim], "ink"))
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (V.pt(apex), V.pt([0, 0, 1])))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, text=ndInk] {$t$};" % (V.pt([0, 0, 1]), V.pt([0, 0, 1.55])))
    b.append("\\node[dot] at %s {};" % V.pt(apex))
    b.append("\\node[note, anchor=west] (r) at %s {the rim $\\norm{x}_2=t$:\\\\ round, no corners};" % N.pt(V.P([0, 1, 1]) + np.array([0.35, 0.75])))
    b.append("\\draw[pointer] (r.south west) to[bend right=20] %s;" % V.pt([np.cos(1.9), np.sin(1.9), 1.0]))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,-0.75) {(b) the Lorentz cone $t\\ge\\norm{x}_2$};" % ox)


def elliptope(b, ox, az=45.0, el=15.0):
    V = N.View(az=az, el=el, R=1.25, shift=(ox, 0.8))
    d = V.d
    normal = lambda p: np.stack([p[:, 0] - p[:, 1] * p[:, 2], p[:, 1] - p[:, 0] * p[:, 2], p[:, 2] - p[:, 0] * p[:, 1]], 1)
    det = lambda p: 1 + 2 * p[:, 0] * p[:, 1] * p[:, 2] - (p ** 2).sum(1)

    def slice_z(z, n=181):
        th = np.linspace(0, 2 * np.pi, n)
        u, v = np.sqrt(1 + z) * np.cos(th), np.sqrt(1 - z) * np.sin(th)
        return np.stack([(u + v) / np.sqrt(2), (u - v) / np.sqrt(2), np.full_like(th, z)], 1)

    def slice_x(x, n=181):
        p = slice_z(x, n)
        return p[:, [2, 0, 1]]

    surface = np.concatenate([slice_z(z, 90) for z in np.linspace(-0.999, 0.999, 120)])
    assert np.abs(det(surface)).max() < 1e-9                 # the samples lie on the boundary
    outline = hull(np.array([V.P(p) for p in surface]))
    b.append("\\fill[ndSky, fill opacity=0.55] %s -- cycle;" % " -- ".join(N.pt(p) for p in outline))
    for z in np.linspace(-0.75, 0.75, 7):
        p = slice_z(z)
        b += split(V, p, normal(p) @ d >= 0, "line width=0.4pt, ndBlue!70", "line width=0.3pt, ndBlue!40, densely dotted")
    verts = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], dtype=float)
    for v in verts:                                          # the vertices are the rank-one correlation matrices
        assert np.linalg.matrix_rank(np.array([[1, v[0], v[1]], [v[0], 1, v[2]], [v[1], v[2], 1]])) == 1
    s = np.linspace(-1, 1, 121)[1:-1]
    for i in range(4):
        for j in range(i + 1, 4):
            p = (verts[i] + verts[j]) / 2 + np.outer(s, (verts[i] - verts[j]) / 2)
            assert np.abs(det(p)).max() < 1e-12              # the six edges lie on the boundary
            b += split(V, p, normal(p) @ d >= 0, "ink", "dashedink")
    b.append("\\draw[ink] %s -- cycle;" % " -- ".join(N.pt(p) for p in outline))
    for v in verts:
        b.append("\\node[dot, fill=ndRed] at %s {};" % V.pt(v))
    b.append("\\node[note, anchor=west, text=ndRed] (v) at %s {four corners:\\\\ the rank-one\\\\ matrices};" % N.pt(V.P(verts[3]) + np.array([0.45, 0.35])))
    b.append("\\draw[pointer] (v.west) to[bend right=15] %s;" % V.pt(verts[3]))
    b.append("\\node[font=\\small, anchor=north, align=center] at (%.3f,-0.75) {(c) a spectrahedron:\\\\[2pt] $\\begin{pmatrix}1&x&y\\\\ x&1&z\\\\ y&z&1\\end{pmatrix}\\succeq 0$};" % ox)


def main():
    b = []
    orthant(b, 0.0)
    lorentz(b, 5.6)
    elliptope(b, 10.9)
    N.write("ch01_cones", b)
    print("wrote ch01_cones.tex")


if __name__ == "__main__":
    main()
