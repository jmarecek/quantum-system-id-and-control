"""ch01_spectrahedron.py -- a slice of the positive semidefinite cone: the
elliptope {(x,y,z): [[1,x,y],[x,1,z],[y,z,1]] >= 0} cut at z = 0.3, i.e. a
spectrahedron in the plane, compared with the box |x|, |y| <= 1 cut out by the
2x2 minors alone.  In the plane z = 0.3 the determinant is quadratic, so the
slice is the ellipse x^2 + y^2 - 0.6 xy <= 0.91, which touches the box at the
four points (+-1, +-0.3), (+-0.3, +-1).
Writes figures/ch01_spectrahedron.tex.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

Z = 0.3


def psd(x, y, z=Z):
    return np.linalg.eigvalsh(np.array([[1, x, y], [x, 1, z], [y, z, 1]])).min() >= -1e-12


def main():
    # boundary of the slice: along each ray from the origin, the largest r with the matrix PSD
    pts = []
    for ang in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        lo, hi = 0.0, 1.5
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if psd(mid * np.cos(ang), mid * np.sin(ang)) else (lo, mid)
        pts.append((lo * np.cos(ang), lo * np.sin(ang)))
    pts = np.array(pts)
    q = pts[:, 0] ** 2 + pts[:, 1] ** 2 - 2 * Z * pts[:, 0] * pts[:, 1]
    assert np.allclose(q, 1 - Z ** 2, atol=1e-9)                 # the slice is an ellipse
    assert abs(np.abs(pts[:, 0]).max() - 1) < 1e-4 and abs(np.abs(pts[:, 1]).max() - 1) < 1e-4   # inscribed in the box
    for x, y in [(1, Z), (-1, -Z), (Z, 1), (-Z, -1)]:
        assert psd(x, y)                                         # the four points of contact
    assert not psd(1, 1) and not psd(0.9, -0.9)                  # the corners of the box are not in it

    S = 2.0                                                      # cm per unit
    P = lambda p: (S * p[0], S * p[1])
    b = []
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[right, text=ndInk] {$x$};" % (-1.3 * S, 1.35 * S))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[above, text=ndInk] {$y$};" % (-1.3 * S, 1.35 * S))
    b.append("\\draw[dashedink] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (-S, -S, S, S))
    b.append("\\fill[ndSky, fill opacity=0.7] %s -- cycle;" % " -- ".join(N.pt(P(p)) for p in pts))
    b.append("\\draw[line width=1.1pt, ndBlue] %s -- cycle;" % " -- ".join(N.pt(P(p)) for p in pts))
    for x, y in [(1, Z), (-1, -Z), (Z, 1), (-Z, -1)]:
        b.append("\\node[dot, fill=ndRed] at %s {};" % N.pt(P((x, y))))
    for v, lab in [(-1, "$-1$"), (1, "$1$")]:
        b.append("\\node[font=\\scriptsize, anchor=north] at (%.2f,-0.04) {%s};" % (S * v + (0.18 if v > 0 else -0.2), lab))
        b.append("\\node[font=\\scriptsize, anchor=east] at (-0.04,%.2f) {%s};" % (S * v + (0.18 if v > 0 else -0.2), lab))
    b.append("\\node[anchor=west] at (%.2f,%.2f) {$\\begin{pmatrix}1&x&y\\\\ x&1&z\\\\ y&z&1\\end{pmatrix}\\succeq 0,\\quad z=%.1f$};" % (1.55 * S, 1.05 * S, Z))
    b.append("\\node[note, anchor=west, align=left, text=ndBlue] (s) at %s {the spectrahedron: an affine slice of the\\\\ PSD cone, convex, with a curved boundary;\\\\ in this plane the determinant is quadratic,\\\\ and the slice is an ellipse};" % N.pt((1.55 * S, 0.3 * S)))
    b.append("\\draw[pointer] (s.west) to[bend right=15] %s;" % N.pt(P((0.62, 0.55))))
    b.append("\\node[note, anchor=west, align=left] (bx) at %s {dashed: the box $\\abs{x},\\abs{y}\\le 1$ cut out by\\\\ the $2\\times2$ minors alone; the ellipse\\\\ touches it at four points (red), and\\\\ the corners of the box are not in it};" % N.pt((1.55 * S, -0.62 * S)))
    b.append("\\draw[pointer] (bx.west) to[bend left=10] %s;" % N.pt(P((1.0, -0.6))))
    N.write("ch01_spectrahedron", b)
    print("written figures/ch01_spectrahedron.tex")


if __name__ == "__main__":
    main()
