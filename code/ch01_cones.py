"""ch01_cones.py -- the three fundamental cones of conic optimisation, as TikZ.

Writes figures/ch01_cones.tex with three panels generated from numpy meshes:
  (a) the nonnegative orthant R^2_+,
  (b) the Lorentz (second-order, ice-cream) cone  t >= ||x||_2  in R^3,
  (c) a spectrahedron (slice of the PSD cone): the elliptope of 3x3 correlation matrices.
Run from the course root:  python3 code/ch01_cones.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tikzexport as T


def proj(x, y, z):
    """Oblique projection of a 3D point onto the page."""
    return (x - 0.5 * y, z + 0.3 * y)


def path(points):
    return " -- ".join("(%.3f,%.3f)" % p for p in points)


out = []
# (a) nonnegative orthant -------------------------------------------------
out.append("\\begin{scope}[shift={(0,0)}]")
out.append("\\fill[qocA!25] (0,0) -- (2.4,0) -- (2.4,2.4) -- (0,2.4) -- cycle;")
out.append("\\draw[->] (-0.3,0) -- (2.7,0) node[right] {$x_1$};")
out.append("\\draw[->] (0,-0.3) -- (0,2.7) node[above] {$x_2$};")
out.append("\\node[font=\\small] at (1.2,1.2) {$\\R^2_+$};")
out.append("\\node[font=\\small] at (1.2,-0.8) {(a) orthant};")
out.append("\\end{scope}")

# (b) Lorentz cone --------------------------------------------------------
out.append("\\begin{scope}[shift={(5.2,0)}]")
h = 2.2
phi = np.linspace(0, 2 * np.pi, 80)
rim = [proj(h * np.cos(p), h * np.sin(p), h) for p in phi]
out.append("\\fill[qocB!25] (0,0) -- %s -- cycle;" % path(rim))
out.append("\\draw[thick] %s -- cycle;" % path(rim))
# generators of the cone (rays from the apex)
for p in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    x, y = proj(h * np.cos(p), h * np.sin(p), h)
    out.append("\\draw[qocB!70!black, thin] (0,0) -- (%.3f,%.3f);" % (x, y))
out.append("\\draw[->] (0,-0.3) -- (0,3.0) node[above] {$t$};")
out.append("\\node[font=\\small] at (0,-0.8) {(b) Lorentz cone $t\\ge\\|x\\|_2$};")
out.append("\\end{scope}")

# (c) a spectrahedron: the elliptope of 3x3 correlation matrices ------------
#     E = {(x,y,z): [[1,x,y],[x,1,z],[y,z,1]] >= 0}, boundary 1 + 2xyz - x^2 - y^2 - z^2 = 0.
#     Slices z = const are ellipses: with u=(x+y)/sqrt2, v=(x-y)/sqrt2,
#     u^2 (1-z) + v^2 (1+z) = 1 - z^2, i.e. u = sqrt(1+z) cos th, v = sqrt(1-z) sin th.
out.append("\\begin{scope}[shift={(10.6,0.4)}, scale=1.15]")
def slice_z(z, n=80):
    th = np.linspace(0, 2 * np.pi, n)
    u = np.sqrt(1 + z) * np.cos(th); v = np.sqrt(1 - z) * np.sin(th)
    x = (u + v) / np.sqrt(2); y = (u - v) / np.sqrt(2)
    return [proj(a, b, z) for a, b in zip(x, y)]
def slice_x(x0, n=80):   # same body, sliced along x (permute coordinates)
    th = np.linspace(0, 2 * np.pi, n)
    u = np.sqrt(1 + x0) * np.cos(th); v = np.sqrt(1 - x0) * np.sin(th)
    y = (u + v) / np.sqrt(2); z = (u - v) / np.sqrt(2)
    return [proj(x0, a, b) for a, b in zip(y, z)]
out.append("\\fill[qocC!20, opacity=0.7] %s -- cycle;" % path(slice_z(0.0)))
for z in np.linspace(-0.9, 0.9, 7):
    out.append("\\draw[qocC!%d!black, thin] %s -- cycle;" % (int(35 + 60 * (z + 1) / 2), path(slice_z(z))))
for x0 in np.linspace(-0.9, 0.9, 7):
    out.append("\\draw[qocC!60!black, very thin, opacity=0.6] %s -- cycle;" % path(slice_x(x0)))
# the four vertices (rank-one correlation matrices) and the edges between them
verts = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
for v in verts:
    px, py = proj(*v)
    out.append("\\fill[qocC!50!black] (%.3f,%.3f) circle (1.4pt);" % (px, py))
ax, ay = proj(1.6, 0, 0); out.append("\\draw[->, gray] (0,0) -- (%.3f,%.3f) node[right, black] {$x$};" % (ax, ay))
ax, ay = proj(0, 1.6, 0); out.append("\\draw[->, gray] (0,0) -- (%.3f,%.3f) node[left, black] {$y$};" % (ax, ay))
ax, ay = proj(0, 0, 1.6); out.append("\\draw[->, gray] (0,0) -- (%.3f,%.3f) node[above, black] {$z$};" % (ax, ay))
out.append("\\node[font=\\small, align=center] at (0.3,-1.9) {(c) spectrahedron $\\begin{pmatrix}1&x&y\\\\ x&1&z\\\\ y&z&1\\end{pmatrix}\\succeq 0$};")
out.append("\\end{scope}")

T.write_tikz("ch01_cones", "\n".join(out), options="scale=0.9")
print("wrote ch01_cones.tex")
