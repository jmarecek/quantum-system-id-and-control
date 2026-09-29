"""ch01_spectrahedron.py -- a slice of the positive semidefinite cone: the
elliptope {(x,y,z): [[1,x,y],[x,1,z],[y,z,1]] >= 0} cut at z = 0.3, i.e. a
spectrahedron in the plane, compared with the polytope of its linear
inequalities.  Writes figures/ch01_spectrahedron.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

z = 0.3
g = np.linspace(-1, 1, 401)
Xg, Yg = np.meshgrid(g, g)
# det of the 3x3 correlation matrix and the 2x2 minors
det = 1 + 2 * Xg * Yg * z - Xg ** 2 - Yg ** 2 - z ** 2
feas = (det >= 0) & (np.abs(Xg) <= 1) & (np.abs(Yg) <= 1)
# boundary polygon: trace along angles
pts = []
for ang in np.linspace(0, 2 * np.pi, 240, endpoint=False):
    r = np.linspace(0, 1.5, 600); xs = r * np.cos(ang); ys = r * np.sin(ang)
    ok = (1 + 2 * xs * ys * z - xs ** 2 - ys ** 2 - z ** 2 >= 0) & (np.abs(xs) <= 1) & (np.abs(ys) <= 1)
    k = np.where(~ok)[0]; k = k[0] - 1 if len(k) else len(r) - 1
    pts.append((xs[k], ys[k]))
poly = " -- ".join("(%.3f,%.3f)" % p for p in pts) + " -- cycle"
body = r"""
\begin{scope}[x=2.2cm,y=2.2cm]
\draw[->] (-1.3,0) -- (1.3,0) node[right] {$x$};
\draw[->] (0,-1.3) -- (0,1.3) node[above] {$y$};
\draw[gray, dashed] (-1,-1) rectangle (1,1);
\fill[qocA!30, draw=qocA, thick] %s;
\node[font=\small, align=left, anchor=west] at (1.35,0.7) {$\begin{pmatrix}1&x&y\\ x&1&z\\ y&z&1\end{pmatrix}\succeq 0$, \ $z=%.1f$};
\node[font=\small, align=left, anchor=west, text=qocA] at (1.35,0.1) {spectrahedron: an affine slice\\ of the PSD cone; convex,\\ boundary curved (a cubic)};
\node[font=\small, align=left, anchor=west, text=gray] at (1.35,-0.5) {dashed: the box $|x|,|y|\le 1$\\ from the $2\times 2$ minors alone};
\end{scope}
""" % (poly, z)
T.write_tikz("ch01_spectrahedron", body)
print("written figures/ch01_spectrahedron.tex")
