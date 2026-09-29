"""ch01_duality.py -- a conic program in the plane: minimise c^T x over the
intersection of the second-order cone {(x1,x2,x3): x3 >= |(x1,x2)|} with the
affine plane x3 = 1 (a disc) and a line a^T x = b; the optimum, the level sets of
the objective and the dual certificate (a supporting line).  Writes figures/ch01_duality.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

c = np.array([1.0, 0.6]); c /= np.linalg.norm(c)
a = np.array([0.3, 1.0]); a /= np.linalg.norm(a); b = 0.35
# feasible set: unit disc intersect line a.x = b  -> a chord; minimise c.x on the chord
# chord endpoints
p0 = b * a; tdir = np.array([-a[1], a[0]]); h = np.sqrt(1 - b ** 2)
e1, e2 = p0 - h * tdir, p0 + h * tdir
xopt = e1 if c @ e1 < c @ e2 else e2
val = c @ xopt
body = r"""
\begin{scope}[x=2.4cm,y=2.4cm]
\fill[qocA!15, draw=qocA, thick] (0,0) circle (1);
\node[text=qocA, font=\small] at (-0.55,-0.75) {cone slice $\{x: \|x\|\le 1\}$};
\draw[qocB, very thick] (%.3f,%.3f) -- (%.3f,%.3f) node[pos=0.5, above, sloped, font=\small] {feasible set $a^\top x=b$};
\fill[qocD] (%.3f,%.3f) circle (1.5pt) node[below right, font=\small] {$x^\star$};
""" % (e1[0], e1[1], e2[0], e2[1], xopt[0], xopt[1])
# level sets of c^T x
for k in [-0.9, -0.5, val, 0.3, 0.7]:
    n = np.array([-c[1], c[0]])
    p = k * c
    body += r"\draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);" % ("qocD, thick" if abs(k - val) < 1e-9 else "gray!60, dashed", *(p - 1.4 * n), *(p + 1.4 * n)) + "\n"
body += r"\draw[->, thick] (0,0) -- (%.3f,%.3f) node[right, font=\small] {$c$};" % (0.5 * c[0], 0.5 * c[1]) + "\n"
body += r"\node[font=\small, align=left, anchor=west] at (1.15,0.75) {primal: $\min\ c^\top x$ s.t. $a^\top x=b$, $x\in K$};" + "\n"
body += r"\node[font=\small, align=left, anchor=west] at (1.15,0.45) {dual: $\max\ b\,y$ s.t. $c-a\,y\in K^*$};" + "\n"
body += r"\node[font=\small, align=left, anchor=west, text=qocD] at (1.15,0.1) {the level set $c^\top x=c^\top x^\star$ touches the\\ feasible set: zero duality gap};" + "\n"
body += r"\end{scope}"
T.write_tikz("ch01_duality", body)
print("written figures/ch01_duality.tex  value %.3f" % val)
