"""ch01_pauli_bases.py -- the six eigenstates of the Pauli matrices on the Bloch
sphere: measuring sigma_x, sigma_y, sigma_z is an informationally complete set of
measurements for a qubit.  Writes figures/ch01_pauli_bases.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

pts = [((1, 0, 0), r"$\ket{+}$"), ((-1, 0, 0), r"$\ket{-}$"), ((0, 1, 0), r"$\ket{+i}$"), ((0, -1, 0), r"$\ket{-i}$")]
body = T.bloch_sphere(radius=1.8, points=pts)
R = 1.8
def P(v):
    x, y, z = v; return (R * (0.9 * x - 0.45 * y), R * (z - 0.2 * y))
for a, b, col, lab in [((1, 0, 0), (-1, 0, 0), "qocA", r"$\sigma_x$"), ((0, 1, 0), (0, -1, 0), "qocB", r"$\sigma_y$"), ((0, 0, 1), (0, 0, -1), "qocC", r"$\sigma_z$")]:
    pa, pb = P(a), P(b)
    body += "\n\\draw[very thick, %s, <->] (%.3f,%.3f) -- (%.3f,%.3f);" % (col, pa[0], pa[1], pb[0], pb[1])
    body += "\n\\node[text=%s, font=\\small] at (%.3f,%.3f) {%s};" % (col, 0.55 * pa[0] + 0.25, 0.55 * pa[1] + 0.25, lab)
body += r"""
\node[font=\small, align=left, anchor=west] at (3.0,1.4) {three measurement axes,\\ six outcomes: an\\ \emph{informationally complete}\\ set for one qubit};
\node[font=\small, align=left, anchor=west] at (3.0,-0.6) {$\rho=\tfrac12\big(\Id+r_x\sigma_x+r_y\sigma_y+r_z\sigma_z\big)$,\\ $r_j=\Tr(\rho\sigma_j)=\Prob(+1)-\Prob(-1)$};
"""
T.write_tikz("ch01_pauli_bases", body)
print("written figures/ch01_pauli_bases.tex")
