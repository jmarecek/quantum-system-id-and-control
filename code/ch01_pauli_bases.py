"""ch01_pauli_bases.py -- the six eigenstates of the Pauli matrices on the Bloch
sphere: measuring sigma_x, sigma_y, sigma_z is an informationally complete set of
measurements for a qubit.  A Bloch vector r is drawn with its three components,
which are what the three measurements estimate: r_j = P(+1) - P(-1).
Writes figures/ch01_pauli_bases.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

V = N.View(az=25.0, el=18.0, R=2.0)
AXES = [((1, 0, 0), "ndBlue", "x", "$\\ket{+}$", "$\\ket{-}$"),
        ((0, 1, 0), "ndGold", "y", "$\\ket{+\\imath}$", "$\\ket{-\\imath}$"),
        ((0, 0, 1), "ndGreen", "z", "$\\ket{0}$", "$\\ket{1}$")]
R_STATE = np.array([0.45, 0.35, 0.6])                    # a mixed state, |r| < 1


def main():
    assert np.linalg.norm(R_STATE) < 1
    b = V.sphere()
    b += V.great_circle([0, 0, 1])
    for ax, col, j, plus, minus in AXES:
        a = np.array(ax, dtype=float)
        # the axis: dotted inside the ball, solid outside it
        b.append("\\draw[%s, line width=0.5pt, densely dotted] %s -- %s;" % (col, V.pt(-a), V.pt(a)))
        b.append("\\draw[%s, line width=1.1pt, ->, >=stealth] %s -- %s node[%s, text=%s!85!ndInk] {$\\sigma_%s$};"
                 % (col, V.pt(a), V.pt(1.32 * a), {"x": "below left", "y": "below", "z": "above"}[j], col, j))
        b.append("\\draw[%s, line width=1.1pt] %s -- %s;" % (col, V.pt(-a), V.pt(-1.15 * a)))
        vis = lambda v: "" if V.visible(v) else ", fill=white, draw=%s, line width=0.6pt" % col
        b.append("\\node[dot, fill=%s%s] at %s {};" % (col, vis(a), V.pt(a)))
        b.append("\\node[dot, fill=%s%s] at %s {};" % (col, vis(-a), V.pt(-a)))
        pos_p = {"x": "above left", "y": "above", "z": "above right"}[j]
        pos_m = {"x": "above right", "y": "above", "z": "below right"}[j]
        b.append("\\node[font=\\footnotesize, %s, inner sep=2pt] at %s {%s};" % (pos_p, V.pt(a), plus))
        b.append("\\node[font=\\footnotesize, %s, inner sep=2pt] at %s {%s};" % (pos_m, V.pt(-a), minus))
    # a state and its three components
    r = R_STATE
    for k, (ax, col, j, _, _) in enumerate(AXES):
        foot = r[k] * np.array(ax, dtype=float)
        b.append("\\draw[%s, line width=0.5pt, dash pattern=on 2pt off 1.5pt] %s -- %s;" % (col, V.pt(r), V.pt(foot)))
        b.append("\\fill[%s] %s circle (1.6pt);" % (col, V.pt(foot)))
    b.append("\\draw[vec, ndRed] %s -- %s;" % (V.pt((0, 0, 0)), V.pt(r)))
    b.append("\\node[font=\\footnotesize, text=ndRed, above right, inner sep=1pt] at %s {$\\Bloch$};" % V.pt(r))
    # notes
    x0 = V.R + 1.1
    b.append("\\node[note, anchor=west, align=left] at (%.2f,1.45) {three measurement axes,\\\\ six outcomes: an\\\\"
             " \\emph{informationally complete}\\\\ set for one qubit};" % x0)
    b.append("\\node[note, anchor=west, align=left, text=ndRed] (r) at (%.2f,-0.25) {the dots on the axes are\\\\"
             " the components $r_x,r_y,r_z$\\\\ that the three measurements\\\\ estimate};" % x0)
    b.append("\\node[anchor=west, align=left, font=\\footnotesize] at (%.2f,-1.75) {$\\rho=\\tfrac12\\big(\\Id+r_x\\sigma_x+r_y\\sigma_y+r_z\\sigma_z\\big)$,\\\\"
             " $r_j=\\Tr(\\rho\\sigma_j)=\\Prob(+1)-\\Prob(-1)$};" % x0)
    N.write("ch01_pauli_bases", b)
    print("written figures/ch01_pauli_bases.tex")


if __name__ == "__main__":
    main()
