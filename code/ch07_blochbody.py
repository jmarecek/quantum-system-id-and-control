"""ch07_blochbody.py -- planar sections of the generalised Bloch body.

For a qutrit, rho = 1/3 + (1/2) sum_j x_j lambda_j with the Gell-Mann matrices
lambda_j (Tr lambda_j lambda_k = 2 delta_jk), so that x_j = Tr(rho lambda_j).
We draw the set of coherence vectors x lying in a 2-plane for which rho >= 0,
by bisection on the radius in each direction, and compare with the qubit disc.
Output: figures/ch07_blochbody.tex
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T


def gell_mann():
    l = np.zeros((8, 3, 3), dtype=complex)
    l[0][0, 1] = l[0][1, 0] = 1
    l[1][0, 1] = -1j; l[1][1, 0] = 1j
    l[2][0, 0] = 1; l[2][1, 1] = -1
    l[3][0, 2] = l[3][2, 0] = 1
    l[4][0, 2] = -1j; l[4][2, 0] = 1j
    l[5][1, 2] = l[5][2, 1] = 1
    l[6][1, 2] = -1j; l[6][2, 1] = 1j
    l[7] = np.diag([1, 1, -2]) / np.sqrt(3)
    return l


def is_state(x, basis, d):
    rho = np.eye(d) / d + 0.5 * sum(xj * Fj for xj, Fj in zip(x, basis))
    return np.linalg.eigvalsh(rho).min() >= -1e-12


def section(basis, d, a, b, n_angles=240):
    """Boundary of {x in span(e_a, e_b) : rho(x) >= 0} as a polygon."""
    pts = []
    for th in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        lo, hi = 0.0, 2.0
        for _ in range(40):                     # bisection on the radius
            mid = 0.5 * (lo + hi)
            x = np.zeros(len(basis)); x[a] = mid * np.cos(th); x[b] = mid * np.sin(th)
            if is_state(x, basis, d):
                lo = mid
            else:
                hi = mid
        pts.append((lo * np.cos(th), lo * np.sin(th)))
    return np.array(pts)


def panel(poly, xshift, title, r_in, r_out, colour):
    s = 1.6                                     # cm per unit
    out = ["\\begin{scope}[shift={(%.2f,0)}]" % xshift]
    path = " -- ".join("(%.3f,%.3f)" % (s * p[0], s * p[1]) for p in poly)
    out.append("\\fill[%s!25] %s -- cycle;" % (colour, path))
    out.append("\\draw[thick,%s] %s -- cycle;" % (colour, path))
    out.append("\\draw[dashed,gray] (0,0) circle (%.3f);" % (s * r_out))
    out.append("\\draw[dotted,gray] (0,0) circle (%.3f);" % (s * r_in))
    out.append("\\draw[->,gray] (%.2f,0) -- (%.2f,0);" % (-s * 1.3, s * 1.3))
    out.append("\\draw[->,gray] (0,%.2f) -- (0,%.2f);" % (-s * 1.3, s * 1.3))
    out.append("\\node[font=\\small] at (0,%.2f) {%s};" % (s * 1.55, title))
    out.append("\\end{scope}")
    return "\n".join(out)


if __name__ == "__main__":
    I, X, Y, Z = T.pauli()
    qubit = section([X, Y, Z], 2, 0, 2)          # (x, z) plane: the unit disc
    gm = gell_mann()
    tri = section(gm, 3, 2, 7)                    # (lambda_3, lambda_8): diagonal states
    mix = section(gm, 3, 0, 7)                    # (lambda_1, lambda_8)
    r_out3, r_in3 = np.sqrt(4.0 / 3.0), np.sqrt(1.0 / 3.0)
    body = "\n".join([
        panel(qubit, 0.0, "qubit, $(x_1,x_3)$", 1.0, 1.0, "qocA"),
        panel(tri, 5.2, "qutrit, $(x_3,x_8)$", r_in3, r_out3, "qocB"),
        panel(mix, 10.4, "qutrit, $(x_1,x_8)$", r_in3, r_out3, "qocC"),
    ])
    T.write_tikz("ch07_blochbody", body, options="scale=0.9")
    print("wrote figures/ch07_blochbody.tex")
