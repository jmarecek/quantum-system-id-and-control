"""ch12_landscape_traps.py -- a control landscape with a trap (Pechen & Tannor).

Three-level system, H = H0 + lambda V with a constant control lambda on [0,1]; the
objective <O>(lambda) has a local maximum (trap) at lambda = 0 and global maxima
elsewhere.  Writes figures/ch12_landscape_traps.tex.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm
import tikzexport as T

H0 = np.diag([0.0, 1.0, 2.0])
V = np.zeros((3, 3)); V[0, 2] = V[2, 0] = V[1, 2] = V[2, 1] = 1.0
O = np.diag([0.1, 0.0, 1.0])
psi0 = np.array([1, 0, 0], dtype=complex)


def objective(lam, Ttot=1.0):
    psi = expm(-1j * Ttot * (H0 + lam * V)) @ psi0
    return np.real(np.vdot(psi, O @ psi))


def main():
    lams = np.linspace(-6, 6, 601)
    J = np.array([objective(l) for l in lams])
    # local maxima on the grid
    idx = [i for i in range(1, len(J) - 1) if J[i] >= J[i - 1] and J[i] >= J[i + 1]]
    best = idx[int(np.argmax(J[idx]))]
    body = T.pgfplots_axis([dict(x=lams, y=J, label="$\\langle O\\rangle(\\lambda)$")],
                           xlabel="control amplitude $\\lambda$", ylabel="objective",
                           legend_pos="south east")
    marks = "\n".join("\\addplot[only marks, mark=x, mark size=3pt, red, thick] coordinates {(%.4f,%.4f)};"
                      % (lams[i], J[i]) for i in idx)
    glob = "\\addplot[only marks, mark=*, mark size=3pt, qocC] coordinates {(%.4f,%.4f)};" % (lams[best], J[best])
    body = body.replace("\\end{axis}", marks + "\n" + glob + "\n\\end{axis}")
    T.write_tikz("ch12_landscape_traps", body)
    print("local maxima at", [round(lams[i], 2) for i in idx], "global", round(lams[best], 2))


if __name__ == "__main__":
    main()
