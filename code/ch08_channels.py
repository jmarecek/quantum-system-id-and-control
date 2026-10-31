"""ch08_channels.py -- Choi matrices and the image of the Bloch ball.

Part 1 (snippet): Kraus operators -> Choi matrix, and the CPTP test.
Part 2: the (x,z)-section of the Bloch ball under amplitude damping and
dephasing channels, drawn as ellipses.  Output: figures/ch08_channels.tex
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T


def choi(kraus):
    """Choi matrix J = sum_ij |i><j| (x) E(|i><j|) of the channel with Kraus ops."""
    d = kraus[0].shape[0]
    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex); Eij[i, j] = 1
            out = sum(K @ Eij @ K.conj().T for K in kraus)
            J += np.kron(Eij, out)
    return J


def is_cptp(kraus, tol=1e-10):
    J = choi(kraus)
    cp = np.linalg.eigvalsh(J).min() >= -tol                     # complete positivity
    tp = np.allclose(sum(K.conj().T @ K for K in kraus), np.eye(kraus[0].shape[0]))
    return cp and tp


def amplitude_damping(g):
    return [np.array([[1, 0], [0, np.sqrt(1 - g)]]), np.array([[0, np.sqrt(g)], [0, 0]])]


def dephasing(p):
    I, X, Y, Z = T.pauli()
    return [np.sqrt(1 - p) * I, np.sqrt(p) * Z]


def bloch_image(kraus, n=120):
    """Image of the unit circle in the (x,z)-plane of the Bloch ball."""
    I, X, Y, Z = T.pauli()
    pts = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        rho = 0.5 * (I + np.cos(th) * X + np.sin(th) * Z)
        out = sum(K @ rho @ K.conj().T for K in kraus)
        r = T.bloch_vector(out)
        pts.append((r[0], r[2]))
    return np.array(pts)


def panel(curves, xshift, title):
    s = 1.7
    out = ["\\begin{scope}[shift={(%.2f,0)}]" % xshift,
           "\\draw[->,gray] (%.2f,0) -- (%.2f,0) node[right,font=\\footnotesize] {$x$};" % (-1.3 * s, 1.3 * s),
           "\\draw[->,gray] (0,%.2f) -- (0,%.2f) node[above,font=\\footnotesize] {$z$};" % (-1.3 * s, 1.3 * s),
           "\\draw[thick] (0,0) circle (%.3f);" % s]
    for i, (pts, lab) in enumerate(curves):
        col = T.PALETTE[i % len(T.PALETTE)][0]
        path = " -- ".join("(%.3f,%.3f)" % (s * p[0], s * p[1]) for p in pts)
        out.append("\\draw[thick,%s] %s -- cycle;" % (col, path))
        out.append("\\node[%s,font=\\scriptsize,anchor=west] at (%.2f,%.2f) {%s};"
                   % (col, 1.15 * s, 1.1 * s - 0.35 * i, lab))
    out.append("\\node[font=\\small] at (0,%.2f) {%s};" % (1.6 * s, title))
    out.append("\\end{scope}")
    return "\n".join(out)


if __name__ == "__main__":
    for g in (0.3, 0.6, 0.9):
        assert is_cptp(amplitude_damping(g)) and is_cptp(dephasing(g))
    ad = [(bloch_image(amplitude_damping(g)), "$\\gamma=%.1f$" % g) for g in (0.3, 0.6, 0.9)]
    dp = [(bloch_image(dephasing(p)), "$p=%.1f$" % p) for p in (0.15, 0.3, 0.45)]
    body = panel(ad, 0.0, "amplitude damping") + "\n" + panel(dp, 7.2, "dephasing")
    T.write_tikz("ch08_channels", body)
    print("wrote figures/ch08_channels.tex")
