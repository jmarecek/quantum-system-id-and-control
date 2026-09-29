"""ch02_matrices.py -- what a matrix does to the unit circle.

Real 2x2 slices of the three classes of matrices:
  (a) positive definite Hermitian: the circle becomes an ellipse whose axes are
      the orthonormal eigenvectors, stretched by the (positive) eigenvalues;
  (b) indefinite Hermitian: orthogonal axes again, but the negative eigenvalue
      maps its unit eigenvector v to a multiple of -v: that axis is flipped;
  (c) unitary (a rotation): the circle is mapped onto itself, lengths and
      angles are kept.
A marked point x and its image Ax show how points move.
Writes figures/ch02_matrices.tex.
"""
import numpy as np
import needham as N

S = 1.25                                         # page units per unit length
CASES = [
    ("(a) positive definite, $A\\succ0$", np.array([[1.45, 0.45], [0.45, 0.75]])),
    ("(b) Hermitian, indefinite", np.array([[0.9, 0.9], [0.9, -0.3]])),
    ("(c) unitary: a rotation", np.array([[np.cos(1.0), -np.sin(1.0)], [np.sin(1.0), np.cos(1.0)]])),
]
X = np.array([np.cos(2.3), np.sin(2.3)])         # the marked point


def panel(A, title, dx):
    b = []
    o = np.array([dx, 0.0])
    P = lambda v: o + S * np.asarray(v)
    t = np.linspace(0, 2 * np.pi, 200)
    circ = np.stack([np.cos(t), np.sin(t)], axis=1)
    b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (N.pt(P((-1.9, 0))), N.pt(P((1.9, 0)))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (N.pt(P((0, -1.7))), N.pt(P((0, 1.8)))))
    b.append(N.polyline([P(c) for c in circ], "dashedink"))
    img = circ @ A.T
    b.append("\\fill[ndSky, fill opacity=0.55] %s -- cycle;" % " -- ".join(N.pt(P(q)) for q in img))
    b.append(N.polyline([P(q) for q in img], "line width=1.2pt, ndBlue"))
    sym = np.allclose(A, A.T)
    if sym:
        w, V = np.linalg.eigh(A)
        assert np.allclose(V.T @ V, np.eye(2))              # orthonormal eigenvectors
        for lam, v in zip(w, V.T):
            v = v if v[1] >= 0 else -v                         # a definite orientation
            col = "ndRed" if lam < 0 else "ndGreen"
            b.append("\\draw[dashedink] %s -- %s;" % (N.pt(P(-1.6 * v)), N.pt(P(1.6 * v))))
            b.append("\\draw[->, >=stealth, line width=0.7pt, ndInk] %s -- %s;" % (N.pt(P((0, 0))), N.pt(P(v))))
            b.append("\\draw[vec, %s] %s -- %s;" % (col, N.pt(P((0, 0))), N.pt(P(lam * v))))
            b.append("\\node[font=\\scriptsize, text=%s, fill=white, inner sep=1pt, fill opacity=0.8, text opacity=1] at %s {$%.1f$};"
                     % (col, N.pt(P(lam * v + 0.28 * np.sign(lam if lam else 1) * v)), lam))
    else:
        assert np.allclose(A.T @ A, np.eye(2))              # orthogonal = real unitary
    # the marked point and its image
    b.append("\\draw[->, >=stealth, line width=0.7pt, ndInk!70] %s to[bend left=15] %s;"
             % (N.pt(P(X)), N.pt(P(A @ X))))
    b.append("\\node[dot] at %s {};" % N.pt(P(X)))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % N.pt(P(A @ X)))
    b.append("\\node[font=\\scriptsize, above left] at %s {$x$};" % N.pt(P(X)))
    b.append("\\node[font=\\scriptsize, text=ndBlue, right] at %s {$Ax$};" % N.pt(P(A @ X)))
    b.append("\\node[font=\\small] at %s {%s};" % (N.pt(P((0, -2.15))), title))
    return b


def main():
    b = []
    for k, (title, A) in enumerate(CASES):
        b += panel(A, title, 5.6 * k)
    b.append("\\node[note] at (0,2.75) {unit eigenvectors $v$ (black) go to $\\lambda v$:\\\\ orthogonal axes, stretched by $\\lambda$};")
    b.append("\\node[note] at (5.6,2.75) {a negative eigenvalue \\emph{flips}\\\\ its axis: $v\\mapsto\\lambda v$ points back};")
    b.append("\\node[note] at (11.2,2.75) {lengths and angles kept:\\\\ the circle goes to itself};")
    N.write("ch02_matrices", b)
    print("written figures/ch02_matrices.tex; eigenvalues", [np.round(np.linalg.eigvalsh(A), 3).tolist() for _, A in CASES[:2]])


if __name__ == "__main__":
    main()
