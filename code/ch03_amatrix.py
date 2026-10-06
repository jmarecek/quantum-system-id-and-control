"""ch03_amatrix.py -- the coherence-vector system matrix of two qubits.

For H = -0.5 Z1 - 0.4 Z2 + 0.3 X1X2 (|0> the ground state) and jump operators sqrt(0.3) sigma_-^{(1)},
sqrt(0.2) Z2, the coherence vector x_i = Tr[F_i rho] in the normalised Pauli
basis obeys dx/dt = (A^(l) + A^(d)) x + b with
    A_ij = Tr[F_i L(F_j)],   b_i = Tr[F_i L(I/N)],
computed here directly from the generator, split into its unitary part A^(l)
(antisymmetric) and dissipative part A^(d), with the affine term b.
Writes figures/ch03_amatrix.tex.
"""
import numpy as np
import itertools
import tikzexport as T
import needham as N

I, X, Y, Z = T.pauli()
PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}
NAMES = ["".join(p) for p in itertools.product("IXYZ", repeat=2)][1:]
F = [np.kron(PAULI[a], PAULI[b]) / 2 for a, b in NAMES]       # Tr(F_i F_j) = delta_ij
SM = np.array([[0, 1], [0, 0]], complex)


def generator(H, jumps):
    def L(r):
        out = -1j * (H @ r - r @ H)
        for J in jumps:
            out += J @ r @ J.conj().T - 0.5 * (J.conj().T @ J @ r + r @ J.conj().T @ J)
        return out
    return L


def matrices(L):
    A = np.array([[np.trace(Fi @ L(Fj)).real for Fj in F] for Fi in F])
    b = np.array([np.trace(Fi @ L(np.eye(4) / 4)).real for Fi in F])
    return A, b


def heat(b, M, x0, y0, c, title, vmax, labels=True):
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            v = M[i, j] / vmax
            if abs(v) < 1e-12:
                col = "white"
            else:
                col = "ndBlue!%d!white" % int(15 + 85 * min(1, abs(v))) if v > 0 else "ndRed!%d!white" % int(15 + 85 * min(1, abs(v)))
            b.append("\\fill[%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, x0 + j * c, y0 - i * c, x0 + (j + 1) * c, y0 - (i + 1) * c))
    b.append("\\draw[ink, line width=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0, y0, x0 + m * c, y0 - n * c))
    if labels:
        for i, nm in enumerate(NAMES):
            b.append("\\node[font=\\tiny, left, text=ndInk!80] at (%.3f,%.3f) {%s};" % (x0, y0 - (i + 0.5) * c, nm))
    b.append("\\node[font=\\small, above] at (%.3f,%.3f) {%s};" % (x0 + m * c / 2, y0 + 0.05, title))


def main():
    H = -0.5 * np.kron(Z, I) - 0.4 * np.kron(I, Z) + 0.3 * np.kron(X, X)
    jumps = [np.sqrt(0.3) * np.kron(SM, I), np.sqrt(0.2) * np.kron(I, Z)]
    Al, bl = matrices(generator(H, []))
    A, b_ = matrices(generator(H, jumps))
    Ad = A - Al
    assert np.allclose(Al, -Al.T) and np.allclose(bl, 0)
    # check: the coherence-vector system reproduces the evolution of a state
    rho = np.diag([0.1, 0.2, 0.3, 0.4]).astype(complex) + 0.05 * np.kron(X, Y)
    x = np.array([np.trace(Fi @ rho).real for Fi in F])
    Lr = generator(H, jumps)(rho)
    assert np.allclose(A @ x + b_, [np.trace(Fi @ Lr).real for Fi in F])
    vmax = max(abs(Al).max(), abs(Ad).max(), abs(b_).max())
    c = 0.32
    out = []
    heat(out, Al, 0.0, 0.0, c, "$\\Amat^{(l)}$: unitary part", vmax)
    heat(out, Ad, 6.0, 0.0, c, "$\\Amat^{(d)}$: dissipative part", vmax)
    heat(out, b_[:, None], 11.6, 0.0, c, "$\\bvec$", vmax, labels=False)
    out.append("\\node[note, anchor=north] at (2.4,%.2f) {antisymmetric: a rotation\\\\ of the coherence vector};" % (-15 * c - 0.15))
    out.append("\\node[note, anchor=north] at (8.4,%.2f) {no symmetry; with $\\bvec$, it drives the\\\\ state to the unique steady state};" % (-15 * c - 0.15))
    out.append("\\node[note, anchor=west] at (12.3,-3.2) {$\\bvec$: pushes towards\\\\ the steady state};")
    out.append("\\node[note, anchor=west] at (12.3,-1.0) {\\textcolor{ndBlue}{blue}: positive\\\\ \\textcolor{ndRed}{red}: negative\\\\ white: zero};")
    # Hurwitz: every eigenvalue of A has negative real part, so x(t) -> -A^{-1} b
    assert np.linalg.eigvals(A).real.max() < -1e-9
    N.write("ch03_amatrix", out)
    print("written figures/ch03_amatrix.tex; nonzeros A(l) %d, A(d) %d, b %d"
          % ((abs(Al) > 1e-12).sum(), (abs(Ad) > 1e-12).sum(), (abs(b_) > 1e-12).sum()))


if __name__ == "__main__":
    main()
