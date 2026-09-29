"""ch02_closure.py -- the dynamical Lie algebra, bracket by bracket.

Starting from the generators, repeatedly add all commutators and record the
dimension of the real span after brackets of depth <= k:
  qubit:      X = -i sx/2, Z = -i sz/2                    -> su(2), dim 3;
  two qubits: i zz, i x1, i y1, i x2, i y2 (Exercise)      -> su(4), dim 15.
Writes figures/ch02_closure.tex.
"""
import numpy as np
import needham as N

I = np.eye(2)
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.diag([1.0 + 0j, -1.0])
kron = np.kron


def real_vec(m):
    return np.concatenate([m.real.ravel(), m.imag.ravel()])


def dims_by_depth(gens, depth=6):
    """dim of span{brackets of depth <= k}, k = 0..depth."""
    basis, level, dims = [], list(gens), []
    def add(ms):
        for m in ms:
            cand = basis + [m]
            if np.linalg.matrix_rank(np.array([real_vec(c) for c in cand]), 1e-9) > len(basis):
                basis.append(m)
    add(level)
    dims.append(len(basis))
    for _ in range(depth):
        level = [a @ b - b @ a for a in list(basis) for b in gens]
        add(level)
        dims.append(len(basis))
    return dims


def bars(b, x0, dims, full, col, title):
    W, H = 0.55, 2.6 / full
    for k, d in enumerate(dims):
        xa = x0 + k * (W + 0.2)
        b.append("\\fill[%s, fill opacity=0.7] (%.2f,0) rectangle (%.2f,%.3f);" % (col, xa, xa + W, H * d))
        b.append("\\draw[ink, line width=0.5pt] (%.2f,0) rectangle (%.2f,%.3f);" % (xa, xa + W, H * d))
        b.append("\\node[font=\\scriptsize, above] at (%.2f,%.3f) {%d};" % (xa + W / 2, H * d, d))
        b.append("\\node[font=\\scriptsize, below] at (%.2f,0) {%d};" % (xa + W / 2, k))
    xe = x0 + len(dims) * (W + 0.2)
    b.append("\\draw[faint] (%.2f,0) -- (%.2f,0);" % (x0 - 0.2, xe))
    b.append("\\draw[dashedink] (%.2f,%.3f) -- (%.2f,%.3f) node[right, font=\\scriptsize] {%s};"
             % (x0 - 0.2, H * full, xe, H * full, title))
    b.append("\\node[font=\\scriptsize] at (%.2f,-0.65) {bracket depth $k$};" % ((x0 + xe) / 2))


def main():
    q = dims_by_depth([-0.5j * sx, -0.5j * sz], depth=3)
    t = dims_by_depth([1j * kron(sz, sz), 1j * kron(sx, I), 1j * kron(sy, I), 1j * kron(I, sx), 1j * kron(I, sy)], depth=3)
    assert q[-1] == 3 and t[-1] == 15
    b = []
    bars(b, 0.0, q, 3, "ndBlue", "$\\dim\\liesu[2]=3$")
    bars(b, 6.3, t, 15, "ndRed", "$\\dim\\liesu[4]=15$")
    b.append("\\node[note, anchor=south] at (1.3,3.15) {qubit: $X,Z$\\\\ one bracket $\\comm ZX=Y$ completes $\\liesu[2]$};")
    b.append("\\node[note, anchor=south] at (8.5,3.15) {two qubits: $\\sigz\\otimes\\sigz$ and local $\\sigx,\\sigy$\\\\"
             "the span grows level by level until it fills $\\liesu[4]$};")
    b.append("\\node[note, anchor=west] at (11.6,1.2) {once a level adds\\\\ nothing new, the span\\\\ is closed: it is the\\\\ dynamical Lie algebra};")
    N.write("ch02_closure", b)
    print("written figures/ch02_closure.tex; qubit", q, "two qubits", t)


if __name__ == "__main__":
    main()
