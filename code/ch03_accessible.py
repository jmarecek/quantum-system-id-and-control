"""ch03_accessible.py -- the accessible set, step by step.

Three qubits, measured observable Z_1 (Pauli string ZII).  The iteration
M_0 = {ZII},  M_j = [M_{j-1}, F] u M_{j-1}
adds every Pauli string obtained by commuting an element of M_{j-1} with a
generator present in the Hamiltonian (two Pauli strings have a non-zero
commutator iff they anticommute, and it is then proportional to their product).
  transverse-field Ising chain  X1, X2, X3, Z1Z2, Z2Z3  -> saturates at  6 of 63;
  XY chain  X1X2, Y1Y2, X2X3, Y2Y3, Z1                  -> saturates at 15 of 63.
Only these coherence-vector components, and the parameters coupling to them,
influence the measured output.  Writes figures/ch03_accessible.tex.
"""
import numpy as np
import needham as N

MODELS = [("Ising: $X_1,X_2,X_3,Z_1Z_2,Z_2Z_3$", ["XII", "IXI", "IIX", "ZZI", "IZZ"], "ndBlue"),
          ("XY chain: $X_1X_2,Y_1Y_2,X_2X_3,Y_2Y_3,Z_1$", ["XXI", "YYI", "IXX", "IYY", "ZII"], "ndRed")]


def pmul(a, b):
    if a == "I":
        return b, False
    if b == "I":
        return a, False
    if a == b:
        return "I", False
    return ({"X", "Y", "Z"} - {a, b}).pop(), True


def comm(p, q):
    """Pauli string proportional to [p, q], or None if they commute."""
    prod, anti = zip(*(pmul(a, b) for a, b in zip(p, q)))
    return "".join(prod) if sum(anti) % 2 else None


def accessible(obs, gens):
    G, sizes = {obs}, [1]
    while True:
        new = set(G)
        for g in G:
            for h in gens:
                c = comm(g, h)
                if c and c != "III":
                    new.add(c)
        if new == G:
            return sizes, G
        G = new
        sizes.append(len(G))


def check_with_matrices(G, gens):
    """Closure check with actual matrices: [G, gens] stays in span(G)."""
    P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    M = lambda s: np.kron(np.kron(P[s[0]], P[s[1]]), P[s[2]])
    basis = np.array([M(g).ravel() for g in G]).T
    for g in G:
        for h in gens:
            c = (M(g) @ M(h) - M(h) @ M(g)).ravel()
            coef = np.linalg.lstsq(basis, c, rcond=None)[0]
            assert np.allclose(basis @ coef, c)


def main():
    b = []
    W, H = 0.55, 3.6 / 63
    x0 = 0.0
    for title, gens, col in MODELS:
        sizes, G = accessible("ZII", gens)
        check_with_matrices(G, gens)
        for k, d in enumerate(sizes):
            xa = x0 + k * (W + 0.18)
            b.append("\\fill[%s, fill opacity=0.75] (%.2f,0) rectangle (%.2f,%.3f);" % (col, xa, xa + W, H * d))
            b.append("\\draw[ink, line width=0.5pt] (%.2f,0) rectangle (%.2f,%.3f);" % (xa, xa + W, H * d))
            b.append("\\node[font=\\scriptsize, above] at (%.2f,%.3f) {%d};" % (xa + W / 2, H * d, d))
            b.append("\\node[font=\\scriptsize, below] at (%.2f,0) {%d};" % (xa + W / 2, k))
        xe = x0 + len(sizes) * (W + 0.18)
        b.append("\\draw[faint] (%.2f,0) -- (%.2f,0);" % (x0 - 0.2, xe))
        b.append("\\draw[dashedink] (%.2f,%.3f) -- (%.2f,%.3f);" % (x0 - 0.2, H * 63, xe, H * 63))
        b.append("\\node[note, anchor=south] at (%.2f,%.3f) {all $N^2-1=63$ components};" % ((x0 + xe) / 2, H * 63 + 0.05))
        b.append("\\node[font=\\scriptsize, text=%s] at (%.2f,-0.75) {%s};" % (col, (x0 + xe) / 2, title))
        b.append("\\node[note] at (%.2f,%.3f) {$\\tilde n=%d$ of $63$};" % ((x0 + xe) / 2, H * 63 * 0.55, sizes[-1]))
        x0 = xe + 1.6
    b.append("\\node[note, anchor=west] at (%.2f,1.8) {measured: $Z_1$;\\\\ step $j$: add the commutators\\\\ with the generators;\\\\ stop when nothing is new};" % (x0 - 0.8))
    b.append("\\node[font=\\scriptsize] at (%.2f,-1.25) {step $j$ of the iteration};" % ((x0 - 1.6) / 2))
    N.write("ch03_accessible", b)
    print("written figures/ch03_accessible.tex")


if __name__ == "__main__":
    main()
