"""ch04_distinct.py -- controllability, observability and identifiability are different questions.

Six experiments, each property computed rather than asserted:
  controllable:  rank of [B, AB, A^2 B, A^3 B] for the embedded qubit (4) or
                 two qubits (16), B the matrix of prepared states;
  observable:    rank of the observability matrix of the chosen readout;
  identifiable:  rank of the Jacobian of the sampled outputs with respect to
                 (omega, gamma_1, gamma_phi) (local structural identifiability),
                 or, for the black box, the dimension of the similarity gauge,
                 and for two qubits the rank of the reconstruction matrix R_2.
Writes figures/ch04_distinct.tex.
"""
import itertools
import numpy as np
from scipy.linalg import expm
import needham as N
import ch04_observability as M

I2, X, Y, Z = M.I2, M.X, M.Y, M.Z
SM = np.array([[0, 1], [0, 0]], complex)                  # sigma_- = |0><1|
B2 = M.pauli_basis(1)
THETA = np.array([1.0, 0.2, 0.1])                         # omega, gamma_1, gamma_phi
TS = np.linspace(0, 12, 25)
BLOCH = {"+": (1, 0, 0), "+i": (0, 1, 0), "1": (0, 0, -1), "0": (0, 0, 1)}
ANY = ["+", "+i", "1", "0"]


def embedded(w, g1, gp):
    A, beta = M.coherence_representation(M.lindbladian(-w / 2 * Z, [np.sqrt(g1) * SM, np.sqrt(gp / 2) * Z]), B2)
    E = np.zeros((4, 4)); E[:3, :3] = A; E[:3, 3] = beta
    return E


def prep_matrix(preps):
    return np.array([np.append(np.array(BLOCH[p]) / np.sqrt(2), 1.0) for p in preps]).T


def readout(obs):
    return np.hstack([M.readout_matrix(obs, B2), np.zeros((len(obs), 1))])


def outputs(theta, preps, obs):
    E, Bm, C = embedded(*theta), prep_matrix(preps), readout(obs)
    return np.concatenate([(C @ expm(E * t) @ Bm).ravel() for t in TS])


def qubit_row(preps, obs):
    E = embedded(*THETA)
    Bm, C = prep_matrix(preps), readout(obs)
    ctrl = np.linalg.matrix_rank(np.hstack([np.linalg.matrix_power(E, k) @ Bm for k in range(4)]), 1e-9)
    obsv = np.linalg.matrix_rank(M.observability_matrix(E, C), 1e-9)
    J = np.array([(outputs(THETA + d, preps, obs) - outputs(THETA - d, preps, obs)) / 2e-6 for d in 1e-6 * np.eye(3)]).T
    s = np.linalg.svd(J, compute_uv=False)
    ident = int(np.sum(s > 1e-6 * s[0]))
    _, _, vt = np.linalg.svd(J)
    return ctrl, obsv, ident, vt[:ident]


def black_box_gauge():
    """Markov parameters C A^k B with A, B, C all unknown: the rank deficiency of the Jacobian."""
    E = expm(embedded(*THETA) * 0.7)
    Bm, C = prep_matrix(ANY), readout([X, Y, Z])
    def markov(v):
        A, Bv, Cv = v[:16].reshape(4, 4), v[16:32].reshape(4, 4), v[32:44].reshape(3, 4)
        return np.concatenate([(Cv @ np.linalg.matrix_power(A, k) @ Bv).ravel() for k in range(9)])
    v0 = np.concatenate([E.ravel(), Bm.ravel(), C.ravel()])
    J = np.array([(markov(v0 + e) - markov(v0 - e)) / 2e-6 for e in 1e-6 * np.eye(44)]).T
    s = np.linalg.svd(J, compute_uv=False)
    return 44 - int(np.sum(s > 1e-7 * s[0]))


def two_qubit_R2_rank():
    b4 = M.pauli_basis(2)
    def Ad(l, m):
        D = lambda r: b4[l] @ r @ b4[m] - 0.5 * (b4[m] @ b4[l] @ r + r @ b4[m] @ b4[l])
        return np.array([[np.trace(Fj @ D(Fk)) for Fk in b4] for Fj in b4])
    R2 = np.array([Ad(l, m).reshape(-1) for l in range(15) for m in range(15)]).T
    return np.linalg.matrix_rank(R2, 1e-9)


def main():
    r1 = qubit_row(["1"], [X, Y, Z])
    r2 = qubit_row(ANY, [Z])
    r3 = qubit_row(ANY, [X])
    r4 = qubit_row(ANY, [X, Z])
    assert r1[:3] == (2, 4, 1) and np.allclose(np.abs(r1[3]), [[0, 1, 0]], atol=1e-3)            # gamma_1 only
    assert r2[:3] == (4, 2, 1) and np.allclose(np.abs(r2[3]), [[0, 1, 0]], atol=1e-3)            # gamma_1 only
    comb = np.array([0, 1, 2]) / np.sqrt(5)                                                        # gamma_1/2 + gamma_phi
    assert r3[:3] == (4, 2, 2) and np.allclose(np.linalg.norm(r3[3] @ comb), 1, atol=1e-3)       # omega and the combination
    assert r4[:3] == (4, 4, 3)
    gauge = black_box_gauge()
    rankR2 = two_qubit_R2_rank()
    assert gauge == 16 and rankR2 == 210
    print("rows:", r1[:3], r2[:3], r3[:3], r4[:3], "gauge", gauge, "rank R2", rankR2)

    yes, no = "$\\checkmark$", "$\\times$"
    def mark(ok, detail=""):
        col = "ndGreen" if ok else "ndRed"
        return "\\textcolor{%s}{\\large %s}%s" % (col, yes if ok else no, ("\\ {\\scriptsize %s}" % detail) if detail else "")
    rows = [
        ("prepare only $\\ket1$; read $\\sigma_x,\\sigma_y,\\sigma_z$", mark(False, "rank $2$"), mark(True), mark(False), "$\\gamma_1$ only: $x,y$ are never excited"),
        ("prepare any state; read $\\sigma_z$", mark(True), mark(False, "rank $2$"), mark(False), "$\\gamma_1$ only: $x,y$ are never seen"),
        ("prepare any state; read $\\sigma_x$", mark(True), mark(False, "rank $2$"), mark(False), "$\\omega$ and $\\gamma_1/2+\\gamma_\\phi$ only"),
        ("prepare any state; read $\\sigma_x,\\sigma_z$", mark(True), mark(True), mark(True), "$\\omega,\\gamma_1,\\gamma_\\phi$"),
        ("preparation and readout both unknown", mark(True), mark(True), mark(False), "$\\mathbf A$ only up to similarity:\\\\ a $%d$-dimensional gauge" % gauge),
        ("two qubits, all $15$ Paulis, any $\\gamma$", mark(True), mark(True), mark(False), "$\\rank\\mathbf R_2=%d<225$;\\\\ on the physical support: \\textcolor{ndGreen}{$\\checkmark$}" % rankR2),
    ]
    xs = [0.0, 6.7, 8.7, 10.7, 12.1]
    b = []
    heads = ["experiment", "controllable", "observable", "identifiable", "what the data determine"]
    for x, h in zip(xs, heads):
        anchor = "west" if x in (0.0, 12.1) else "center"
        b.append("\\node[anchor=%s, font=\\small\\bfseries] at (%.2f,0.6) {%s};" % (anchor, x if anchor == "west" else x, h))
    b.append("\\draw[ink] (-0.1,0.25) -- (17.3,0.25);")
    for i, row in enumerate(rows):
        y = -0.45 - 0.85 * i
        if i % 2 == 1:
            b.append("\\fill[ndSky, opacity=0.35] (-0.1,%.2f) rectangle (17.3,%.2f);" % (y + 0.42, y - 0.43))
        b.append("\\node[anchor=west, font=\\small] at (0.0,%.2f) {%s};" % (y, row[0]))
        for x, cell in zip(xs[1:4], row[1:4]):
            b.append("\\node[font=\\small] at (%.2f,%.2f) {%s};" % (x, y, cell))
        b.append("\\node[anchor=west, font=\\small, align=left] at (12.1,%.2f) {%s};" % (y, row[4]))
    b.append("\\draw[ink] (-0.1,%.2f) -- (17.3,%.2f);" % (-0.45 - 0.85 * 5 - 0.43, -0.45 - 0.85 * 5 - 0.43))
    N.write("ch04_distinct", b)
    print("written figures/ch04_distinct.tex")


if __name__ == "__main__":
    main()
