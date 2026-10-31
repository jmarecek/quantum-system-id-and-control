"""ch07_tn_entropy.py -- area law versus volume law for a chain of qubits.

The controlled Ising chain of Balada Gaggioli and Marecek,
    H = J sum_{i<N} Z_i Z_{i+1} + u sum_i X_i,      J = 1, open boundary,
with a constant control u.  For N = 14 qubits we compute
  (a) the entanglement entropy S_k (in bits) across every cut k = 1..N-1 of the
      ground states at u = 0.5 (ordered: on a finite chain the ground state is
      a cat state, which carries about one bit across every cut), u = 1
      (critical: S grows slowly, like a logarithm, towards the middle) and
      u = 2 (paramagnetic: small), and of a Haar-random state, whose entropy
      is the tent min(k, N-k) less about 1/(2 ln 2) = 0.72 bits at the middle
      (Page 1993);
  (b) the Schmidt spectrum across the middle cut on a logarithmic scale, and the
      bond dimension D needed to keep the discarded weight below 1e-8.
Low-energy states of gapped local Hamiltonians obey an area law (Hastings
2007): a small bond dimension suffices.  A typical state needs D = 2^(N/2).

This module also holds the matrix-product helpers used by ch07_tn_cut.py and
ch07_tn_manifold.py: the unfolding of a state across a cut, its Schmidt
coefficients, the entropy, the TT-SVD (successive SVDs with a largest bond
dimension), and the sparse Hamiltonian of the chain.
Writes figures/ch07_tn_entropy.tex.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

SX = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
SZ = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))          # |0> is spin up, Z|0> = |0>


def site_op(op, i, n):
    """op acting on qubit i (0-based) of n, as a sparse matrix."""
    return sp.kron(sp.kron(sp.identity(2 ** i, format="csr"), op), sp.identity(2 ** (n - i - 1), format="csr"), format="csr")


def ising(n, J=1.0, u=1.0):
    """H = J sum Z_i Z_{i+1} + u sum X_i on n qubits, sparse."""
    H = sp.csr_matrix((2 ** n, 2 ** n))
    for i in range(n - 1):
        H = H + J * site_op(SZ, i, n) @ site_op(SZ, i + 1, n)
    for i in range(n):
        H = H + u * site_op(SX, i, n)
    return H.tocsr()


def unfold(psi, k, n, d=2):
    """The state as a d^k x d^(n-k) matrix: sites 1..k against sites k+1..n."""
    return np.asarray(psi).reshape(d ** k, d ** (n - k))


def schmidt(psi, k, n):
    """Schmidt coefficients across the cut after site k (singular values of the unfolding)."""
    return np.linalg.svd(unfold(psi, k, n), compute_uv=False)


def entropy_bits(s):
    """Entanglement entropy -sum p log2 p of the Schmidt weights p = s^2."""
    p = np.asarray(s) ** 2
    p = p[p > 1e-300] / p.sum()
    return float(-(p * np.log2(p)).sum())


def tt_svd(psi, n, dmax=None, d=2):
    """Matrix product state of psi by successive SVDs, keeping at most dmax singular values per cut.

    Returns cores A[k] of shape (D_{k-1}, d, D_k), with D_0 = D_n = 1.
    """
    cores, rest, Dl = [], np.asarray(psi, dtype=complex).reshape(1, -1), 1
    for k in range(n - 1):
        M = rest.reshape(Dl * d, -1)
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        keep = int(np.sum(s > 1e-14 * s[0]))
        if dmax is not None:
            keep = min(keep, dmax)
        cores.append(U[:, :keep].reshape(Dl, d, keep))
        rest = s[:keep, None] * Vh[:keep]
        Dl = keep
    cores.append(rest.reshape(Dl, d, 1))
    return cores


def mps_vector(cores):
    """Contract a matrix product state into a vector."""
    v = cores[0]
    for A in cores[1:]:
        v = np.tensordot(v, A, axes=([v.ndim - 1], [0]))
    return v.reshape(-1)


def bonds(cores):
    return [A.shape[2] for A in cores[:-1]]


def random_mps(n, D, rng, d=2):
    """A random matrix product state with the bond dimensions D[0..n-2]."""
    dims = [1] + list(D) + [1]
    return [rng.normal(size=(dims[k], d, dims[k + 1])) + 1j * rng.normal(size=(dims[k], d, dims[k + 1]))
            for k in range(n)]


def ground_state(n, u, J=1.0):
    vals, vecs = eigsh(ising(n, J, u), k=2, which="SA", tol=1e-12, maxiter=20000)
    order = np.argsort(vals)
    return vals[order], vecs[:, order[0]]


def bond_for(weights, tol=1e-8):
    """Least D with discarded weight sum_{a > D} p_a below tol (weights in decreasing order)."""
    tail = np.concatenate([np.cumsum(weights[::-1])[::-1][1:], [0.0]])
    return int(np.argmax(tail < tol) + 1)


def main():
    n = 14
    rng = np.random.default_rng(7)
    states, gaps = {}, {}
    for u in (0.5, 1.0, 2.0):
        vals, psi = ground_state(n, u)
        states[u], gaps[u] = psi, vals[1] - vals[0]
    r = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    states["rand"] = r / np.linalg.norm(r)
    S = {key: np.array([entropy_bits(schmidt(psi, k, n)) for k in range(1, n)]) for key, psi in states.items()}
    lam = {key: schmidt(psi, n // 2, n) ** 2 for key, psi in states.items()}
    need = {key: bond_for(lam[key]) for key in states}
    for key in states:
        print("%-5s S(middle) = %.3f bits, S(1) = %.3f, required D (discarded < 1e-8) = %d, largest weight %.4f"
              % (key, S[key][n // 2 - 1], S[key][0], need[key], lam[key][0]))
    print("gaps: %s" % {u: "%.2e" % g for u, g in gaps.items()})
    # the checks
    for key in states:
        assert abs(lam[key].sum() - 1) < 1e-10
    page = n // 2 - 1 / (2 * np.log(2))                        # Page's value at the middle, large-N form
    assert abs(S["rand"][n // 2 - 1] - page) < 0.05
    assert np.all(np.abs(S["rand"] - np.minimum(np.arange(1, n), n - np.arange(1, n))) < 0.75)   # the tent, less a little
    assert abs(S[0.5][n // 2 - 1] - 1.0) < 0.1                  # the cat state: about one bit at every cut
    assert np.all(np.abs(S[0.5][2:-2] - 1.0) < 0.1) and S[0.5].min() > 0.8
    assert S[2.0].max() < 0.3                                   # paramagnet: little entanglement
    assert S[1.0][n // 2 - 1] > S[1.0][0] and S[1.0].max() < 1.0   # critical: grows slowly towards the middle
    assert need["rand"] == 2 ** (n // 2) and max(need[u] for u in (0.5, 1.0, 2.0)) <= 32

    b = []
    styles = {"rand": "ndInk", 0.5: "ndGold", 1.0: "ndRed", 2.0: "ndBlue"}
    # ---- (a) entropies across the cuts
    P = N.Plot((0.5, n - 0.5), (0.0, 7.4), 6.2, 5.0, origin=(0.0, 0.0))
    b += P.axes("", "$S_k$ (bits)", xticks=[(k, "$%d$" % k) for k in (1, 4, 7, 10, 13)],
                yticks=[(1, "$1$"), (3, "$3$"), (5, "$5$"), (7, "$7$")])
    b.append("\\node[font=\\small, anchor=north] at %s {cut $k$};" % N.pt(P.P(n / 2, 0) + np.array([0, -0.35])))
    ks = np.arange(1, n)
    b.append(P.curve(ks, np.minimum(ks, n - ks), "dashedink"))
    for key in ("rand", 0.5, 1.0, 2.0):
        b.append(P.curve(ks, S[key], "line width=1.2pt, %s" % styles[key]))
        for k_, s_ in zip(ks, S[key]):
            b.append("\\node[circle, fill=%s, inner sep=0pt, minimum size=2.6pt] at %s {};" % (styles[key], P.pt(k_, s_)))
    b.append("\\node[note, anchor=west, align=left] (vol) at %s {random state:\\\\ a volume law};" % P.pt(0.7, 6.4))
    b.append("\\draw[pointer] (vol.east) to[bend left=20] %s;" % N.pt(P.P(6, S["rand"][5]) + np.array([-0.08, 0.06])))
    b.append("\\node[note, anchor=west] (tent) at %s {$\\min(k,N-k)$};" % P.pt(10.3, 5.6))
    b.append("\\draw[pointer] (tent.south) to[bend left=10] %s;" % N.pt(P.P(9.6, 4.4) + np.array([0.05, 0.02])))
    labels = [(0.5, "$u=0.5$, ordered:\\\\ a cat state, one bit", 2.35), (1.0, "$u=1$, critical", 1.3), (2.0, "$u=2$, paramagnet", 0.55)]
    for key, text, y in labels:
        b.append("\\node[note, anchor=west, align=left, text=%s!85!ndInk] (l%s) at %s {%s};" % (styles[key], str(key).replace(".", ""), P.pt(13.75, y), text))
        b.append("\\draw[pointer] (l%s.west) -- %s;" % (str(key).replace(".", ""), N.pt(P.P(13, S[key][-1]) + np.array([0.08, 0.0]))))
    b.append("\\node[font=\\small] at %s {(a) the entropy across each cut, $N=%d$};" % (N.pt(P.P(n / 2, 0) + np.array([0.6, -1.15])), n))
    # ---- (b) Schmidt spectra across the middle cut, against log2 alpha
    m = 2 ** (n // 2)
    Q = N.Plot((-0.2, np.log2(m) + 0.2), (-16.5, 0.5), 6.2, 5.0, origin=(10.6, 0.0))
    b += Q.axes("", "$\\log_{10}p_\\alpha$", xticks=[(np.log2(a), "$%d$" % a) for a in (1, 2, 4, 8, 16, 32, 64, 128)],
                yticks=[(0, "$0$"), (-4, "$-4$"), (-8, "$-8$"), (-12, "$-12$"), (-16, "$-16$")])
    b.append("\\node[font=\\small, anchor=north] at %s {index $\\alpha$ (logarithmic)};" % N.pt(Q.P(np.log2(m) / 2, -16.5) + np.array([0, -0.35])))
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (Q.pt(-0.2, -8), Q.pt(np.log2(m) + 0.2, -8)))
    for key in ("rand", 0.5, 1.0, 2.0):
        p = np.log10(np.maximum(lam[key], 1e-17))
        keep = p > -16.5
        a = np.log2(np.arange(1, len(p) + 1)[keep])
        b.append(Q.curve(a, p[keep], "line width=1.1pt, %s" % styles[key]))
    for key in (0.5, 1.0, 2.0):
        D_ = need[key]
        b.append("\\node[circle, fill=%s, inner sep=0pt, minimum size=4pt] at %s {};" % (styles[key], Q.pt(np.log2(D_), np.log10(lam[key][D_ - 1]))))
    b.append("\\node[note, anchor=south] at %s {random: all $%d$ weights\\\\ near $2^{-7}$ are needed};" % (Q.pt(4.4, -1.3), m))
    b.append("\\node[note, anchor=west, align=left] (gs) at %s {ground states:\\\\ the dots, at\\\\ $D=%d$, $%d$ and $%d$,\\\\ leave out less\\\\ than $10^{-8}$};"
             % (Q.pt(4.55, -12.0), need[0.5], need[1.0], need[2.0]))
    b.append("\\draw[pointer] (gs.west) to[bend right=15] %s;" % N.pt(Q.P(np.log2(need[1.0]), np.log10(lam[1.0][need[1.0] - 1])) + np.array([0.1, -0.05])))
    b.append("\\node[font=\\scriptsize, text=ndInk!70, anchor=south west] at %s {weight $10^{-8}$};" % Q.pt(4.6, -7.9))
    b.append("\\node[font=\\small] at %s {(b) the Schmidt weights across the middle};" % N.pt(Q.P(np.log2(m) / 2, -16.5) + np.array([0, -1.15])))
    N.write("ch07_tn_entropy", b)
    print("written figures/ch07_tn_entropy.tex")


if __name__ == "__main__":
    main()
