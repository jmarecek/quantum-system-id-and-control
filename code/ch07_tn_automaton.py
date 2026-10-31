"""ch07_tn_automaton.py -- a matrix product operator is a finite automaton that writes the Hamiltonian.

The controlled Ising chain of Balada Gaggioli and Marecek (PRA 112, 062612, 2025),
    H(t) = H_0 + u(t) H_c,   H_0 = J sum_{i<N} sz_i sz_{i+1},   H_c = sum_i sx_i,
is written as matrix product operators (MPOs) with the tensors of that paper:
    H_0:  W^[1] = (0, J sz, I),   W^[i] = [[I, 0, 0], [sz, 0, 0], [0, J sz, I]],   W^[N] = (I, sz, 0)^T,
    H_c:  W^[1] = (I, sx),        W^[i] = [[I, sx], [0, I]],                      W^[N] = (sx, I)^T.
Read W^[i]_{ab} as the operator written when passing from the state a to the
state b of an automaton (Crosswhite and Bacon, PRA 78, 012356, 2008).  For H_0
the states are 3 "not yet", 2 "one sz written", 1 "done"; every path from
"not yet" to "done" writes exactly one term J sz_k sz_{k+1}.  For H_c they are
1 "not yet" and 2 "done", and each path writes one sx_k.  The bond dimensions
3 and 2 do not depend on N, so that the MPO stores 4 N r^2 numbers where the
dense matrix stores 4^N.

This module also holds the MPO helpers used by ch07_tn_algebra.py and
ch07_tn_ladder.py.  A core is an array of shape (r_left, r_right, d, d); the
first core has r_left = 1 and the last r_right = 1.
Writes figures/ch07_tn_automaton.tex.
"""
import numpy as np
import needham as N

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.diag([1.0, -1.0]).astype(complex)
O2 = np.zeros((2, 2), dtype=complex)


def cores_from_blocks(first, middle, last, n):
    """MPO of n sites from operator-valued row, matrix and column (lists of 2x2 arrays)."""
    row = np.array([first], dtype=complex)                    # (1, r, 2, 2)
    mid = np.array(middle, dtype=complex)                     # (r, r, 2, 2)
    col = np.array([[w] for w in last], dtype=complex)        # (r, 1, 2, 2)
    return [row] + [mid.copy() for _ in range(n - 2)] + [col]


def ising_mpo(n, J=1.0):
    """H_0 = J sum sz_i sz_{i+1}: bond dimension 3 (states 1 done, 2 one sz written, 3 not yet)."""
    return cores_from_blocks([O2, J * SZ, I2],
                             [[I2, O2, O2], [SZ, O2, O2], [O2, J * SZ, I2]],
                             [I2, SZ, O2], n)


def control_mpo(n):
    """H_c = sum sx_i: bond dimension 2 (states 1 not yet, 2 done)."""
    return cores_from_blocks([I2, SX], [[I2, SX], [O2, I2]], [SX, I2], n)


def mpo_to_dense(mpo):
    """Contract the bonds left to right; site 1 is the leftmost factor of the Kronecker product."""
    ops = list(mpo[0][0])                                     # one operator per right bond index
    for W in mpo[1:]:
        ops = [sum(np.kron(ops[a], W[a, b]) for a in range(W.shape[0])) for b in range(W.shape[1])]
    assert len(ops) == 1
    return ops[0]


def mpo_scale(mpo, c):
    """c times the operator: only the first core changes."""
    return [c * mpo[0]] + [W.copy() for W in mpo[1:]]


def mpo_sum(A, B):
    """A + B: the bonds add (end cores side by side, middle cores block diagonal)."""
    out = []
    for k, (WA, WB) in enumerate(zip(A, B)):
        if k == 0:
            out.append(np.concatenate([WA, WB], axis=1))
        elif k == len(A) - 1:
            out.append(np.concatenate([WA, WB], axis=0))
        else:
            ra, ra2 = WA.shape[:2]
            rb, rb2 = WB.shape[:2]
            C = np.zeros((ra + rb, ra2 + rb2) + WA.shape[2:], dtype=complex)
            C[:ra, :ra2] = WA
            C[ra:, ra2:] = WB
            out.append(C)
    return out


def mpo_product(A, B):
    """The operator product A B: the bonds multiply, C[(a,a'),(b,b')] = A[a,b] B[a',b'] on each site."""
    out = []
    for WA, WB in zip(A, B):
        C = np.einsum("abij,cdjk->acbdik", WA, WB)
        out.append(C.reshape(WA.shape[0] * WB.shape[0], WA.shape[1] * WB.shape[1], WA.shape[2], WB.shape[3]))
    return out


def bond_dims(mpo):
    return [W.shape[1] for W in mpo[:-1]]


def dense_chain(n, J=1.0):
    """Dense H_0 and H_c for comparison."""
    def site(op, k):
        out = np.array([[1.0 + 0j]])
        for i in range(n):
            out = np.kron(out, op if i == k else I2)
        return out
    H0 = sum(J * site(SZ, k) @ site(SZ, k + 1) for k in range(n - 1))
    Hc = sum(site(SX, k) for k in range(n))
    return H0, Hc


def checks():
    for n in range(2, 9):
        H0, Hc = dense_chain(n)
        assert np.allclose(mpo_to_dense(ising_mpo(n)), H0)
        assert np.allclose(mpo_to_dense(control_mpo(n)), Hc)
        if n > 2:
            assert set(bond_dims(ising_mpo(n))) == {3} and set(bond_dims(control_mpo(n))) == {2}
    # the highlighted path writes J sz_3 sz_4 on six sites: states after each site 3, 3, 2, 1, 1
    W = ising_mpo(6)
    states, word = [3, 3, 2, 1, 1], []
    a = 3
    for k, b in enumerate(states + [None]):
        if k == 0:
            word.append(W[0][0, b - 1])
        elif b is None:
            word.append(W[-1][a - 1, 0])
        else:
            word.append(W[k][a - 1, b - 1])
        a = b
    term = word[0]
    for w in word[1:]:
        term = np.kron(term, w)
    H0, _ = dense_chain(6)
    single = np.kron(np.kron(np.kron(np.kron(np.kron(I2, I2), SZ), SZ), I2), I2)
    assert np.allclose(term, single)
    n = 20
    r0, rc = 3, 2
    dense = 4 ** n
    print("N = %d: dense matrix 4^N = %.3g numbers; MPO of H_0 about 4 N r^2 = %d, of H_c %d (exactly %d and %d)"
          % (n, dense, 4 * n * r0 ** 2, 4 * n * rc ** 2,
             sum(W.size for W in ising_mpo(n)), sum(W.size for W in control_mpo(n))))


def state(name, pos, label, num, fill="ndSky!35", accept=False, start=None):
    out = ["\\node[circle, draw=ndInk, line width=0.8pt, fill=%s, minimum size=1.15cm, align=center, "
           "font=\\scriptsize\\itshape] (%s) at %s {%s};" % (fill, name, N.pt(pos), label)]
    if accept:
        out.append("\\draw[ndInk, line width=0.5pt] (%s) circle (0.64cm);" % name)
    out.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%d$};" % (N.pt(np.asarray(pos) + np.array([0.0, -0.82 if not accept else -0.88])), num))
    if start is not None:
        out.append("\\draw[->, >=stealth, line width=0.8pt] %s -- (%s.west) node[pos=0, left, font=\\scriptsize\\itshape] {start};"
                   % (N.pt(np.asarray(pos) + np.array([-1.15, 0.0])), name))
    return out


def loop(name, lab, side="above"):
    if side == "above":
        return "\\draw[->, >=stealth, line width=0.8pt] (%s.110) to[out=120, in=60, looseness=5] node[above, font=\\scriptsize] {$%s$} (%s.70);" % (name, lab, name)
    return "\\draw[->, >=stealth, line width=0.8pt] (%s.250) to[out=240, in=300, looseness=5] node[below, font=\\scriptsize] {$%s$} (%s.290);" % (name, lab, name)


def core(name, pos, label, up="", down=""):
    out = ["\\node[rectangle, rounded corners=3pt, minimum size=0.62cm, fill=ndRose, draw=ndRed, line width=0.9pt, "
           "font=\\scriptsize] (%s) at %s {%s};" % (name, N.pt(pos), label)]
    out.append("\\draw[ink] (%s.north) -- ++(0,0.42) node[above, font=\\scriptsize] {%s};" % (name, up))
    out.append("\\draw[ink] (%s.south) -- ++(0,-0.42) node[below, font=\\scriptsize] {%s};" % (name, down))
    return out


def main():
    checks()
    b = []
    # ---- (a) the automaton of H_0
    y = 4.3
    b += state("n3", (0.0, y), "not\\\\ yet", 3, start=True)
    b += state("n2", (2.7, y), "one $\\sigma^z$\\\\ written", 2)
    b += state("n1", (5.4, y), "done", 1, fill="ndGold!30", accept=True)
    b.append(loop("n3", "\\Id"))
    b.append(loop("n1", "\\Id"))
    b.append("\\draw[->, >=stealth, line width=0.9pt, ndRed] (n3) -- node[above, font=\\scriptsize] {$J\\sigma^z$} (n2);")
    b.append("\\draw[->, >=stealth, line width=0.9pt, ndRed] (n2) -- node[above, font=\\scriptsize] {$\\sigma^z$} (n1);")
    b.append("\\node[font=\\scriptsize, anchor=west] at (6.55,%.2f) {$W^{[i]}=\\begin{pmatrix}\\Id&0&0\\\\ \\sigma^z&0&0\\\\ 0&J\\sigma^z&\\Id\\end{pmatrix}$};" % y)
    b.append("\\node[note, anchor=north] at (2.7,%.2f) {every path from \\emph{start} to \\emph{done}\\\\ writes one term $J\\sigma^z_k\\sigma^z_{k+1}$};" % (y - 1.2))
    b.append("\\node[font=\\small] at (4.2,%.2f) {(a) $H_0=J\\sum_k\\sigma^z_k\\sigma^z_{k+1}$: three states};" % (y - 2.35))
    # ---- (b) the automaton of H_c
    xb = 12.9
    b += state("c1", (xb, y), "not\\\\ yet", 1, start=True)
    b += state("c2", (xb + 2.7, y), "done", 2, fill="ndGold!30", accept=True)
    b.append(loop("c1", "\\Id"))
    b.append(loop("c2", "\\Id"))
    b.append("\\draw[->, >=stealth, line width=0.9pt, ndRed] (c1) -- node[above, font=\\scriptsize] {$\\sigma^x$} (c2);")
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.2f,%.2f) {$W^{[i]}=\\begin{pmatrix}\\Id&\\sigma^x\\\\ 0&\\Id\\end{pmatrix}$};" % (xb + 1.35, y - 1.1))
    b.append("\\node[font=\\small] at (%.2f,%.2f) {(b) $H_c=\\sum_k\\sigma^x_k$: two states};" % (xb + 1.35, y - 2.35))
    # ---- (c) the chain of six cores and one path through the automaton
    y0, dx, x0 = 0.0, 1.9, 2.2
    for k in range(6):
        b += core("w%d" % k, (x0 + k * dx, y0), "$W^{[%d]}$" % (k + 1), "$s_%d$" % (k + 1), "$s'_%d$" % (k + 1))
    for k in range(5):
        b.append("\\draw[ink] (w%d.east) -- node[above, font=\\scriptsize] {$3$} (w%d.west);" % (k, k + 1))
    letters = ["\\Id", "\\Id", "J\\sigma^z", "\\sigma^z", "\\Id", "\\Id"]
    path_states = [3, 3, 2, 1, 1]
    yl = y0 - 1.45
    for k, lab in enumerate(letters):
        col = "ndRed" if "sigma" in lab else "ndInk"
        b.append("\\node[font=\\small, text=%s] at (%.3f,%.3f) {$%s$};" % (col, x0 + k * dx, yl, lab))
    for k, s_ in enumerate(path_states):
        b.append("\\node[circle, draw=ndInk!60, fill=ndSky!35, inner sep=0.5pt, minimum size=0.36cm, font=\\scriptsize] at (%.3f,%.3f) {$%d$};"
                 % (x0 + (k + 0.5) * dx, yl, s_))
    b.append("\\node[note, anchor=east] at (%.3f,%.3f) {one path, with the\\\\ state on each bond:};" % (x0 - 0.45, yl))
    b.append("\\node[note, anchor=west] at (%.3f,%.3f) {writes $J\\sigma^z_3\\sigma^z_4$};" % (x0 + 5 * dx + 0.55, yl))
    b.append("\\node[note, anchor=west] at (%.3f,%.3f) {bond dimension $3$ for\\\\ every $N$: $4Nr^2$ numbers\\\\ instead of $4^N$};" % (x0 + 5 * dx + 0.75, y0 + 0.1))
    b.append("\\node[note, anchor=east] at (%.3f,%.3f) {the bond index is\\\\ the state of the\\\\ automaton};" % (x0 - 0.55, y0 + 0.2))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(c) the MPO of $H_0$ on six sites, and the path that writes $J\\sigma^z_3\\sigma^z_4$};" % (x0 + 2.5 * dx, yl - 0.75))
    N.write("ch07_tn_automaton", b)
    print("written figures/ch07_tn_automaton.tex")


if __name__ == "__main__":
    main()
