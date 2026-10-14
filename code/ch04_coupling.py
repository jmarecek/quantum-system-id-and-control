"""ch04_coupling.py -- observability as information flow on the coupling graph.

The two-qubit example of Chapter 4 in the normalised Pauli basis: the vertices
are the 15 Pauli strings and the constant II of the affine embedding; there is
an arrow F -> F' whenever the component along F appears in the equation for
the component along F', i.e. A_{F'F} != 0 (beta for II).  The graph splits
into two sectors that never interact: populations and exchange
(II, ZI, IZ, ZZ, XX, XY, YX, YY), and single-qubit coherences (the rest), which
form a cube.  Measuring sigma_z^1 (ZI): seven components have a path to it, but
the observability matrix has rank 5 only, since XX, YY, XY, YX enter through
the two combinations XX+YY and XY-YX.  Adding sigma_x^1 (XI) reaches the
other sector: 15 components, rank 13.  Writes figures/ch04_coupling.tex.
"""
import itertools
import numpy as np
import needham as N
import ch04_observability as M

I2, X, Y, Z = M.I2, M.X, M.Y, M.Z
NAMES = ["".join(p) for p in itertools.product("IXYZ", repeat=2)][1:] + ["II"]


def model():
    sm, sp = (X + 1j * Y) / 2, (X - 1j * Y) / 2          # sigma_- = |0><1|, as in Chapter 3
    k1 = lambda P: np.kron(P, I2)
    k2 = lambda P: np.kron(I2, P)
    w1, w2, delta, gz, gm, gp = 1.0, 1.3, 0.4, 0.05, 0.08, 0.02
    H = -w1 / 2 * k1(Z) - w2 / 2 * k2(Z) + delta * (k1(sp) @ k2(sm) + k1(sm) @ k2(sp))
    jumps = []
    for kk in (k1, k2):
        jumps += [np.sqrt(gz / 2) * kk(Z), np.sqrt(2 * gm) * kk(sm), np.sqrt(2 * gp) * kk(sp)]
    A, beta = M.coherence_representation(M.lindbladian(H, jumps), M.pauli_basis(2))
    Ae = np.zeros((16, 16)); Ae[:15, :15] = A; Ae[:15, 15] = beta
    return Ae


def backward_reach(Ae, measured):
    seen, todo = set(measured), list(measured)
    while todo:
        i = todo.pop()
        for j in range(16):
            if abs(Ae[i, j]) > 1e-12 and j not in seen:
                seen.add(j); todo.append(j)
    return seen


def rank_and_space(Ae, rows):
    C = np.zeros((len(rows), 16))
    for r, i in enumerate(rows):
        C[r, i] = 1.0
    O = M.observability_matrix(Ae, C)
    _, s, vt = np.linalg.svd(O)
    r = int(np.sum(s > 1e-9))
    return r, vt[:r]


def in_space(V, vec):
    return np.allclose(V.T @ (V @ vec), vec)


def cube_layout(nodes, edges):
    """Find an embedding of an 8-vertex graph as the cube {0,1}^3 (edges = Hamming distance 1)."""
    verts = list(itertools.product((0, 1), repeat=3))
    for perm in itertools.permutations(verts):
        pos = dict(zip(nodes, perm))
        if all(sum(a != b for a, b in zip(pos[u], pos[v])) == 1 for u, v in edges):
            return pos
    raise ValueError("not a cube")


def main():
    Ae = model()
    idx = {n: i for i, n in enumerate(NAMES)}
    und = {tuple(sorted((NAMES[i], NAMES[j]))) for i in range(16) for j in range(16) if i != j and abs(Ae[i, j]) > 1e-12}
    sector1 = ["II", "ZI", "IZ", "ZZ", "XX", "XY", "YX", "YY"]
    sector2 = [n for n in NAMES if n not in sector1]
    assert all((u in sector1) == (v in sector1) for u, v in und)          # the two sectors never couple
    # sigma_z^1 readout
    reach_z = {NAMES[i] for i in backward_reach(Ae, [idx["ZI"]])}
    rank_z, Vz = rank_and_space(Ae, [idx["ZI"]])
    assert reach_z == set(sector1) - {"ZZ"} and rank_z == 5
    vec = lambda pairs: sum(c * np.eye(16)[idx[n]] for n, c in pairs)
    assert in_space(Vz, vec([("XX", 1), ("YY", 1)])) and in_space(Vz, vec([("XY", 1), ("YX", -1)]))
    assert not any(in_space(Vz, vec([(n, 1)])) for n in ("XX", "YY", "XY", "YX"))
    # sigma_z^1 and sigma_x^1 readout
    reach_zx = {NAMES[i] for i in backward_reach(Ae, [idx["ZI"], idx["XI"]])}
    rank_zx, _ = rank_and_space(Ae, [idx["ZI"], idx["XI"]])
    assert reach_zx == set(NAMES) - {"ZZ"} and rank_zx == 13
    assert abs(Ae[idx["ZZ"], idx["ZI"]]) > 0 and abs(Ae[idx["ZZ"], idx["IZ"]]) > 0
    assert Ae[idx["ZI"], idx["ZZ"]] == 0 and Ae[idx["IZ"], idx["ZZ"]] == 0       # ZZ is driven, never drives
    print("reach(ZI) =", len(reach_z), "rank", rank_z, "; reach(ZI,XI) =", len(reach_zx), "rank", rank_zx)

    # layout: sector 1 by hand, sector 2 as a cube in oblique projection
    pos = {"II": (0.0, 2.7), "ZI": (-1.35, 1.6), "IZ": (1.35, 1.6), "ZZ": (0.0, 1.6),
           "XY": (-0.8, 0.3), "YX": (0.8, 0.3), "XX": (-0.8, -1.0), "YY": (0.8, -1.0)}
    edges2 = [e for e in und if e[0] in sector2]
    cube = cube_layout(sector2, edges2)
    for n, (a, c, d) in cube.items():
        pos[n] = (5.2 + 2.3 * a + 1.0 * d, -0.9 + 2.3 * c + 0.75 * d)

    fill = {n: "white" for n in NAMES}
    for n in reach_z:
        fill[n] = "ndRose"
    for n in sector2:
        fill[n] = "ndGold!35"
    b = []
    for u, v in sorted(und):
        uv, vu = abs(Ae[idx[v], idx[u]]) > 1e-12, abs(Ae[idx[u], idx[v]]) > 1e-12     # u drives v / v drives u
        tip = "<->" if (uv and vu) else ("->" if uv else "<-")
        b.append("\\draw[%s, >=stealth, line width=0.6pt, ndInk!70, shorten <=8pt, shorten >=8pt] (%.2f,%.2f) -- (%.2f,%.2f);"
                 % (tip, pos[u][0], pos[u][1], pos[v][0], pos[v][1]))
    for n in NAMES:
        ring = {"ZI": "draw=ndRed, line width=1.6pt", "XI": "draw=ndGold!80!black, line width=1.6pt"}.get(n, "draw=ndInk, line width=0.6pt")
        b.append("\\node[circle, %s, fill=%s, inner sep=0pt, minimum size=0.62cm, font=\\scriptsize] at (%.2f,%.2f) {$%s$};"
                 % (ring, fill[n], pos[n][0], pos[n][1], n))
    b.append("\\node[font=\\small] at (0.0,-1.9) {populations and exchange};")
    b.append("\\node[font=\\small] at (6.35,-1.9) {single-qubit coherences};")
    b.append("\\node[note, anchor=east] (a) at (-1.9,2.2) {measure $\\sigma_z^1$ (red ring):\\\\ $7$ components have a\\\\ path to it (rose) ...};")
    b.append("\\draw[pointer] (a.east) to[bend left=10] (%.2f,%.2f);" % (pos["ZI"][0] - 0.25, pos["ZI"][1] + 0.25))
    b.append("\\node[note, anchor=east] at (-1.4,-0.35) {... but $\\rank\\OM=5$:\\\\ $XX,YY,XY,YX$ enter\\\\ only as $XX+YY$\\\\ and $XY-YX$};")
    b.append("\\node[note, anchor=south] at (0.0,3.15) {$II$: the constant of the affine embedding};")
    b.append("\\node[note, anchor=west] (zz) at (1.95,2.55) {$ZZ$ is driven by\\\\ $ZI$ and $IZ$, but\\\\ never drives them};")
    b.append("\\draw[pointer] (zz.west) to[bend right=10] (%.2f,%.2f);" % (pos["ZZ"][0] + 0.2, pos["ZZ"][1] + 0.25))
    b.append("\\node[note, anchor=west] at (9.2,2.3) {no arrow joins the two\\\\ sectors: $\\sigma_z^1$ alone never\\\\ sees the coherences};")
    b.append("\\node[note, anchor=west] at (9.2,0.4) {add $\\sigma_x^1$ (gold ring):\\\\ $15$ components, $\\rank\\OM=13$};")
    N.write("ch04_coupling", b)
    print("written figures/ch04_coupling.tex")


if __name__ == "__main__":
    main()
