"""ch07_tn_networks.py -- one graphical language for states, operators and networks.

(a) A state of N = 6 qubits is one tensor with N legs, 2^N numbers.
(b) A matrix product state (MPS) writes it as a chain of order-three tensors
    A^[k]_{alpha s beta}, one per site, joined by bonds of dimension D:
    about N d D^2 numbers.
(c) A matrix product operator (MPO) has two physical legs per site, the output
    s_k up and the input s'_k down, and bonds of dimension r: about N d^2 r^2.
(d) <psi|O|psi> is the closed sandwich of a bra, an MPO and a ket: a network
    with no open legs, i.e. a number.
(e) Beyond chains: a tree tensor network and a 3 x 3 grid (PEPS).
Every network is a graph: tensors on the vertices, summed indices on the
edges, physical indices on the open legs.  The contractions are checked with
numpy.einsum on random tensors, and the parameter counts are printed.
Writes figures/ch07_tn_networks.tex.
"""
import numpy as np
import needham as N

GOLD = "draw=ndGold!70!ndInk, fill=ndGold!25, rounded corners=3pt, line width=0.9pt, inner sep=1pt, font=\\scriptsize"
KET = "circle, draw=ndBlue, fill=ndSky, line width=0.9pt, minimum size=0.52cm, inner sep=0pt, font=\\scriptsize"
BRA = "circle, draw=ndBlue, fill=ndSky!40, line width=0.9pt, minimum size=0.52cm, inner sep=0pt, font=\\scriptsize"
MPO = "rounded corners=3pt, draw=ndRed, fill=ndRose, line width=0.9pt, minimum size=0.55cm, inner sep=0pt, font=\\scriptsize"
SMALL = "circle, draw=ndBlue, fill=ndSky, line width=0.8pt, minimum size=0.34cm, inner sep=0pt"
NS, D_, PITCH = 6, 2, 1.0


def bonds(n, D, d=2):
    """Open-boundary bond dimensions D_0 = 1, ..., D_n = 1, capped by D and by the dimensions on either side."""
    return [min(D, d ** k, d ** (n - k)) for k in range(n + 1)]


def mps_vector(As):
    v = As[0][0]                                                   # (d, D_1)
    for A in As[1:]:
        v = np.einsum("xa,asb->xsb", v, A).reshape(-1, A.shape[2])
    return v[:, 0]


def mpo_dense(Ws):
    M = Ws[0][0]                                                   # (d_out, d_in, r_1)
    for W in Ws[1:]:
        M = np.einsum("ija,aklb->ikjlb", M, W).reshape(M.shape[0] * W.shape[1], M.shape[1] * W.shape[2], W.shape[3])
    return M[:, :, 0]


def sandwich(As, Ws):
    """<psi|O|psi> contracted site by site, from left to right."""
    E = np.ones((1, 1, 1), dtype=complex)                          # (bra bond, MPO bond, ket bond)
    for A, W in zip(As, Ws):
        E = np.einsum("abc,asx,bsty,ctz->xyz", E, A.conj(), W, A)
    return E[0, 0, 0]


def checks():
    rng = np.random.default_rng(5)
    c = lambda *s: rng.normal(size=s) + 1j * rng.normal(size=s)
    Dk, rk = bonds(NS, 4), [1] + [3] * (NS - 1) + [1]
    As = [c(Dk[k], D_, Dk[k + 1]) for k in range(NS)]
    Ws = [c(rk[k], D_, D_, rk[k + 1]) for k in range(NS)]
    psi, O = mps_vector(As), mpo_dense(Ws)
    assert psi.shape == (D_ ** NS,) and O.shape == (D_ ** NS, D_ ** NS)
    assert np.isclose(sandwich(As, Ws), np.vdot(psi, O @ psi))
    for n in (6, 50):
        Dk = bonds(n, 8)
        mps = sum(D_ * Dk[k] * Dk[k + 1] for k in range(n))
        print("N = %d: state 2^N = %.3g numbers; MPS with D = 8: %d (at most N d D^2 = %d); "
              "operator 4^N = %.3g; MPO with r = 8: at most N d^2 r^2 = %d"
              % (n, 2.0 ** n, mps, n * D_ * 64, 4.0 ** n, n * D_ ** 2 * 64))


def chain(b, style, y, name, labels=None, up=None, down=None, bond=None):
    """Six cores in a row at height y, with optional labels, up/down legs and bond labels."""
    xs = [0.6 + PITCH * k for k in range(NS)]
    for k, x in enumerate(xs):
        b.append("\\node[%s] (%s%d) at (%.3f,%.3f) {%s};" % (style, name, k, x, y, labels[k] if labels else ""))
        if up:
            b.append("\\draw[ndInk, line width=0.9pt] (%s%d.north) -- ++(0,0.4) node[above, font=\\scriptsize, inner sep=1pt] {%s};" % (name, k, up[k]))
        if down:
            b.append("\\draw[ndInk, line width=0.9pt] (%s%d.south) -- ++(0,-0.4) node[below, font=\\scriptsize, inner sep=1pt] {%s};" % (name, k, down[k]))
    for k in range(NS - 1):
        lab = "" if bond is None else " node[midway, above, font=\\scriptsize, inner sep=1.5pt] {$%s$}" % bond
        b.append("\\draw[ndInk, line width=0.9pt] (%s%d.east) --%s (%s%d.west);" % (name, k, lab, name, k + 1))
    return xs


def main():
    checks()
    b = []
    s_up = ["$s_%d$" % (k + 1) for k in range(NS)]
    # ---- (a) one tensor with N legs
    ya = 7.25
    xs = [0.6 + PITCH * k for k in range(NS)]
    b.append("\\node[%s, minimum width=%.2fcm, minimum height=0.55cm] (psi) at (%.3f,%.3f) {$\\psi$};" % (GOLD, 5 * PITCH + 0.5, 0.5 * (xs[0] + xs[-1]), ya))
    for k, x in enumerate(xs):
        b.append("\\draw[ndInk, line width=0.9pt] (%.3f,%.3f) -- ++(0,0.4) node[above, font=\\scriptsize, inner sep=1pt] {%s};" % (x, ya + 0.275, s_up[k]))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) a state of $N=6$ qubits: $2^N$ numbers};" % (0.5 * (xs[0] + xs[-1]), ya - 0.75))
    # ---- (b) a matrix product state
    yb = 4.75
    chain(b, KET, yb, "a", ["$A^{[%d]}$" % (k + 1) for k in range(NS)], up=s_up, bond="D")
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) a matrix product state: about $NdD^2$ numbers};" % (0.5 * (xs[0] + xs[-1]), yb - 0.75))
    # ---- (c) a matrix product operator
    yc = 2.15
    chain(b, MPO, yc, "w", ["$W^{[%d]}$" % (k + 1) for k in range(NS)], up=s_up,
          down=["$s'_%d$" % (k + 1) for k in range(NS)], bond="r")
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(c) a matrix product operator: about $Nd^2r^2$ numbers};" % (0.5 * (xs[0] + xs[-1]), yc - 1.25))

    # ---- (d) <psi|O|psi>: the closed sandwich
    x0, yd = 8.75, 6.3
    sx = [x0 + 0.95 * k for k in range(NS)]
    for k, x in enumerate(sx):
        b.append("\\node[%s] (u%d) at (%.3f,%.3f) {};" % (BRA, k, x, yd + 1.0))
        b.append("\\node[%s] (o%d) at (%.3f,%.3f) {};" % (MPO, k, x, yd))
        b.append("\\node[%s] (d%d) at (%.3f,%.3f) {};" % (KET, k, x, yd - 1.0))
        b.append("\\draw[ndInk, line width=0.9pt] (u%d.south) -- (o%d.north);" % (k, k))
        b.append("\\draw[ndInk, line width=0.9pt] (o%d.south) -- (d%d.north);" % (k, k))
    for k in range(NS - 1):
        for row in "uod":
            b.append("\\draw[ndInk, line width=0.9pt] (%s%d.east) -- (%s%d.west);" % (row, k, row, k + 1))
    for row, lab in [("u", "$\\langle\\psi|$"), ("o", "$O$"), ("d", "$|\\psi\\rangle$")]:
        b.append("\\node[font=\\scriptsize, anchor=east] at ($(%s0.west)+(-0.12,0)$) {%s};" % (row, lab))
    b.append("\\node[note, anchor=west] (nnum) at ($(o%d.east)+(0.45,0)$) {no open legs:\\\\ a number};" % (NS - 1))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(d) the expectation value $\\langle\\psi|O|\\psi\\rangle$};" % (0.5 * (sx[0] + sx[-1]), yd - 1.75))

    # ---- (e) beyond chains: a tree and a grid
    ye = 1.9
    leaves = [8.05, 8.85, 9.85, 10.65]
    mids = [0.5 * (leaves[0] + leaves[1]), 0.5 * (leaves[2] + leaves[3])]
    root = 0.5 * (mids[0] + mids[1])
    b.append("\\node[%s, minimum size=0.38cm] (rt) at (%.3f,%.3f) {};" % (GOLD, root, ye))
    for i, m in enumerate(mids):
        b.append("\\node[%s, minimum size=0.38cm] (m%d) at (%.3f,%.3f) {};" % (GOLD, i, m, ye + 0.8))
        b.append("\\draw[ndInk, line width=0.9pt] (rt) -- (m%d);" % i)
    for i, x in enumerate(leaves):
        b.append("\\node[%s] (l%d) at (%.3f,%.3f) {};" % (SMALL, i, x, ye + 1.6))
        b.append("\\draw[ndInk, line width=0.9pt] (m%d) -- (l%d);" % (i // 2, i))
        b.append("\\draw[ndInk, line width=0.9pt] (l%d.north) -- ++(0,0.32);" % i)
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {a tree};" % (root, ye - 0.3))
    gx, gy, h = 12.3, ye - 0.05, 0.72
    for i in range(3):
        for j in range(3):
            b.append("\\node[%s] (g%d%d) at (%.3f,%.3f) {};" % (SMALL, i, j, gx + h * j, gy + h * i))
            b.append("\\draw[ndInk, line width=0.9pt] (g%d%d.north east) -- ++(0.2,0.24);" % (i, j))
    for i in range(3):
        for j in range(2):
            b.append("\\draw[ndInk, line width=0.9pt] (g%d%d.east) -- (g%d%d.west);" % (i, j, i, j + 1))
            b.append("\\draw[ndInk, line width=0.9pt] (g%d%d.north) -- (g%d%d.south);" % (j, i, j + 1, i))
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {a grid (PEPS)};" % (gx + h, ye - 0.3))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(e) beyond chains};" % (0.5 * (leaves[0] + gx + 2 * h), yc - 1.25))
    b.append("\\node[note] at (7.9,%.3f) {every network is a graph: tensors on the vertices, summed indices on the edges, physical indices on the open legs};" % (yc - 1.85))
    N.write("ch07_tn_networks", b)
    print("written figures/ch07_tn_networks.tex")


if __name__ == "__main__":
    main()
