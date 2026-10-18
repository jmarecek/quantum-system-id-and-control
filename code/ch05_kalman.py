"""ch05_kalman.py -- the Kalman decomposition as four rooms and the doors between them.

In coordinates adapted to the controllable subspace and the unobservable
subspace, with the state ordered as (x_co, x_cO, x_Co, x_CO) (c: controllable,
o: observable, capital letter: not),

        [ A_co  0     A_13  0    ]        [ B_co ]
    A = [ A_21  A_cO  A_23  A_24 ],   B = [ B_cO ],   C = [ C_co  0  C_Co  0 ].
        [ 0     0     A_Co  0    ]        [ 0    ]
        [ 0     0     A_43  A_CO ]        [ 0    ]

An arrow j -> i is drawn when block A_ij may be nonzero, u -> i when B_i may be
nonzero, and i -> y when C_i may be nonzero.  The input reaches only the two
controllable rooms, the output reads only the two observable ones, and the
only route from u to y passes through x_co, so that
    G(s) = C (sI - A)^{-1} B = C_co (sI - A_co)^{-1} B_co.
This is checked on a random instance.  Writes figures/ch05_kalman.tex.
"""
import numpy as np
import needham as N

SIZES = (2, 1, 2, 1)


def random_kalman_form(rng):
    """Random (A, B, C) with the block structure of the docstring."""
    n = sum(SIZES)
    idx = np.cumsum((0,) + SIZES)
    blk = lambda i: slice(idx[i], idx[i + 1])
    allowed = {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2), (3, 2), (3, 3)}
    A = np.zeros((n, n))
    for i in range(4):
        for j in range(4):
            if (i, j) in allowed:
                A[blk(i), blk(j)] = rng.normal(size=(SIZES[i], SIZES[j]))
    A -= 3.0 * np.eye(n)
    B = np.zeros((n, 1)); C = np.zeros((1, n))
    for i in (0, 1):
        B[blk(i)] = rng.normal(size=(SIZES[i], 1))
    for i in (0, 2):
        C[:, blk(i)] = rng.normal(size=(1, SIZES[i]))
    return A, B, C, blk


def check():
    rng = np.random.default_rng(7)
    A, B, C, blk = random_kalman_form(rng)
    n = len(A)
    Aco, Bco, Cco = A[blk(0), blk(0)], B[blk(0)], C[:, blk(0)]
    for s in (0.3 + 1.0j, -0.7 + 2.5j, 4.0 + 0.0j):
        G = C @ np.linalg.solve(s * np.eye(n) - A, B)
        Gco = Cco @ np.linalg.solve(s * np.eye(len(Aco)) - Aco, Bco)
        assert np.allclose(G, Gco)
    # the controllable subspace is blocks (co, cO); the unobservable one is (cO, CO)
    K = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    O = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])
    assert np.allclose(K[blk(2)], 0) and np.allclose(K[blk(3)], 0)
    assert np.allclose(O[:, blk(1)], 0) and np.allclose(O[:, blk(3)], 0)


def main():
    check()
    w, h = 3.3, 1.35
    pos = {"co": (0.0, 1.5), "cO": (-3.2, -0.9), "Co": (3.2, -0.9), "CO": (0.0, -2.9)}
    title = {"co": "controllable\\\\ and observable", "cO": "controllable,\\\\ not observable",
             "Co": "observable,\\\\ not controllable", "CO": "neither"}
    sym = {"co": "$\\vec x_{co}$", "cO": "$\\vec x_{c\\bar o}$", "Co": "$\\vec x_{\\bar co}$", "CO": "$\\vec x_{\\bar c\\bar o}$"}
    b = []
    for k, (x, y) in pos.items():
        fill = "ndRose" if k == "co" else "ndSky"
        edge = "ndRed, line width=1.3pt" if k == "co" else "ink"
        b.append("\\draw[%s, fill=%s, fill opacity=0.7, rounded corners=3pt] (%.2f,%.2f) rectangle (%.2f,%.2f);"
                 % (edge, fill, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        b.append("\\node[font=\\small] at (%.2f,%.2f) {%s};" % (x, y + 0.33, sym[k]))
        b.append("\\node[note] at (%.2f,%.2f) {%s};" % (x, y - 0.22, title[k]))
    U, Y = (-6.6, 1.5), (6.6, 1.5)
    b.append("\\node[font=\\large] (u) at (%.2f,%.2f) {$u$};" % U)
    b.append("\\node[font=\\large] (y) at (%.2f,%.2f) {$y$};" % Y)
    E = lambda k, dx, dy: (pos[k][0] + dx * w / 2, pos[k][1] + dy * h / 2)
    arr = "->, >=stealth, line width=0.8pt, ndInk!80"
    hot = "->, >=stealth, line width=1.8pt, ndRed"
    lab = "font=\\scriptsize, text=ndInk!80"
    # the route u -> co -> y
    b.append("\\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f) node[midway, above, %s, text=ndRed] {$\\mathbf B_{co}$};" % (hot, U[0] + 0.25, U[1], *E("co", -1, 0), lab))
    b.append("\\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f) node[midway, above, %s, text=ndRed] {$\\mathbf C_{co}$};" % (hot, *E("co", 1, 0), Y[0] - 0.25, Y[1], lab))
    # u -> cO, Co -> y
    b.append("\\draw[%s] (%.2f,%.2f) to[out=-90, in=180] (%.2f,%.2f);" % (arr, U[0] + 0.05, U[1] - 0.25, *E("cO", -1, 0)))
    b.append("\\node[%s] at (%.2f,%.2f) {$\\mathbf B_{c\\bar o}$};" % (lab, U[0] + 0.95, 0.15))
    b.append("\\draw[%s] (%.2f,%.2f) to[out=0, in=-90] (%.2f,%.2f);" % (arr, *E("Co", 1, 0), Y[0] - 0.05, Y[1] - 0.25))
    b.append("\\node[%s] at (%.2f,%.2f) {$\\mathbf C_{\\bar co}$};" % (lab, Y[0] - 0.95, 0.15))
    # couplings A_ij: j -> i
    b.append("\\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f) node[midway, above left, %s] {$\\mathbf A_{21}$};" % (arr, *E("co", -0.55, -1), *E("cO", 0.45, 1), lab))
    b.append("\\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f) node[midway, above right, %s] {$\\mathbf A_{13}$};" % (arr, *E("Co", -0.45, 1), *E("co", 0.55, -1), lab))
    b.append("\\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f) node[midway, above, %s] {$\\mathbf A_{23}$};" % (arr, *E("Co", -1, 0), *E("cO", 1, 0), lab))
    b.append("\\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f) node[midway, below left, %s] {$\\mathbf A_{24}$};" % (arr, *E("CO", -1, 0.1), *E("cO", 0.3, -1), lab))
    b.append("\\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f) node[midway, below right, %s] {$\\mathbf A_{43}$};" % (arr, *E("Co", -0.3, -1), *E("CO", 1, 0.1), lab))
    # notes
    b.append("\\node[note, anchor=north] (d1) at (-5.0,-2.15) {reached from $u$,\\\\ but never seen};")
    b.append("\\draw[pointer] (d1.north) to[bend left=20] (%.2f,%.2f);" % E("cO", -0.75, -1))
    b.append("\\node[note, anchor=north] (d2) at (5.0,-2.15) {seen in $y$, but\\\\ never reached from $u$};")
    b.append("\\draw[pointer] (d2.north) to[bend right=20] (%.2f,%.2f);" % E("Co", 0.75, -1))
    b.append("\\node[note, anchor=south, text=ndRed] at (0.0,%.2f) {the only path from $u$ to $y$:\\ "
             "$\\TF(s)=\\mathbf C_{co}(s\\Id-\\mathbf A_{co})^{-1}\\mathbf B_{co}$};" % (1.5 + h / 2 + 0.12))
    b.append("\\node[note, anchor=north] at (0.0,%.2f) {no arrow leaves $\\vec x_{c\\bar o}$ towards $y$,\\\\ "
             "no arrow enters $\\vec x_{\\bar co}$ from $u$};" % (-2.9 - h / 2 - 0.12))
    N.write("ch05_kalman", b)
    print("written figures/ch05_kalman.tex")


if __name__ == "__main__":
    main()
