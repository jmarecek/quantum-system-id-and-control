"""ch07_bell.py -- the Bell-diagonal states of two qubits: a tetrahedron and an octahedron.

A two-qubit state with no local Bloch vectors and a diagonal correlation
matrix is
    rho(t) = (1/4) (I + t1 X(x)X + t2 Y(x)Y + t3 Z(x)Z),   t = (<XX>, <YY>, <ZZ>).
It is diagonal in the Bell basis, with the eigenvalues lambda_B = (1 + t.v_B)/4,
where v_B is the correlation vector of the Bell state B:
    Phi+ = (1,-1,1),  Phi- = (-1,1,1),  Psi+ = (1,1,-1),  Psi- = (-1,-1,-1).
Hence rho(t) is a state exactly when t lies in the tetrahedron with these four
vertices.  It is separable if and only if its partial transpose is positive,
which happens if and only if |t1| + |t2| + |t3| <= 1: the octahedron whose
vertices +-e_i are the midpoints of the edges of the tetrahedron.  The four
corners cut off are entangled.  The Werner states p|Psi-><Psi-| + (1-p) I/4,
t = -p(1,1,1), are separable for p <= 1/3.  The octahedron vertex e3 is the
classically correlated mixture (|00><00| + |11><11|)/2.
The script checks the eigenvalues, compares the partial-transpose test with
the octahedron on a grid, and checks the Werner threshold.
Writes figures/ch07_bell.tex.
"""
import itertools
import numpy as np
import needham as N

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)
k0, k1 = np.array([1, 0], complex), np.array([0, 1], complex)
BELL = {
    "Phi^+": (np.kron(k0, k0) + np.kron(k1, k1)) / np.sqrt(2),
    "Phi^-": (np.kron(k0, k0) - np.kron(k1, k1)) / np.sqrt(2),
    "Psi^+": (np.kron(k0, k1) + np.kron(k1, k0)) / np.sqrt(2),
    "Psi^-": (np.kron(k0, k1) - np.kron(k1, k0)) / np.sqrt(2),
}
VERT = {"Phi^+": np.array([1.0, -1, 1]), "Phi^-": np.array([-1.0, 1, 1]),
        "Psi^+": np.array([1.0, 1, -1]), "Psi^-": np.array([-1.0, -1, -1])}


def rho(t):
    return 0.25 * (np.eye(4) + t[0] * np.kron(X, X) + t[1] * np.kron(Y, Y) + t[2] * np.kron(Z, Z))


def partial_transpose(r):
    return r.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)


def check():
    for name, psi in BELL.items():                                  # the vertices are the Bell states
        P = np.outer(psi, psi.conj())
        t = [np.real(np.trace(P @ np.kron(S, S))) for S in (X, Y, Z)]
        assert np.allclose(t, VERT[name]) and np.allclose(rho(VERT[name]), P)
    rng = np.random.default_rng(5)
    for t in rng.uniform(-1, 1, size=(200, 3)):                      # eigenvalues (1 + t.v_B)/4 on the Bell states
        for name, psi in BELL.items():
            assert np.allclose(rho(t) @ psi, (1 + t @ VERT[name]) / 4 * psi)
    g = np.linspace(-1, 1, 41)
    n_state = n_sep = 0
    for t in itertools.product(g, g, g):
        t = np.array(t)
        lam = np.array([(1 + t @ v) / 4 for v in VERT.values()])
        if lam.min() < 1e-9:                                         # outside, or on the boundary
            continue
        assert np.linalg.eigvalsh(rho(t)).min() > 0
        n_state += 1
        s = np.abs(t).sum()
        if abs(s - 1) < 1e-9:
            continue
        ppt = np.linalg.eigvalsh(partial_transpose(rho(t))).min() >= 0
        assert ppt == (s < 1)                                       # PPT <=> octahedron
        assert ppt == (lam.max() < 0.5)                              # <=> every Bell weight below 1/2
        n_sep += ppt
    for p in np.linspace(0, 1, 301):                                 # Werner states
        r = p * np.outer(BELL["Psi^-"], BELL["Psi^-"].conj()) + (1 - p) * np.eye(4) / 4
        assert np.allclose(r, rho(-p * np.ones(3)))
        if abs(p - 1 / 3) > 1e-9:
            assert (np.linalg.eigvalsh(partial_transpose(r)).min() >= -1e-12) == (p < 1 / 3)
    assert np.allclose(rho([0, 0, 1]), 0.5 * (np.diag([1, 0, 0, 0]) + np.diag([0, 0, 0, 1])))
    print("Bell-diagonal: %d interior grid points are states, %d of them separable (%.3f; volume ratio octahedron/tetrahedron = 1/2)"
          % (n_state, n_sep, n_sep / n_state))


def main():
    check()
    az, el = 192.0, 16.0
    V = N.View(az=az, el=el, R=1.75)
    d = V.d
    names = list(VERT)
    b = []
    faces = []                                                       # the face opposite each vertex, outward normal -v
    for n in names:
        others = [m for m in names if m != n]
        faces.append((n, others, -VERT[n] @ d > 0))
    vis = {n: v for n, _, v in faces}
    # visible faces: rose (entangled corners) with the middle triangle sky (separable)
    for n, others, visible in faces:
        if not visible:
            continue
        tri = [VERT[m] for m in others]
        mid = [(tri[i] + tri[j]) / 2 for i, j in ((0, 1), (1, 2), (2, 0))]
        b.append("\\fill[ndRose, fill opacity=0.85] %s -- %s -- %s -- cycle;" % tuple(V.pt(p) for p in tri))
        b.append("\\fill[ndSky, fill opacity=0.95] %s -- %s -- %s -- cycle;" % tuple(V.pt(p) for p in mid))
    # the octahedron: every edge lies in one face of the tetrahedron
    octv = [s * e for e in np.eye(3) for s in (1, -1)]
    for i, j in itertools.combinations(range(6), 2):
        a, c = octv[i], octv[j]
        if abs(a @ c) > 0.5:                                         # opposite vertices: not an edge
            continue
        face = [n for n in names if np.isclose(-VERT[n] @ a, 1) and np.isclose(-VERT[n] @ c, 1)]
        assert len(face) == 1
        style = "line width=0.8pt, ndBlue" if vis[face[0]] else "line width=0.5pt, ndBlue!70, densely dashed"
        b.append("\\draw[%s] %s -- %s;" % (style, V.pt(a), V.pt(c)))
    # the tetrahedron: an edge is in front if one of its two faces is
    for m, n in itertools.combinations(names, 2):
        adj = [f for f in names if f not in (m, n)]
        style = "ink" if (vis[adj[0]] or vis[adj[1]]) else "dashedink"
        b.append("\\draw[%s] %s -- %s;" % (style, V.pt(VERT[m]), V.pt(VERT[n])))
    # a small tripod for the axes, below left
    o = V.P([0, 0, 0]) + np.array([-3.1, -2.1])
    for e, lab, anc in zip(np.eye(3), ("\\langle XX\\rangle", "\\langle YY\\rangle", "\\langle ZZ\\rangle"), ("west", "east", "south")):
        q = o + 0.75 * (V.P(e) - V.P([0, 0, 0])) / V.R
        b.append("\\draw[faint, ->, >=stealth, ndInk!60] %s -- %s node[anchor=%s, font=\\scriptsize, text=ndInk] {$%s$};"
                 % (N.pt(o), N.pt(q), anc, lab))
    # the Werner segment, and the threshold p = 1/3 on the octahedron
    w3 = -np.ones(3) / 3
    assert np.isclose(np.abs(w3).sum(), 1.0)
    b.append("\\draw[ndRed, line width=1.0pt, densely dashed] %s -- %s;" % (V.pt([0, 0, 0]), V.pt(w3)))
    b.append("\\draw[ndRed, line width=1.2pt] %s -- %s;" % (V.pt(w3), V.pt(VERT["Psi^-"])))
    b.append("\\node[dot, fill=ndRed, minimum size=4pt] at %s {};" % V.pt(w3))
    b.append("\\node[dot, minimum size=3pt] at %s {};" % V.pt([0, 0, 0]))
    b.append("\\node[font=\\small, anchor=east, fill=ndSky, inner sep=1pt] at %s {$\\tfrac14\\Id$};" % N.pt(V.P([0, 0, 0]) + np.array([-0.06, 0.0])))
    # vertices
    c0 = V.P([0, 0, 0])
    for n in names:
        p = V.P(VERT[n])
        u = (p - c0) / np.linalg.norm(p - c0)
        b.append("\\node[dot, fill=ndRed, minimum size=3.6pt] at %s {};" % N.pt(p))
        b.append("\\node[font=\\small] at %s {$\\%s$};" % (N.pt(p + 0.32 * u), n.replace("^", "^{") + "}"))
    e3 = V.P([0, 0, 1])
    b.append("\\node[dot, fill=ndBlue, minimum size=3.4pt] at %s {};" % N.pt(e3))
    return b, V, c0


def notes(b, V, c0):
    e3 = V.P([0, 0, 1])
    b.append("\\node[note, anchor=south east] (cc) at %s {$\\tfrac12(\\ket{00}\\bra{00}+\\ket{11}\\bra{11})$:\\\\ classically correlated};"
             % N.pt(e3 + np.array([-0.35, 0.6])))
    b.append("\\draw[pointer] (cc.south east) to[bend left=12] %s;" % N.pt(e3 + np.array([-0.04, 0.05])))
    w3 = V.P(-np.ones(3) / 3)
    b.append("\\node[note, anchor=north west, text=ndRed] (w) at %s {Werner states $t=-p(1,1,1)$:\\\\ separable for $p\\le\\frac13$};"
             % N.pt(w3 + np.array([1.0, -1.15])))
    b.append("\\draw[pointer] (w.north west) to[bend left=12] %s;" % N.pt(w3 + np.array([0.05, -0.05])))
    # separable middle, entangled corners
    sep = V.P(np.array([-1.0, 1.0, -1.0]) / 3)                      # centre of the middle triangle of the face opposite Phi^+
    b.append("\\node[note, anchor=east, text=ndBlue] (s) at %s {separable:\\\\ $\\abs{t_1}+\\abs{t_2}+\\abs{t_3}\\le1$,\\\\ the octahedron};"
             % N.pt(V.P(VERT["Psi^+"]) + np.array([-0.45, 1.1])))
    b.append("\\draw[pointer] (s.east) to[bend left=12] %s;" % N.pt(sep))
    corner = V.P(0.8 * VERT["Phi^+"] + 0.1 * VERT["Psi^-"] + 0.1 * VERT["Phi^-"])
    b.append("\\node[note, anchor=west, text=ndRed!80!ndInk] (en) at %s {entangled:\\\\ the four corners};"
             % N.pt(V.P(VERT["Phi^+"]) + np.array([0.2, -1.1])))
    b.append("\\draw[pointer] (en.west) to[bend left=12] %s;" % N.pt(corner))
    return b


if __name__ == "__main__":
    b, V, c0 = main()
    N.write("ch07_bell", notes(b, V, c0))
    print("written figures/ch07_bell.tex")
