"""ch08_pauli.py -- the tetrahedron of Pauli channels, and the Markovian ones inside it.

A unital qubit map acts on Bloch vectors as r -> M r, and after rotations
before and after it, M = diag(l1, l2, l3) (signed singular values).  Then
  * the map is positive (the ellipsoid M B lies in the ball) iff
    |l_i| <= 1: the cube [-1,1]^3;
  * it is completely positive iff |l1 +- l2| <= 1 +- l3 (Fujiwara and Algoet;
    King and Ruskai): the tetrahedron whose vertices are the Pauli unitary
    channels Id = (1,1,1), X = (1,-1,-1), Y = (-1,1,-1), Z = (-1,-1,1); the
    other four corners of the cube (product of the l_i equal to -1), such as
    the transposition T = (1,-1,1), are positive but not completely positive;
  * it is entanglement breaking iff |l1| + |l2| + |l3| <= 1 (Ruskai): the
    octahedron whose vertices are the midpoints of the edges of the tetrahedron.
A Pauli channel sum_i p_i s_i rho s_i has l1 = p0 + p1 - p2 - p3, and so on, and
its Choi state is Bell-diagonal with t = (l1, -l2, l3): the transpose in the
Choi matrix flips the sign of sigma_y.  The tetrahedron and the octahedron are
therefore those of the Bell-diagonal states, figures/ch07_bell.tex.
(b) The Pauli semigroups e^{tL}, L(rho) = sum_i g_i (s_i rho s_i - rho), g_i >= 0,
give l_j = exp(-2t(g_k + g_l)), and so fill the region
    { l in (0,1]^3 : l_i >= l_j l_k for every i },
a curved solid inside the corner of the tetrahedron at Id, bounded by the
surfaces l_i = l_j l_k.  It is not convex: A = (0.9, 0.3, 0.3) and
B = (0.3, 0.9, 0.3) lie in it, their mixture (0.6, 0.6, 0.3) does not, since
its rate g3 would be negative.
Writes figures/ch08_pauli.tex.
"""
import itertools
import numpy as np
from scipy.linalg import expm
import needham as N

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)
SIG = (X, Y, Z)
VERT = {"Id": np.array([1.0, 1, 1]), "X": np.array([1.0, -1, -1]),
        "Y": np.array([-1.0, 1, -1]), "Z": np.array([-1.0, -1, 1])}
T_CORNER = np.array([1.0, -1, 1])
A_PT, B_PT = np.array([0.9, 0.3, 0.3]), np.array([0.3, 0.9, 0.3])


def unital(lam):
    """The unital qubit map with M = diag(lam), acting on 2 x 2 matrices."""
    return lambda A: 0.5 * (np.trace(A) * I2 + sum(l * np.trace(S @ A) * S for l, S in zip(lam, SIG)))


def choi(phi):
    """J = sum_ij |i><j| (x) phi(|i><j|)."""
    J = np.zeros((4, 4), complex)
    for i, j in itertools.product(range(2), range(2)):
        E = np.zeros((2, 2), complex)
        E[i, j] = 1
        J += np.kron(E, phi(E))
    return J


def tetrahedron(lam, tol=1e-12):
    l1, l2, l3 = lam
    return abs(l1 + l2) <= 1 + l3 + tol and abs(l1 - l2) <= 1 - l3 + tol


def partial_transpose(r):
    return r.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)


def positive(lam, psis, tol=1e-9):
    """Is the image of every pure state psi (rows of psis) under the unital map diag(lam) positive?"""
    rho = np.einsum("ni,nj->nij", psis, psis.conj())
    out = 0.5 * np.trace(rho, axis1=1, axis2=2)[:, None, None] * I2
    for l, S in zip(lam, SIG):
        out = out + 0.5 * l * np.einsum("ij,nji->n", S, rho)[:, None, None] * S
    return np.linalg.eigvalsh(out).min() >= -tol


def kraus_channel(ks):
    return lambda A: sum(K @ A @ K.conj().T for K in ks)


def lambdas(phi):
    """The diagonal of M for a map that is diagonal in the Pauli basis."""
    return np.real([0.5 * np.trace(S @ phi(S)) for S in SIG])


def pauli_rates(lam):
    """The rates g with lam_j = exp(-2 (g_k + g_l)) at t = 1."""
    s = -0.5 * np.log(lam)                                 # s_j = g_k + g_l
    return np.array([(s[1] + s[2] - s[0]) / 2, (s[0] + s[2] - s[1]) / 2, (s[0] + s[1] - s[2]) / 2])


def semigroup_lambdas(g, t=1.0):
    """lam of exp(tL), L(rho) = sum g_i (s_i rho s_i - rho), by the matrix exponential of L."""
    L = sum(gi * (np.kron(S, S.conj()) - np.eye(4)) for gi, S in zip(g, SIG))   # acting on row-stacked rho
    E = expm(t * L)
    phi = lambda A: (E @ A.reshape(4)).reshape(2, 2)
    return lambdas(phi)


def markovian(lam, tol=1e-12):
    l1, l2, l3 = lam
    return min(lam) > 0 and l1 >= l2 * l3 - tol and l2 >= l1 * l3 - tol and l3 >= l1 * l2 - tol


def checks():
    rng = np.random.default_rng(11)
    # the vertices are the Pauli unitary channels; the transposition is a corner, positive but not CP
    for (name, v), S in zip(VERT.items(), (I2, X, Y, Z)):
        assert np.allclose(lambdas(kraus_channel([S])), v)
    transpose = lambda A: A.T
    assert np.allclose(lambdas(transpose), T_CORNER)
    assert np.linalg.eigvalsh(choi(transpose)).min() < -0.5
    # Pauli channels: lam from the probabilities, and the Choi state is Bell-diagonal with t = (l1, -l2, l3)
    for _ in range(50):
        p = rng.dirichlet(np.ones(4))
        phi = kraus_channel([np.sqrt(pi) * S for pi, S in zip(p, (I2, X, Y, Z))])
        lam = lambdas(phi)
        assert np.allclose(lam, [p[0] + p[1] - p[2] - p[3], p[0] - p[1] + p[2] - p[3], p[0] - p[1] - p[2] + p[3]])
        assert np.allclose(phi(I2 / 2 + 0.3 * X), unital(lam)(I2 / 2 + 0.3 * X))
        rho = choi(phi) / 2
        t = np.array([lam[0], -lam[1], lam[2]])
        assert np.allclose(rho, 0.25 * (np.eye(4) + sum(ti * np.kron(S, S) for ti, S in zip(t, SIG))))
    # CP (Choi PSD) = tetrahedron; positive = cube; entanglement breaking (Choi PPT) = octahedron
    g = np.linspace(-1, 1, 21)
    psis = rng.normal(size=(400, 2)) + 1j * rng.normal(size=(400, 2))
    psis /= np.linalg.norm(psis, axis=1, keepdims=True)
    n_cp = n_eb = n = 0
    for lam in itertools.product(g, g, g):
        lam = np.array(lam)
        n += 1
        J = choi(unital(lam))
        cp = np.linalg.eigvalsh(J).min() >= -1e-9
        assert cp == tetrahedron(lam, 1e-9)
        assert positive(lam, psis[:60])                    # every point of the cube is positive
        if cp:
            n_cp += 1
            s1 = np.abs(lam).sum()
            if abs(s1 - 1) > 1e-9:
                eb = np.linalg.eigvalsh(partial_transpose(J / 2)).min() >= -1e-12
                assert eb == (s1 < 1)
                n_eb += eb
    for lam in rng.uniform(-1.3, 1.3, size=(200, 3)):      # outside the cube some pure state is mapped outside
        if np.abs(lam).max() > 1.02:
            assert not positive(lam, psis)
    # depolarising: CP for -1/3 <= l <= 1, entanglement breaking for |l| <= 1/3; the bit flip of Chapter 3
    for l in np.linspace(-1, 1, 61):
        lam = l * np.ones(3)
        assert tetrahedron(lam, 1e-12) == (l >= -1 / 3 - 1e-12)
        if tetrahedron(lam):
            assert (np.abs(lam).sum() <= 1 + 1e-12) == (abs(l) <= 1 / 3 + 1e-12)
    pflip = 0.3
    assert np.allclose(lambdas(kraus_channel([np.sqrt(1 - pflip) * I2, np.sqrt(pflip) * X])), [1, 0.4, 0.4])
    # the Pauli semigroups: lam_j = exp(-2t(g_k + g_l)); the region l_i >= l_j l_k is g >= 0
    for _ in range(200):
        gg = rng.uniform(0, 1.5, 3)
        tt = rng.uniform(0.1, 2)
        lam = semigroup_lambdas(gg, tt)
        assert np.allclose(lam, np.exp(-2 * tt * (gg.sum() - gg)))
        assert markovian(lam) and tetrahedron(lam)
        assert np.allclose(pauli_rates(lam), gg * tt)
    for lam in rng.uniform(0.01, 1, size=(2000, 3)):
        assert markovian(lam) == (pauli_rates(lam).min() >= -1e-12)
    # non-convexity: A and B are Markovian, their mixture is not
    M = 0.5 * (A_PT + B_PT)
    rates = {k: pauli_rates(v) for k, v in (("A", A_PT), ("B", B_PT), ("mixture", M))}
    assert markovian(A_PT) and markovian(B_PT) and not markovian(M)
    assert tetrahedron(A_PT) and tetrahedron(B_PT) and tetrahedron(M)
    assert np.allclose(semigroup_lambdas(rates["A"]), A_PT)
    # the mixture of the two channels is the channel at the midpoint (the map is affine in lam)
    mix = lambda Aop: 0.5 * unital(A_PT)(Aop) + 0.5 * unital(B_PT)(Aop)
    assert np.allclose(lambdas(mix), M)
    print("grid: %d points of the cube, %d completely positive, %d of those entanglement breaking" % (n, n_cp, n_eb))
    for k, v in rates.items():
        print("rates at %s = %s: g = %s" % (k, np.round(A_PT if k == "A" else B_PT if k == "B" else M, 2), np.round(v, 4)))
    return rates


# ---------------------------------------------------------------- drawing

def edges_of(verts, faces):
    out = set()
    for f in faces:
        for i in range(len(f)):
            out.add(tuple(sorted((f[i], f[(i + 1) % len(f)]))))
    return out


def panel_a(b, V):
    d = V.d
    cube = [np.array(c, float) for c in itertools.product((-1, 1), repeat=3)]
    # visible cube faces, washed rose: positive maps, not completely positive outside the tetrahedron
    for axis in range(3):
        for s in (-1, 1):
            nrm = np.zeros(3); nrm[axis] = s
            if nrm @ d <= 0:
                continue
            others = [k for k in range(3) if k != axis]
            corners = []
            for a, c in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3); p[axis] = s; p[others[0]] = a; p[others[1]] = c
                corners.append(p)
            b.append("\\fill[ndRose, fill opacity=0.55] %s -- cycle;" % " -- ".join(V.pt(p) for p in corners))
    # cube edges: hidden when both adjacent faces face away; the front ones are drawn after the tetrahedron
    front_edges = []
    for p, q in itertools.combinations(cube, 2):
        if np.sum(p != q) != 1:
            continue
        axis = int(np.where(p != q)[0][0])
        fronts = []
        for k in range(3):
            if k == axis:
                continue
            nrm = np.zeros(3); nrm[k] = p[k]
            fronts.append(nrm @ d > 0)
        if any(fronts):
            front_edges.append((p, q))
        else:
            b.append("\\draw[line width=0.4pt, ndRed!35, densely dashed] %s -- %s;" % (V.pt(p), V.pt(q)))
    # the tetrahedron: visible faces, with the middle triangle of the octahedron
    names = list(VERT)
    vis = {}
    for n in names:
        others = [m for m in names if m != n]
        vis[n] = (-VERT[n]) @ d > 0                         # the face opposite n has outward normal -v_n
        if not vis[n]:
            continue
        tri = [VERT[m] for m in others]
        mid = [(tri[i] + tri[j]) / 2 for i, j in ((0, 1), (1, 2), (2, 0))]
        b.append("\\fill[white] %s -- %s -- %s -- cycle;" % tuple(V.pt(p) for p in tri))
        b.append("\\fill[ndSky!45] %s -- %s -- %s -- cycle;" % tuple(V.pt(p) for p in tri))
        b.append("\\fill[ndSky] %s -- %s -- %s -- cycle;" % tuple(V.pt(p) for p in mid))
    octv = [s * e for e in np.eye(3) for s in (1, -1)]
    for i, j in itertools.combinations(range(6), 2):
        a, c = octv[i], octv[j]
        if abs(a @ c) > 0.5:
            continue
        face = [n for n in names if np.isclose(-VERT[n] @ a, 1) and np.isclose(-VERT[n] @ c, 1)]
        assert len(face) == 1
        style = "line width=0.8pt, ndBlue" if vis[face[0]] else "line width=0.5pt, ndBlue!70, densely dashed"
        b.append("\\draw[%s] %s -- %s;" % (style, V.pt(a), V.pt(c)))
    for m, n in itertools.combinations(names, 2):
        adj = [f for f in names if f not in (m, n)]
        style = "ink" if (vis[adj[0]] or vis[adj[1]]) else "dashedink"
        b.append("\\draw[%s] %s -- %s;" % (style, V.pt(VERT[m]), V.pt(VERT[n])))
    for p, q in front_edges:
        b.append("\\draw[line width=0.6pt, ndRed!55] %s -- %s;" % (V.pt(p), V.pt(q)))
    # the depolarising segment (l,l,l), -1/3 <= l <= 1; its entanglement-breaking part |l| <= 1/3
    lo, hi = -np.ones(3) / 3, np.ones(3) / 3
    b.append("\\draw[ndGold!80!ndInk, line width=1.1pt] %s -- %s;" % (V.pt(hi), V.pt(VERT["Id"])))
    b.append("\\draw[ndGold!80!ndInk, line width=1.1pt, densely dashed] %s -- %s;" % (V.pt(lo), V.pt(hi)))
    b.append("\\node[dot, fill=ndGold!80!ndInk, minimum size=3.2pt] at %s {};" % V.pt(lo))
    b.append("\\node[dot, minimum size=3.4pt] at %s {};" % V.pt([0, 0, 0]))
    # complete dephasing (0,0,1) on the edge Id--Z, and the bit flip of Chapter 3 on the edge Id--X
    b.append("\\node[dot, fill=ndBlue, minimum size=3.4pt] at %s {};" % V.pt([0, 0, 1]))
    b.append("\\node[dot, fill=ndGreen, minimum size=3.8pt] at %s {};" % V.pt([1, 0.4, 0.4]))
    # vertices
    c0 = V.P([0, 0, 0])
    for n in names:
        p = V.P(VERT[n])
        u = (p - c0) / np.linalg.norm(p - c0)
        b.append("\\node[dot, fill=ndInk, minimum size=3.6pt] at %s {};" % N.pt(p))
        lab = "\\Id" if n == "Id" else n
        b.append("\\node[font=\\small] at %s {$%s$};" % (N.pt(p + 0.3 * u), lab))
    for corner in cube:
        if np.prod(corner) < 0 and any(np.sign(corner[k]) * d[k] > 0 for k in range(3)):
            b.append("\\node[dot, fill=ndRed, minimum size=3.2pt] at %s {};" % V.pt(corner))
    pT = V.P(T_CORNER)
    u = (pT - c0) / np.linalg.norm(pT - c0)
    b.append("\\node[font=\\small, text=ndRed] at %s {$T$};" % N.pt(pT + 0.3 * u))
    return c0


def panel_a_notes(b, V, c0):
    P = V.P
    # vertices: unitary channels
    b.append("\\node[note, anchor=east] (u) at %s {vertices: the unitary\\\\ channels $\\rho\\mapsto\\sigma\\rho\\sigma$};" % N.pt(P(VERT["Z"]) + np.array([-0.25, -0.75])))
    b.append("\\draw[pointer] (u.north east) to[bend left=15] %s;" % N.pt(P(VERT["Z"]) + np.array([-0.05, -0.07])))
    # the octahedron
    face_mid = (VERT["X"] + VERT["Y"] + VERT["Z"]) / 3                 # centre of the face opposite Id, on the octahedron
    b.append("\\node[note, anchor=east, text=ndBlue] (eb) at %s {entanglement\\\\ breaking: the\\\\ octahedron};" % N.pt(np.array([P(VERT["Z"])[0] - 0.12, P(VERT["Y"])[1] - 0.6])))
    b.append("\\draw[pointer] (eb.north east) to[bend right=15] %s;" % N.pt(P(0.5 * (VERT["X"] + VERT["Y"]) + 0.15 * (VERT["Z"] - 0.5 * (VERT["X"] + VERT["Y"])))))
    # positive, not CP: the corners of the cube, T the transposition
    b.append("\\node[note, anchor=west, text=ndRed!85!ndInk] (pc) at %s {the corners: positive,\\\\ not completely positive;\\\\ $T$, the transpose};" % N.pt(P(VERT["Id"]) + np.array([0.4, 0.3])))
    b.append("\\draw[pointer] (pc.south west) to[bend left=25] %s;" % N.pt(P(T_CORNER) + np.array([0.08, -0.02])))
    # depolarising, dephasing, bit flip
    b.append("\\node[note, anchor=west, text=ndGold!70!ndInk] (dp) at %s {depolarising,\\\\ $-\\tfrac13\\le\\lambda\\le1$};" % N.pt(P(VERT["Id"]) + np.array([0.35, -1.05])))
    b.append("\\draw[pointer] (dp.west) to[bend left=15] %s;" % N.pt(P(0.6 * np.ones(3)) + np.array([0.04, -0.04])))
    b.append("\\node[note, anchor=west, text=ndGreen!70!ndInk] (bf) at %s {bit flip of Chapter 3,\\\\ $p=0.3$};" % N.pt(P(VERT["X"]) + np.array([0.95, 0.15])))
    b.append("\\draw[pointer] (bf.north west) to[bend right=15] %s;" % N.pt(P(np.array([1, 0.4, 0.4])) + np.array([0.07, -0.02])))
    b.append("\\node[note, anchor=south, text=ndBlue] (cd) at %s {complete dephasing};" % N.pt(P(np.array([0, 0, 1])) + np.array([0.0, 0.4])))
    b.append("\\draw[pointer] (cd.south) -- %s;" % N.pt(P(np.array([0, 0, 1])) + np.array([0.0, 0.07])))
    b.append("\\node[font=\\small] at %s {(a) unital qubit maps $\\Bloch\\mapsto\\operatorname{diag}(\\lambda)\\,\\Bloch$};" % N.pt(np.array([P(np.zeros(3))[0] + 0.3, -3.05])))


def surface_patches(V, n=16):
    """Patches of the three curved faces l_i = l_j l_k of the Markovian region, with their outward normals."""
    us = np.linspace(0, 1, n + 1)
    patches = []
    for i in range(3):
        j, k = [m for m in range(3) if m != i]

        def S(u, v):
            p = np.zeros(3); p[i] = u * v; p[j] = u; p[k] = v
            return p
        for a in range(n):
            for c in range(n):
                quad = [S(us[a], us[c]), S(us[a + 1], us[c]), S(us[a + 1], us[c + 1]), S(us[a], us[c + 1])]
                um, vm = 0.5 * (us[a] + us[a + 1]), 0.5 * (us[c] + us[c + 1])
                nrm = np.zeros(3); nrm[i] = -1.0; nrm[j] = vm; nrm[k] = um     # -grad(l_i - l_j l_k): outward
                nrm /= np.linalg.norm(nrm)
                iso = []
                if a % 4 == 0:
                    iso.append((quad[0], quad[3]))
                if c % 4 == 0:
                    iso.append((quad[0], quad[1]))
                patches.append((np.mean(quad, axis=0) @ V.d, quad, nrm, iso))
    return patches


def section_outline(c=0.3, n=80):
    """The boundary of the section of the Markovian region by the plane l3 = c: the lines l1 = c l2 and
    l2 = c l1 from the axis, and the hyperbola l1 l2 = c between them."""
    s = np.linspace(0, 1, n)
    left = [np.array([c * t, t, c]) for t in s]                      # l1 = c l2, up to (c, 1)
    hyp = [np.array([c / t, t, c]) for t in np.linspace(1, c, n)]    # l1 l2 = c, from (c, 1) to (1, c)
    right = [np.array([t, c * t, c]) for t in s[::-1]]               # l2 = c l1, back to the axis
    pts = left + hyp[1:] + right[1:]
    for p in pts:                                                    # on the boundary of the region (or the axis)
        l1, l2, l3 = p
        assert min(p) >= 0 and l1 >= l2 * l3 - 1e-9 and l2 >= l1 * l3 - 1e-9 and l3 >= l1 * l2 - 1e-9
    return pts, hyp


def panel_b(b, V):
    d = V.d
    O = np.zeros(3)
    E = np.eye(3)
    Id = np.ones(3)
    c = A_PT[2]
    # axes
    for e, lab in zip(E, ("\\lambda_1", "\\lambda_2", "\\lambda_3")):
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[pos=1.0, anchor=%s, font=\\scriptsize, text=ndInk] {$%s$};"
                 % (V.pt(O), V.pt(1.3 * e), "south" if lab.endswith("3") else "north", lab))
    # the corner of the tetrahedron at Id: the bipyramid conv{0, e1, e2, e3, Id}; its lower half is the octahedron
    b.append("\\fill[ndSky, fill opacity=0.35] %s -- %s -- %s -- cycle;" % tuple(V.pt(e) for e in E))
    for e, f in itertools.combinations(E, 2):
        b.append("\\draw[line width=0.5pt, ndBlue!70, densely dashed] %s -- %s;" % (V.pt(e), V.pt(f)))
    # the three faces l_i = l_j l_k are saddles (hyperbolic paraboloids) on the skew quadrilaterals
    # Id - e_j - 0 - e_k; drawn translucent, with their straight rulings, solid where they face the viewer
    for i in range(3):
        j, k = [m for m in range(3) if m != i]

        def S(u, v):
            q = np.zeros(3); q[i] = u * v; q[j] = u; q[k] = v
            return q

        def facing(u, v):
            nrm = np.zeros(3); nrm[i] = -1.0; nrm[j] = v; nrm[k] = u          # outward normal
            return nrm @ d > 0
        outline = [S(1, 1), S(1, 0), S(0, 0), S(0, 1)]
        assert np.allclose(outline, [Id, E[j], O, E[k]])
        b.append("\\fill[ndGold, fill opacity=0.22] %s -- cycle;" % " -- ".join(V.pt(q) for q in outline))
        ws = np.linspace(0, 1, 41)
        for r in (0.2, 0.4, 0.6, 0.8):
            for line in ([(r, w) for w in ws], [(w, r) for w in ws]):
                for a0, a1 in zip(line[:-1], line[1:]):
                    f = facing(0.5 * (a0[0] + a1[0]), 0.5 * (a0[1] + a1[1]))
                    style = "line width=0.35pt, ndGold!60!ndInk" if f else "line width=0.25pt, ndGold!50!ndInk, opacity=0.45"
                    b.append("\\draw[%s] %s -- %s;" % (style, V.pt(S(*a0)), V.pt(S(*a1))))
    # the section by the plane l3 = c: a curvilinear triangle whose third side, the hyperbola, curves inwards
    pts, hyp = section_outline(c)
    b.append("\\fill[white, fill opacity=0.75] %s -- cycle;" % " -- ".join(V.pt(p) for p in pts))
    b.append(N.polyline([V.P(p) for p in pts] + [V.P(pts[0])], "line width=0.7pt, ndGold!50!ndInk"))
    b.append(N.polyline([V.P(p) for p in hyp], "line width=1.0pt, ndInk"))
    # the straight edges of the region: the one-parameter semigroups Id -> e_i, and the axes 0 -> e_i
    for e in E:
        b.append("\\draw[ink, line width=0.8pt] %s -- %s;" % (V.pt(Id), V.pt(e)))
        b.append("\\draw[ink, line width=0.5pt] %s -- %s;" % (V.pt(O), V.pt(e)))
    b.append("\\node[dot, minimum size=3.4pt] at %s {};" % V.pt(Id))
    b.append("\\node[font=\\small, anchor=south] at %s {$\\Id$};" % N.pt(V.P(Id) + np.array([0, 0.06])))
    for e in E:
        b.append("\\node[dot, fill=ndBlue, minimum size=3pt] at %s {};" % V.pt(e))
    # the chord from A to B, through a non-Markovian mixture
    M = 0.5 * (A_PT + B_PT)
    ts = np.linspace(0, 1, 201)
    chord = np.array([(1 - t) * A_PT + t * B_PT for t in ts])
    inside = np.array([markovian(p) for p in chord])
    for seg_in in (True, False):
        idx = np.where(inside == seg_in)[0]
        for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(seg) > 1:
                style = "ndGreen!70!ndInk, line width=1.0pt" if seg_in else "ndRed, line width=1.3pt"
                b.append(N.polyline([V.P(p) for p in chord[seg]], style))
    b.append("\\node[dot, fill=ndGreen, minimum size=4pt] at %s {};" % V.pt(A_PT))
    b.append("\\node[dot, fill=ndGreen, minimum size=4pt] at %s {};" % V.pt(B_PT))
    b.append("\\node[circle, draw=ndRed, fill=white, line width=1pt, inner sep=0pt, minimum size=4.6pt] at %s {};" % V.pt(M))
    return M


def panel_c(b, origin, size):
    """The section l3 = c in the (l1, l2) plane, magnified."""
    c = A_PT[2]
    Pl = N.Plot((0.0, 1.0), (0.0, 1.0), size, size, origin=origin)
    b += Pl.axes("$\\lambda_1$", "$\\lambda_2$", xticks=[(c, "$0.3$"), (1, "$1$")], yticks=[(c, "$0.3$"), (1, "$1$")])
    pts, hyp = section_outline(c)
    b.append("\\fill[ndGold!45!white] %s -- cycle;" % " -- ".join(Pl.pt(p[0], p[1]) for p in pts))
    b.append(N.polyline([Pl.P(p[0], p[1]) for p in pts] + [Pl.P(pts[0][0], pts[0][1])], "line width=0.6pt, ndGold!50!ndInk"))
    b.append(N.polyline([Pl.P(p[0], p[1]) for p in hyp], "line width=1.0pt, ndInk"))
    b.append("\\draw[faint, densely dotted] %s -- %s -- %s;" % (Pl.pt(0, 1), Pl.pt(1, 1), Pl.pt(1, 0)))
    ts = np.linspace(0, 1, 201)
    chord = np.array([(1 - t) * A_PT + t * B_PT for t in ts])
    inside = np.array([markovian(p) for p in chord])
    for seg_in in (True, False):
        idx = np.where(inside == seg_in)[0]
        for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(seg) > 1:
                style = "ndGreen!70!ndInk, line width=1.0pt" if seg_in else "ndRed, line width=1.3pt"
                b.append(N.polyline([Pl.P(p[0], p[1]) for p in chord[seg]], style))
    M = 0.5 * (A_PT + B_PT)
    for q, lab, anc in ((A_PT, "A", "north"), (B_PT, "B", "east")):
        b.append("\\node[dot, fill=ndGreen, minimum size=4pt] at %s {};" % Pl.pt(q[0], q[1]))
        b.append("\\node[font=\\small, anchor=%s, text=ndGreen!70!ndInk] at %s {$%s$};" % (anc, Pl.pt(q[0] + (0.0 if anc == "north" else -0.03), q[1] + (-0.04 if anc == "north" else 0.0)), lab))
    b.append("\\node[circle, draw=ndRed, fill=white, line width=1pt, inner sep=0pt, minimum size=4.6pt] at %s {};" % Pl.pt(M[0], M[1]))
    b.append("\\node[note, anchor=south west, text=ndRed] (mx) at %s {their mixture:\\\\ $g_3<0$};" % Pl.pt(0.66, 0.72))
    b.append("\\draw[pointer] (mx.south west) -- %s;" % N.pt(Pl.P(M[0], M[1]) + np.array([0.05, 0.05])))
    b.append("\\node[note, anchor=west] (hy) at %s {$\\lambda_1\\lambda_2=0.3$};" % Pl.pt(1.06, 0.5))
    b.append("\\draw[pointer] (hy.west) to[bend left=15] %s;" % N.pt(Pl.P(0.8, c / 0.8) + np.array([0.04, 0.03])))
    return Pl


def axis_view(beta, R, shift):
    """A view with the diagonal (1,1,1) upright on the page, seen from beta degrees off that axis,
    from the side of the edge e1--e2."""
    V = N.View(R=R, shift=shift)
    w = np.ones(3) / np.sqrt(3)
    h = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
    d = np.cos(np.radians(beta)) * w + np.sin(np.radians(beta)) * h
    e2 = w - (w @ d) * d
    e2 /= np.linalg.norm(e2)
    V.d, V.e2, V.e1 = d, e2, np.cross(e2, d)
    return V


def main():
    rates = checks()
    b = []
    Va = N.View(az=-60.0, el=20.0, R=1.45, shift=(0.0, 0.0))
    c0 = panel_a(b, Va)
    panel_a_notes(b, Va, c0)
    Vb = N.View(az=200.0, el=22.0, R=1.9, shift=(7.3, -0.5))
    panel_b(b, Vb)
    Pl = panel_c(b, origin=(10.0, -1.7), size=2.4)
    b.append("\\node[note, anchor=north] at %s {Pauli semigroups:\\\\ $\\lambda_i\\ge\\lambda_j\\lambda_k$ for every $i$};" % N.pt(np.array([Vb.P([0.5, 0.5, 0.5])[0], -2.05])))
    b.append("\\node[note, anchor=north] at %s {the section $\\lambda_3=0.3$};" % N.pt(Pl.P(0.5, 0.0) + np.array([0.0, -0.55])))
    b.append("\\node[font=\\small] at %s {(b) the channels $\\ee^{t\\mathcal L}$ of the Pauli semigroups};" % N.pt(np.array([9.7, -3.05])))
    N.write("ch08_pauli", b)
    print("written figures/ch08_pauli.tex")


if __name__ == "__main__":
    main()
