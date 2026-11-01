"""ch08_choi.py -- the Choi matrix, the Stinespring dilation, and the Kraus rank.

The first three functions are shown in the chapter: the Choi matrix
J = sum_ij |i><j| (x) Phi(|i><j|) of a linear map Phi (which acts on the
second factor), the test CP <=> J >= 0, TP <=> Tr_2 J = Id, and the Kraus
operators read off the eigenvectors of J.

The figure, in the graphical calculus of Chapter 7: (a) J is the channel
applied to one half of the unnormalised maximally entangled vector
sum_i |ii>, a cup; trace preservation says that discarding the output of
the channel is the same as discarding its input. (b) Every channel is an
isometry V into the system and an environment, followed by discarding the
environment, Phi(rho) = Tr_E V rho V^dagger with V = sum_k K_k (x) |k>; a
unitary on the environment leg leaves Phi unchanged.  (c) The eigenvalues of
J/N for five qubit channels, and for the transposition, which is not one:
the number of non-zero eigenvalues is the least number of Kraus operators.
Writes figures/ch08_choi.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N


def choi(channel, N):
    """Choi matrix J = sum_ij |i><j| (x) Phi(|i><j|); the channel acts on the second factor."""
    J = np.zeros((N * N, N * N), dtype=complex)
    for i in range(N):
        for j in range(N):
            E = np.zeros((N, N), dtype=complex)
            E[i, j] = 1.0
            J += np.kron(E, channel(E))
    return J


def is_cptp(J, N, tol=1e-10):
    """CP iff J >= 0; TP iff the partial trace over the output factor is the identity."""
    cp = np.linalg.eigvalsh(J).min() >= -tol
    tr_out = np.einsum("iaja->ij", J.reshape(N, N, N, N))
    return cp and np.allclose(tr_out, np.eye(N), atol=tol)


def kraus_from_choi(J, N, tol=1e-12):
    """Kraus operators: eigenvectors of J, reshaped to N x N and scaled by sqrt(eigenvalue)."""
    w, V = np.linalg.eigh(J)
    return [np.sqrt(wk) * V[:, k].reshape(N, N).T for k, wk in enumerate(w) if wk > tol]


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [X, Y, Z]


def kraus_channel(K):
    return lambda R: sum(k @ R @ k.conj().T for k in K)


def bloch_channel(M, c):
    """The linear extension of the affine Bloch map r -> M r + c to all 2 x 2 matrices."""
    def phi(R):
        x = np.array([np.trace(P @ R) for P in PAULI])
        return 0.5 * (np.trace(R) * (I2 + sum(ci * P for ci, P in zip(c, PAULI)))
                      + sum((M @ x)[j] * PAULI[j] for j in range(3)))
    return phi


def ch05_channel(t=1.0):
    """The channel of the Chapter 5 qubit over the time t, from its affine Bloch equation."""
    import ch05_spectrum as Q
    E = expm(Q.qubit_A() * t)
    return bloch_channel(E[:3, :3], E[:3, 3])


def channels():
    g, p, lam = 0.5, 0.2, 0.5
    U = expm(-0.5j * (np.pi / 3) * X)
    return [
        ("unitary", kraus_channel([U])),
        ("amplitude damping, $\\gamma=0.5$", kraus_channel([np.diag([1, np.sqrt(1 - g)]).astype(complex),
                                                            np.sqrt(g) * np.array([[0, 1], [0, 0]], dtype=complex)])),
        ("dephasing, $p=0.2$", kraus_channel([np.sqrt(1 - p) * I2, np.sqrt(p) * Z])),
        ("depolarising, $\\lambda=0.5$", lambda R: lam * R + (1 - lam) * np.trace(R) * I2 / 2),
        ("the damped qubit, $t=1$", ch05_channel(1.0)),
        ("transposition", lambda R: R.T),
    ]


def checks():
    rng = np.random.default_rng(3)
    out = []
    for name, phi in channels():
        J = choi(phi, 2)
        w = np.sort(np.linalg.eigvalsh(J / 2))[::-1]
        rank = int(np.sum(np.abs(w) > 1e-10))
        if name == "transposition":
            assert not is_cptp(J, 2) and np.isclose(w.min(), -0.5)          # positive, but not CP
            assert np.allclose(J, np.eye(4)[[0, 2, 1, 3]])                  # J is the swap
            out.append((name, w, rank))
            continue
        assert is_cptp(J, 2), name
        K = kraus_from_choi(J, 2)
        assert len(K) == rank
        assert np.allclose(sum(k.conj().T @ k for k in K), I2)
        for _ in range(5):                                                   # the Kraus operators reproduce Phi
            R = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            assert np.allclose(kraus_channel(K)(R), phi(R))
        # Stinespring: V = sum_k K_k (x) |k>, an isometry; Phi = Tr_E V rho V^dagger
        r = len(K)
        V = sum(np.kron(K[k], np.eye(r)[:, [k]]) for k in range(r))           # (2 r) x 2
        assert np.allclose(V.conj().T @ V, I2)
        u = np.linalg.qr(rng.normal(size=(r, r)) + 1j * rng.normal(size=(r, r)))[0]
        Vu = np.kron(I2, u) @ V                                              # a unitary on the environment
        for _ in range(5):
            R = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            for W in (V, Vu):
                big = (W @ R @ W.conj().T).reshape(2, r, 2, r)
                assert np.allclose(np.einsum("akbk->ab", big), phi(R))
        L = [sum(u[l, k] * K[k] for k in range(r)) for l in range(r)]       # the same channel, other Kraus operators
        R = rng.normal(size=(2, 2))
        assert np.allclose(kraus_channel(L)(R), phi(R))
        out.append((name, w, rank))
    return out


def main():
    data = checks()
    for name, w, rank in data:
        print("%-36s rank %d, eigenvalues of J/N: %s" % (name.replace("$", ""), rank, np.round(w, 4)))

    b = []
    INK = "ndInk, line width=0.9pt"
    ENV = "ndGold!70!ndInk, line width=0.9pt"
    CH = "rounded corners=3pt, draw=ndRed, fill=ndRose, line width=0.9pt, minimum size=0.55cm, inner sep=0pt, font=\\scriptsize"
    GOLD = "draw=ndGold!70!ndInk, fill=ndGold!25, rounded corners=3pt, line width=0.9pt, minimum size=0.5cm, inner sep=1pt, font=\\scriptsize"
    GREEN = "minimum size=0.34cm, inner sep=0pt, fill=ndGreen!20, draw=ndGreen, line width=0.9pt, font=\\tiny"

    def ground(x, y):
        """Discarding a system (the trace), drawn at the end of a leg that points upwards."""
        return ["\\draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);" % (INK, x - w, y + h, x + w, y + h)
                for w, h in [(0.17, 0.0), (0.11, 0.07), (0.05, 0.14)]]

    # ---- (a) the Choi matrix: the channel on one half of the cup sum_i |ii>
    xl, xr, yc = 1.55, 2.55, 3.0
    b.append("\\node[anchor=east] at (1.15,3.75) {$J(\\Phi)\\;=$};")
    b.append("\\draw[%s] (%.3f,4.65) -- (%.3f,%.3f) arc (180:360:0.5) -- (%.3f,3.48);" % (INK, xl, xl, yc, xr))
    b.append("\\node[%s] (phi) at (%.3f,3.75) {$\\Phi$};" % (CH, xr))
    b.append("\\draw[%s] (phi.north) -- (%.3f,4.65);" % (INK, xr))
    b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,4.65) {reference};" % (xl - 0.12))
    b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,4.65) {output};" % (xr + 0.1))
    b.append("\\node[note, anchor=west] (cup) at (3.05,2.55) {the cup:\\\\ $\\sum_i\\ket{ii}$};")
    b.append("\\draw[pointer] (cup.west) to[bend left=20] (%.3f,%.3f);" % (0.5 * (xl + xr) + 0.12, yc - 0.48))
    b.append("\\node[font=\\small] at (1.95,1.95) {CP $\\iff J\\succeq0$};")
    # trace preservation: discarding after Phi is discarding
    b.append("\\node[%s] (phi2) at (0.75,0.85) {$\\Phi$};" % CH)
    b.append("\\draw[%s] (0.75,0.2) -- (phi2.south);" % INK)
    b.append("\\draw[%s] (phi2.north) -- (0.75,1.35);" % INK)
    b += ground(0.75, 1.35)
    b.append("\\node at (1.45,0.85) {$=$};")
    b.append("\\draw[%s] (2.05,0.2) -- (2.05,1.35);" % INK)
    b += ground(2.05, 1.35)
    b.append("\\node[note, anchor=west] at (2.45,0.85) {TP: discarding\\\\ after $\\Phi$ is\\\\ discarding,\\\\ $\\Tr_2J=\\Id$};")
    b.append("\\node[font=\\small] at (2.1,-0.75) {(a) the Choi matrix};")

    # ---- (b) Stinespring: an isometry, then discard the environment
    def chain(y, gauge):
        xs = dict(V=6.75, rho=7.85, Vd=8.95)
        out = ["\\node[anchor=east] at (6.05,%.3f) {$\\Phi(\\rho)\\;=$};" % y,
               "\\node[%s] (V%s) at (%.3f,%.3f) {$V$};" % (CH, gauge, xs["V"], y),
               "\\node[%s] (r%s) at (%.3f,%.3f) {$\\rho$};" % (GOLD, gauge, xs["rho"], y),
               "\\node[%s] (W%s) at (%.3f,%.3f) {$V^\\dagger$};" % (CH, gauge, xs["Vd"], y),
               "\\draw[%s] (6.15,%.3f) -- (V%s.west);" % (INK, y, gauge),
               "\\draw[%s] (V%s.east) -- (r%s.west);" % (INK, gauge, gauge),
               "\\draw[%s] (r%s.east) -- (W%s.west);" % (INK, gauge, gauge),
               "\\draw[%s] (W%s.east) -- (9.55,%.3f);" % (INK, gauge, y)]
        top = y + 0.95
        if gauge:
            out.append("\\node[%s] (u) at (%.3f,%.3f) {$u$};" % (GREEN, xs["V"], y + 0.62))
            out.append("\\node[%s] (ud) at (%.3f,%.3f) {$u^{\\dagger}$};" % (GREEN, xs["Vd"], y + 0.62))
            out.append("\\draw[%s] (V%s.north) -- (u.south);" % (ENV, gauge))
            out.append("\\draw[%s] (W%s.north) -- (ud.south);" % (ENV, gauge))
            out.append("\\draw[%s] (u.north) .. controls (%.3f,%.3f) and (%.3f,%.3f) .. (ud.north);"
                       % (ENV, xs["V"], top + 0.55, xs["Vd"], top + 0.55))
        else:
            out.append("\\draw[%s] (V%s.north) -- (%.3f,%.3f) .. controls (%.3f,%.3f) and (%.3f,%.3f) .. (%.3f,%.3f) -- (W%s.north);"
                       % (ENV, gauge, xs["V"], y + 0.55, xs["V"], top + 0.45, xs["Vd"], top + 0.45, xs["Vd"], y + 0.55, gauge))
        return out

    b += chain(3.6, "")
    b.append("\\node[font=\\scriptsize, text=ndGold!70!ndInk, anchor=south] at (7.85,4.82) {environment $E$, discarded};")
    b.append("\\node[font=\\small] at (7.85,2.72) {$V=\\sum_kK_k\\otimes\\ket k_E,\\quad V^\\dagger V=\\Id$};")
    b += chain(1.1, "g")
    b.append("\\node[note, anchor=north] at (7.85,0.52) {a unitary on the environment:\\\\ the Kraus operators $\\sum_ku_{lk}K_k$\\\\ give the same $\\Phi$};")
    b.append("\\node[font=\\small] at (7.85,-0.75) {(b) Stinespring and Kraus};")

    # ---- (c) the eigenvalues of J/N: the Kraus rank
    x0s, y0s = [10.6, 13.95], [3.75, 2.15, 0.55]
    bw, sc = 0.24, 1.0                                           # bar width, cm per unit eigenvalue
    for k, (name, w, rank) in enumerate(data):
        x0, y0 = x0s[k % 2], y0s[k // 2]
        bad = name == "transposition"
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (x0 - 0.1, y0, x0 + 4 * bw + 0.5, y0))
        for i, v in enumerate(w):
            if abs(v) < 1e-10:
                b.append("\\node[circle, fill=ndInk!40, inner sep=0pt, minimum size=1.6pt] at (%.3f,%.3f) {};"
                         % (x0 + 0.1 + (i + 0.5) * (bw + 0.08), y0))
                continue
            col = "ndRed" if v < 0 else "ndBlue"
            b.append("\\fill[%s!45, draw=%s, line width=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                     % (col, col, x0 + 0.1 + i * (bw + 0.08), y0, x0 + 0.1 + i * (bw + 0.08) + bw, y0 + sc * v))
        lab = "not CP: $-\\tfrac12$" if bad else "rank %d" % rank
        b.append("\\node[font=\\scriptsize, anchor=west, text=%s] at (%.3f,%.3f) {%s};"
                 % ("ndRed" if bad else "ndInk", x0 + 4 * (bw + 0.08) + 0.15, y0 + 0.28, lab))
        b.append("\\node[font=\\scriptsize, anchor=north west, inner sep=1pt] at (%.3f,%.3f) {%s};"
                 % (x0 - 0.12, y0 - 0.06 if not bad else y0 - 0.56, name))
    b.append("\\node[font=\\small] at (13.75,-0.75) {(c) the eigenvalues of $J/N$: the Kraus rank};")
    N.write("ch08_choi", b)
    print("written figures/ch08_choi.tex")


if __name__ == "__main__":
    main()
