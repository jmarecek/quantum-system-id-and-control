"""ch07_tn_ladder.py -- generators are cheap, unitaries are dear.

(a) A cartoon of a Lie group (a sphere) and its Lie algebra (the tangent plane
    at the identity).  Following Balada Gaggioli and Marecek (PRA 112, 062612,
    2025), QCPOP compares the truncated Magnus generator Omega^(n)(T, x) with the
    generator i Theta of the target in the flat algebra, by the cost
    ||Omega^(n) - i Theta||_F^2, instead of comparing the unitaries on the
    curved group.  The exponential map carries both to the group; a truncated
    series for the exponential (Chebyshev, as in the paper, or Taylor, as in the
    cartoon) lands off the group, as the truncated Dyson series of Chapter 2.
    Checked on the controlled Ising chain with N = 4: exp(Omega) is unitary, the
    Chebyshev truncations of degree p = 1..4 are not.
(b) The cost Tr(X^dagger X), X = Omega - i Theta, as a closed network: the
    MPO of X and its complex conjugate, joined site by site by two rungs (the
    legs s and s'), contracted by a left-to-right sweep that keeps an R x R
    environment.  The work is linear in N.  Checked against the dense trace for
    N = 6 with Omega = a H_0 + b H_c + c [H_0, H_c] and Theta = -(pi/4) sum sz sz.
Writes figures/ch07_tn_ladder.tex.
"""
import math
import numpy as np
from scipy.linalg import expm
from scipy.special import jv
import needham as N
import ch07_tn_automaton as M


def chebyshev_exp(Om, p):
    """The paper's approximation exp(Omega) ~ J_0(1) I + 2 sum_{i=1}^p J_i(1) T_i, T_{i+1} = 2 Omega T_i + T_{i-1}."""
    I = np.eye(Om.shape[0], dtype=complex)
    T = [I, Om]
    for i in range(1, p):
        T.append(2 * Om @ T[i] + T[i - 1])
    return jv(0, 1.0) * I + 2 * sum(jv(i, 1.0) * T[i] for i in range(1, p + 1))


def sweep_norm2(X):
    """Tr(X^dagger X) for an MPO X, by a left-to-right sweep over an R x R environment."""
    E = np.ones((1, 1), dtype=complex)
    for W in X:
        E = np.einsum("ac,abst,cdst->bd", E, W.conj(), W)
    return E[0, 0].real, max(W.shape[1] for W in X[:-1])


def checks():
    # (a) unitarity of exp(Omega) and of its truncations, for the paper's first-order Magnus generator
    n = 4
    D0, Dc = M.dense_chain(n)
    T = 1.0 / (2 * n)
    a, b = -1j * T, -1j * (T - T ** 2 / 2 + T ** 3 / 3)                # u(t) = 1 - t + t^2
    Om = a * D0 + b * Dc
    assert np.allclose(Om, -Om.conj().T)
    U = expm(Om)
    I = np.eye(2 ** n)
    assert np.allclose(U.conj().T @ U, I)
    errs = []
    for p in (1, 2, 3, 4):
        Up = chebyshev_exp(Om, p)
        errs.append((np.linalg.norm(Up.conj().T @ Up - I, 2), np.linalg.norm(Up - U, 2)))
    assert all(e[0] > 1e-12 for e in errs) and all(errs[i + 1][1] < errs[i][1] for i in range(3))
    print("N = 4, Magnus order 1, T = 1/(2N): ||U_p^dag U_p - I|| and ||U_p - exp(Omega)|| for p = 1..4:")
    for p, e in zip((1, 2, 3, 4), errs):
        print("   p = %d: %.2e, %.2e" % (p, e[0], e[1]))
    # (b) the closed ladder against the dense trace
    n = 6
    H0, Hc = M.ising_mpo(n), M.control_mpo(n)
    D0, Dc = M.dense_chain(n)
    a, b, c = -0.31j, -0.17j, 0.05
    X = M.mpo_sum(M.mpo_sum(M.mpo_scale(H0, a + 1j * np.pi / 4), M.mpo_scale(Hc, b)),
                  M.mpo_sum(M.mpo_scale(M.mpo_product(H0, Hc), c), M.mpo_scale(M.mpo_product(Hc, H0), -c)))
    Theta = -np.pi / 4 * D0
    Xd = a * D0 + b * Dc + c * (D0 @ Dc - Dc @ D0) - 1j * Theta
    assert np.allclose(M.mpo_to_dense(X), Xd) and np.allclose(Xd, -Xd.conj().T)
    val, R = sweep_norm2(X)
    assert abs(val - np.real(np.trace(Xd.conj().T @ Xd))) < 1e-9 * val
    print("N = 6: Tr(X^dag X) = %.6f by the sweep and by the dense trace; bond dimension R = %d; "
          "environment R^2 = %d numbers" % (val, R, R * R))
    return R


def core(name, pos, label):
    return "\\node[rectangle, rounded corners=3pt, minimum size=0.6cm, fill=ndRose, draw=ndRed, line width=0.9pt, font=\\scriptsize] (%s) at %s {%s};" % (name, N.pt(pos), label)


def main():
    R = checks()
    b = []
    # ---- (a) the group and its algebra
    V = N.View(az=-35.0, el=32.0, R=1.9, shift=(2.6, 0.9))
    b += V.sphere()
    s = 1.3                                                     # half-width of the drawn tangent plane
    corners = [(-s, -s, 1), (s, -s, 1), (s, s, 1), (-s, s, 1)]
    b.append("\\fill[ndGold!30, fill opacity=0.55] %s -- cycle;" % " -- ".join(V.pt(c) for c in corners))
    b.append("\\draw[ndGold!70!ndInk, line width=0.6pt] %s -- cycle;" % " -- ".join(V.pt(c) for c in corners))
    e = np.array([0.0, 0.0, 1.0])
    toward = np.array([V.d[0], V.d[1], 0.0]) / np.hypot(V.d[0], V.d[1])   # horizontal direction towards the viewer
    vO = 0.95 * (toward - 0.75 * V.e1) / np.linalg.norm(toward - 0.75 * V.e1)   # the generator Omega(T, x)
    vT = 1.1 * (toward + 0.45 * V.e1) / np.linalg.norm(toward + 0.45 * V.e1)    # the target generator i Theta

    def expmap(v, t=1.0):
        th = np.linalg.norm(v) * t
        return np.cos(th) * e + np.sin(th) * v / np.linalg.norm(v)

    def trunc(v, p):                                             # Taylor truncation of cos and sin
        th = np.linalg.norm(v)
        cs = sum((-1) ** (k // 2) * th ** k / math.factorial(k) for k in range(0, p + 1, 2))
        sn = sum((-1) ** (k // 2) * th ** k / math.factorial(k) for k in range(1, p + 1, 2))
        return cs * e + sn * v / th

    for v, col in [(vO, "ndBlue"), (vT, "ndRed")]:
        arc = np.array([expmap(v, t) for t in np.linspace(0, 1, 60)])
        b += V.curve(arc, "line width=1.1pt, %s" % col, "line width=0.6pt, %s, densely dotted" % col)
        b.append("\\draw[vec, %s] %s -- %s;" % (col, V.pt(e), V.pt(e + v)))
        b.append("\\node[dot, fill=%s] at %s {};" % (col, V.pt(expmap(v))))
        b.append("\\draw[pointer, densely dashed, bend left=25] %s to %s;" % (N.pt(V.P(e + v) + np.array([0.0, -0.08])), N.pt(V.P(expmap(v)) + np.array([0.06, 0.07]))))
    norms = [np.linalg.norm(trunc(vO, p)) for p in (1, 2, 3, 4)]
    assert abs(norms[0] - np.hypot(1, np.linalg.norm(vO))) < 1e-12 and all(abs(x - 1) > 1e-4 for x in norms)
    b.append("\\node[circle, draw=ndBlue, fill=white, inner sep=0pt, minimum size=3.8pt] at %s {};" % V.pt(trunc(vO, 2)))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % V.pt(expmap(vO)))
    b.append("\\draw[ndInk, line width=1.0pt, dash pattern=on 2pt off 1.2pt] %s -- %s;" % (V.pt(e + vO), V.pt(e + vT)))
    b.append("\\node[dot] at %s {};" % V.pt(e))
    b.append("\\node[font=\\scriptsize, anchor=south east] at %s {$\\Id$};" % N.pt(V.P(e) + np.array([-0.04, 0.03])))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=east] at %s {$\\Omega^{(n)}(T,\\mathbf x)$};" % N.pt(V.P(e + vO) + np.array([-0.2, 0.24])))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=south west] at %s {$\\imath\\Theta$};" % N.pt(V.P(e + vT) + np.array([0.03, 0.02])))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=east] at %s {$U(T,\\mathbf x)$};" % N.pt(V.P(expmap(vO)) + np.array([-0.1, 0.1])))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at %s {$U^\\star$};" % N.pt(V.P(expmap(vT)) + np.array([0.09, 0.0])))
    mid = 0.5 * (V.P(e + vO) + V.P(e + vT))
    b.append("\\node[note, anchor=south west] (cost) at %s {QCPOP compares the\\\\ generators in the flat\\\\ algebra: $\\|\\Omega^{(n)}-\\imath\\Theta\\|_F^2$};" % N.pt(np.array([3.95, 3.25])))
    b.append("\\draw[pointer] (cost.south west) to[bend right=20] %s;" % N.pt(mid + np.array([0.04, 0.05])))
    b.append("\\node[note, anchor=north west, text=ndBlue] (tr) at %s {a truncated series\\\\ for $\\ee^{\\Omega}$ lands off\\\\ the group};" % N.pt(np.array([4.55, 1.05])))
    b.append("\\draw[pointer] (tr.west) to[bend left=8] %s;" % N.pt(V.P(trunc(vO, 2)) + np.array([0.07, 0.0])))
    b.append("\\node[note, anchor=north] at %s {Lie algebra: generators,\\\\ bond dimension $b_n$};" % N.pt(np.array([0.2, 3.85])))
    b.append("\\node[note, anchor=north] at %s {Lie group: unitaries,\\\\ bond dimension $\\approx b_n^{\\,p}$};" % N.pt(np.array([2.6, -1.05])))
    b.append("\\node[font=\\small] at (2.6,-2.45) {(a) compare generators, not unitaries};")
    # ---- (b) the closed ladder Tr(X^dagger X)
    x0, dx, yt, yb = 7.6, 1.15, 2.2, 0.55
    n = 6
    b.append("\\fill[ndGold!25, rounded corners=6pt] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (x0 - 0.5, yb - 0.5, x0 + 2 * dx + 0.42, yt + 0.5))
    b.append("\\draw[ndGold!70!ndInk, line width=0.7pt, rounded corners=6pt] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (x0 - 0.5, yb - 0.5, x0 + 2 * dx + 0.42, yt + 0.5))
    for k in range(n):
        x = x0 + k * dx
        b.append(core("t%d" % k, (x, yt), "$\\bar X$"))
        b.append(core("u%d" % k, (x, yb), "$X$"))
        for off in (-0.12, 0.12):
            b.append("\\draw[ink] ([xshift=%.2fcm]t%d.south) -- ([xshift=%.2fcm]u%d.north);" % (off, k, off, k))
    for k in range(n - 1):
        b.append("\\draw[ink] (t%d.east) -- (t%d.west);" % (k, k + 1))
        b.append("\\draw[ink] (u%d.east) -- (u%d.west);" % (k, k + 1))
    b.append("\\node[font=\\scriptsize, anchor=east] at ([xshift=-0.13cm]$(t0.south)!0.5!(u0.north)$) {$s$};")
    b.append("\\node[font=\\scriptsize, anchor=west] at ([xshift=0.13cm]$(t0.south)!0.5!(u0.north)$) {$s'$};")
    b.append("\\node[font=\\scriptsize, above] at ($(t2.east)!0.5!(t3.west)$) {$R$};")
    b.append("\\node[font=\\scriptsize, below] at ($(u2.east)!0.5!(u3.west)$) {$R$};")
    b.append("\\node[font=\\small, text=ndGold!55!ndInk] at (%.2f,%.2f) {$E_3$};" % (x0 + dx, yt + 0.78))
    b.append("\\draw[->, >=stealth, line width=0.9pt, ndGold!70!ndInk] (%.2f,%.2f) -- (%.2f,%.2f) node[right, font=\\scriptsize\\itshape, text=ndInk!85] {sweep};"
             % (x0 + 2 * dx + 0.1, yt + 0.78, x0 + 3.6 * dx, yt + 0.78))
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {$X=\\Omega^{(n)}-\\imath\\Theta$ and its conjugate, joined by\\\\ the legs $s$ and $s'$: a closed network, worth\\\\ $\\operatorname{Tr}(X^\\dagger X)$. The environment $E_k$ holds $R^2$\\\\ numbers ($R=%d$ here); one site is absorbed at\\\\ a time, and the work grows linearly with $N$};" % (x0 + 2.5 * dx, yb - 0.72, R))
    b.append("\\node[font=\\small] at (%.2f,-2.45) {(b) the cost as a closed network};" % (x0 + 2.5 * dx))
    N.write("ch07_tn_ladder", b)
    print("written figures/ch07_tn_ladder.tex")


if __name__ == "__main__":
    main()
