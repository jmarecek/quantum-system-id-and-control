"""ch06_direction.py -- learning a Hamiltonian from its Gibbs state: direction and scale.

The driven qubit of Chapter 4 (ch04_examples), H = -(omega/2) sigma_z + (u/2) sigma_x,
with omega = 1 and u = 0.6, in thermal equilibrium at beta = 1.  Its Gibbs state
is estimated from N shots of each of sigma_x, sigma_y, sigma_z.  The
polynomial programme of arXiv:2402.05552 (after Bakshi et al.) asks for a
Hamiltonian H^ = -(omega^/2) sigma_z + (u^/2) sigma_x that satisfies two
families of constraints, for Pauli operators A, B1, B2:
  * stationarity, |Tr(A [H^, rho^])| <= eps_0.  Since [c H, rho] = 0 for every
    c, these fix the direction of (omega^, u^) only: for a qubit,
    Tr(A [H^, rho^]) = i (h^ x r^)_A with h^ = (u^, 0, -omega^), and the set is
    a band through the origin;
  * the KMS condition, |Tr(B2 Q(-beta H^ | B1) rho^) - Tr(B1 B2 rho^)| <= eps,
    in which Q(X | B1) = sum_i q_i [X, B1]_i is the nested-commutator form of
    the flat approximation Q_{3,6} of ch06_flat, so that Q(-beta H^ | B1)
    approximates e^{-beta H^} B1 e^{beta H^}, and Tr(B2 e^{-beta H} B1 e^{beta H} rho)
    = Tr(B1 B2 rho) for the Gibbs state rho of H.  (The other order,
    e^{beta H} B1 e^{-beta H}, fails by up to 2.5 here.)  These fix the
    direction and the scale together: for exact data only H itself satisfies
    them, and the set is a bounded region about the truth.
The tolerances are the 95th percentiles, over 2000 simulated data sets, of the
left-hand sides at the true H, so that the truth is admitted by 95 per cent of
data sets; the data set drawn is the first, in a fixed order of seeds, that
admits it.  The polynomial Q is applied to the superoperator ad(-beta H^) by
Clenshaw's recurrence; it agrees with the exponential to 1e-12 on the window.
  (a) N = 100 shots per Pauli; (b) N = 10^4: the band (gold), the KMS region
      (blue), their intersection (dark), and the truth.
Writes figures/ch06_direction.tex.
"""
import numpy as np
from scipy.linalg import expm
from scipy.special import iv
from skimage import measure
import needham as N

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
PAULI = [I2, X, Y, Z]
OMEGA0, U0, BETA = 1.0, 0.6, 1.0
T_, K_, L_ = 1, 3, 6                                    # Q_{3,6} of ch06_flat
COEF = {n: np.concatenate([[iv(0, T_)], 2 * iv(np.arange(1, n + 1), T_)]) for n in (2 * L_, 4 * L_, 8 * L_)}
WIN = (-0.25, 2.3, -0.45, 1.45)                         # omega^, u^ window
QUANT = 0.95


def ham(om, u):
    return -om / 2 * Z + u / 2 * X


def gibbs(H):
    R = expm(-BETA * H)
    return R / np.trace(R)


def ad(Hs):
    """Superoperators vec([Hs, A]) = (Hs x I - I x Hs^T) vec(A), row-major vec, for a stack of Hs."""
    return np.einsum("gij,kl->gikjl", Hs, I2).reshape(-1, 4, 4) - np.einsum("ij,glk->gikjl", I2, Hs).reshape(-1, 4, 4)


def q_super(L):
    """Q_{3,6}(L) = prod_i f_{2^i l}(L / k) for a stack of 4 x 4 matrices L, by Clenshaw's recurrence."""
    M = L / (K_ * T_)
    E = np.broadcast_to(np.eye(4), M.shape)
    out = E.copy()
    for c in COEF.values():
        b1 = np.zeros_like(M)
        b2 = np.zeros_like(M)
        for a in c[:0:-1]:
            b1, b2 = 2 * M @ b1 - b2 + a * E, b1
        out = out @ (M @ b1 - b2 + c[0] * E)
    return out


def kms_maps(Hs):
    """Q(-beta H | B1) for every Pauli B1 and every H of the stack: array (g, 4 [B1], 2, 2)."""
    Qs = q_super(ad(-BETA * Hs))
    return np.stack([(Qs @ B.reshape(4)).reshape(-1, 2, 2) for B in PAULI], axis=1)


def stat_violation(Hs, rho):
    C = Hs @ rho - rho @ Hs
    return np.max([np.abs(np.einsum("ij,gji->g", A, C)) for A in PAULI[1:]], axis=0)


def kms_violation(maps, rho):
    out = np.zeros(maps.shape[0])
    for i, B1 in enumerate(PAULI):
        for B2 in PAULI:
            lhs = np.einsum("ij,gjk,ki->g", B2, maps[:, i], rho)
            out = np.maximum(out, np.abs(lhs - np.trace(B1 @ B2 @ rho)))
    return out


def rho_hat(r, shots, rng):
    rh = 1.0 - 2.0 * rng.binomial(shots, (1.0 - r) / 2.0) / shots
    return (I2 + rh[0] * X + rh[1] * Y + rh[2] * Z) / 2


def clip_halfplane(poly, a, b, c):
    """Sutherland-Hodgman: keep the part of the polygon with a x + b y <= c."""
    out = []
    for i in range(len(poly)):
        p, q = np.asarray(poly[i - 1]), np.asarray(poly[i])
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fq <= 0:
            if fp > 0:
                out.append(p + (q - p) * fp / (fp - fq))
            out.append(q)
        elif fp <= 0:
            out.append(p + (q - p) * fp / (fp - fq))
    return out


def band(r, tol):
    """The stationarity set within the window: |h x r|_inf <= tol, h = (u, 0, -omega)."""
    poly = [(WIN[0], WIN[2]), (WIN[1], WIN[2]), (WIN[1], WIN[3]), (WIN[0], WIN[3])]
    rx, ry, rz = r
    for a, b in [(rx, rz), (-rx, -rz), (ry, 0.0), (-ry, 0.0), (0.0, ry), (0.0, -ry)]:   # in (omega, u)
        if abs(a) + abs(b) > 0:
            poly = clip_halfplane(poly, a, b, tol)
    return poly


def main():
    H0 = ham(OMEGA0, U0)
    rho0 = gibbs(H0)
    r0 = np.real([np.trace(rho0 @ s) for s in PAULI[1:]])
    # the KMS identity, in the order used, and the failure of the other order
    for B1 in PAULI:
        for B2 in PAULI:
            assert abs(np.trace(B2 @ expm(-BETA * H0) @ B1 @ expm(BETA * H0) @ rho0) - np.trace(B1 @ B2 @ rho0)) < 1e-12
    wrong = max(abs(np.trace(B2 @ expm(BETA * H0) @ B1 @ expm(-BETA * H0) @ rho0) - np.trace(B1 @ B2 @ rho0)) for B1 in PAULI for B2 in PAULI)
    assert wrong > 1.0
    # the grid, and Q against the exponential on it
    oms = np.linspace(WIN[0], WIN[1], 205)
    us = np.linspace(WIN[2], WIN[3], 153)
    OO, UU = np.meshgrid(oms, us)
    Hs = np.array([ham(o, u) for o, u in zip(OO.ravel(), UU.ravel())])
    maps = kms_maps(Hs)
    far = np.argmax(np.abs(OO.ravel()) + np.abs(UU.ravel()))
    exact = np.stack([expm(-BETA * Hs[far]) @ B @ expm(BETA * Hs[far]) for B in PAULI])
    assert np.max(np.abs(maps[far] - exact)) < 1e-12
    maps0 = kms_maps(H0[None])
    assert np.allclose(stat_violation(H0[None], rho0), 0) and np.allclose(kms_violation(maps0, rho0), 0, atol=1e-12)

    b = []
    report = []
    for panel, (shots, ox, title) in enumerate([(100, 0.0, "(a) $N=100$ shots per Pauli"), (10 ** 4, 8.9, "(b) $N=10^4$ shots per Pauli")]):
        rng = np.random.default_rng(100 + panel)
        sims = [rho_hat(r0, shots, rng) for _ in range(2000)]
        tol_s = np.quantile([stat_violation(H0[None], rh)[0] for rh in sims], QUANT)
        tol_k = np.quantile([kms_violation(maps0, rh)[0] for rh in sims], QUANT)
        seed = 0
        while True:
            rh = rho_hat(r0, shots, np.random.default_rng(1000 * (panel + 1) + seed))
            if stat_violation(H0[None], rh)[0] <= tol_s and kms_violation(maps0, rh)[0] <= tol_k:
                break
            seed += 1
        rhat = np.real([np.trace(rh @ s) for s in PAULI[1:]])
        S = stat_violation(Hs, rh).reshape(OO.shape) / tol_s
        Kv = kms_violation(maps, rh).reshape(OO.shape) / tol_k
        both = np.maximum(S, Kv)
        k_in, s_in, b_in = Kv <= 1, S <= 1, both <= 1
        # the KMS set is bounded, and lies inside the window
        assert not (k_in[0, :].any() or k_in[-1, :].any() or k_in[:, 0].any() or k_in[:, -1].any())
        # the stationarity set is not bounded by the physics: it holds 0.4 and 1.8 times the truth
        idx = lambda o, u: (np.argmin(np.abs(us - u)), np.argmin(np.abs(oms - o)))
        assert s_in[idx(0.4 * OMEGA0, 0.4 * U0)] and s_in[idx(1.8 * OMEGA0, 1.8 * U0)]
        assert s_in[np.argmin(np.abs(us)), np.argmin(np.abs(oms))]                  # it contains the origin
        cell = (oms[1] - oms[0]) * (us[1] - us[0])
        report.append((shots, tol_s, tol_k, seed, k_in.sum() * cell, b_in.sum() * cell,
                       OO[k_in].min(), OO[k_in].max(), UU[k_in].min(), UU[k_in].max(), rhat))
        Pl = N.Plot((WIN[0], WIN[1]), (WIN[2], WIN[3]), 6.6, 4.9, origin=(ox, 0.0))
        # band
        poly = band(rhat, tol_s)
        b.append("\\fill[ndGold, fill opacity=0.28] %s -- cycle;" % " -- ".join(N.pt(Pl.P(*p)) for p in poly))
        # KMS region and the intersection, from the grid
        to_page = lambda path: [Pl.P(np.interp(cj, np.arange(len(oms)), oms), np.interp(ci, np.arange(len(us)), us)) for ci, cj in path]
        for field, style in [(Kv, "fill=ndSky, fill opacity=0.9, draw=ndBlue, line width=0.7pt"),
                             (both, "fill=ndBlue, fill opacity=0.55, draw=ndBlue, line width=0.5pt")]:
            for path in measure.find_contours(field, 1.0):
                b.append("\\filldraw[%s] %s -- cycle;" % (style, " -- ".join(N.pt(p) for p in to_page(path))))
        # axes through the origin
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk] {$\\hat\\omega$};" % (Pl.pt(WIN[0], 0), Pl.pt(WIN[1] + 0.08, 0)))
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, text=ndInk] {$\\hat u$};" % (Pl.pt(0, WIN[2]), Pl.pt(0, WIN[3] + 0.08)))
        for v in (1, 2):
            b.append("\\draw[faint] %s -- %s node[below, font=\\scriptsize, text=ndInk] {$%d$};" % (Pl.pt(v, 0.03), Pl.pt(v, -0.03), v))
        for v in (-0.4, 0.5, 1.0):
            b.append("\\draw[faint] %s -- %s node[left, font=\\scriptsize, text=ndInk] {$%g$};" % (Pl.pt(0.03, v), Pl.pt(-0.03, v), v))
        b.append("\\draw[ndGold!70!ndInk, line width=0.4pt, dash pattern=on 2pt off 1.5pt] %s -- %s;" % (Pl.pt(0, 0), Pl.pt(WIN[1], WIN[1] * U0 / OMEGA0)))
        b.append("\\node[dot, fill=ndRed, minimum size=4.2pt] (t%d) at %s {};" % (panel, Pl.pt(OMEGA0, U0)))
        if panel == 0:
            b.append("\\node[note, anchor=north west, text=ndGold!75!ndInk] (s) at %s {stationarity: a band\\\\ through the origin,\\\\ the direction only};" % Pl.pt(1.12, -0.17))
            b.append("\\draw[pointer] (s.east) to[bend right=25] %s;" % Pl.pt(2.15, 1.25))
            b.append("\\node[note, anchor=west, text=ndBlue] (k) at %s {KMS condition:\\\\ a bounded region,\\\\ direction and scale};" % Pl.pt(1.7, 0.42))
            b.append("\\draw[pointer] (k.north west) to[bend right=15] %s;" % Pl.pt(1.6, 0.62))
            b.append("\\node[note, anchor=north west, text=ndRed] (tr) at %s {the truth\\\\ $(\\omega,u)=(1,0.6)$};" % Pl.pt(-0.2, 1.42))
            b.append("\\draw[pointer] (tr.east) to[bend left=20] (t%d);" % panel)
        else:
            b.append("\\node[note, anchor=north west] (k) at %s {both sets shrink like $N^{-1/2}$;\\\\ the band still passes through\\\\ the origin, and the bounded\\\\ region alone fixes the scale};" % Pl.pt(0.06, 1.5))
            b.append("\\draw[pointer] (k.south) to[bend right=15] %s;" % Pl.pt(0.32, 0.27))
            b.append("\\node[note, anchor=west] (bd) at %s {dark: the\\\\ intersection};" % Pl.pt(1.35, 0.3))
            b.append("\\draw[pointer] (bd.north west) to[bend left=20] %s;" % Pl.pt(OMEGA0 + 0.03, U0 - 0.06))
        b.append("\\node[font=\\small] at %s {%s};" % (N.pt(Pl.P((WIN[0] + WIN[1]) / 2, WIN[2]) + np.array([0.0, -0.75])), title))
    N.write("ch06_direction", b)
    print("written figures/ch06_direction.tex; Gibbs Bloch vector", np.round(r0, 4), "; other order fails by %.2f" % wrong)
    for (shots, ts, tk, seed, ak, ab, o0, o1, u0, u1, rhat) in report:
        print("  N = %5d: tolerances %.4f (stationarity), %.4f (KMS); seed %d; KMS area %.4f, intersection %.4f;"
              " KMS omega in [%.2f, %.2f], u in [%.2f, %.2f]; r^ = %s" % (shots, ts, tk, seed, ak, ab, o0, o1, u0, u1, np.round(rhat, 3)))


if __name__ == "__main__":
    main()
