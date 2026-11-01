"""ch10_su2.py -- penalised geodesics of SU(2): the shared computations of Chapter 10.

The example of Chapter 9 (Figures ch09:subriemann and ch09:balls): a qubit
without drift, U' = A U with A = -(i/2)(w_x sigma_x + w_y sigma_y + w_z sigma_z),
the metric g in which (1/2)sigma_x, (1/2)sigma_y, (1/2)sigma_z are orthonormal
(so that the speed is the rate of rotation), the cheap directions D the field in
the xy-plane, and the penalised metric
    g_q(A, A) = w_x^2 + w_y^2 + q w_z^2,
a right-invariant metric of a symmetric top.  Target V = exp(-i phi sigma_z/2),
phi = pi/2, on the time interval [0, 1].

All geodesics, in closed form.  Write A = A_p + A_k with A_k the z-part.  The
Euler-Arnold equation of a right-invariant metric, M' = -[M, Omega] with
Omega = U'U^{-1} and M = A_p + q A_k, keeps A_k constant and turns A_p about the
z-axis at the rate (q - 1) w_z, so that
    U(t) = exp(-t (q - 1) A_k) exp(t (A_p + q A_k)).
At t = 1 the second factor turns through Theta = sqrt(w_p^2 + q^2 w_z^2) about a
tilted axis, and U(1) is a rotation about z only if w_p = 0 (the DIRECT route,
w_z = phi + 4 pi k) or Theta = 2 pi m (a TWISTED route), when
    (q - 1) w_z = 2 pi m - phi + 4 pi k,   w_p^2 = 4 pi^2 m^2 - q^2 w_z^2 >= 0,
    L^2 = 4 pi^2 m^2 - q (q - 1) w_z^2.
Comparing all of them, the minimising geodesic is
    the direct route, L = sqrt(q) phi,                          for 1 <= q <= 2 pi/phi (= 4),
    a circle of twisted routes (m = 1, k = 0),
        w_z = (2 pi - phi)/(q - 1),  L^2 = 4 pi^2 - q (2 pi - phi)^2/(q - 1),   for q >= 2 pi/phi.
At q = 2 pi/phi the twisted circle branches off the direct route (w_p = 0 there),
and L^2 and its derivative in q are continuous: there is no jump.  The two pure
branches, sqrt(q) phi and d_inf = sqrt(phi (4 pi - phi)) = 4.156, would cross at
q = 7.0, but the minimiser leaves the direct route earlier, at q = 4.  As
q -> infinity the twisted routes tend to the sub-Riemannian geodesics of
Chapter 9: the field in the plane turns at the fixed rate 2 pi - phi = 3 pi/2
with the amplitude w_p -> 4.156, while the penalised part w_z = 3 pi/(2(q - 1))
vanishes like 1/q, so that the constraint violation int g(P_perp, P_perp) = w_z^2
vanishes like q^{-2}, faster than the a priori bound
(2 min E_inf - d_g^2)/(q - 1) = (d_inf^2 - phi^2)/(q - 1) = 14.80/(q - 1).

Checked independently: minimising the discretised penalised energy (24 constant
steps, L-BFGS with a stiff endpoint penalty, 16 random starts; scratch
computation) gives min E_q = 2.465, 4.312, 4.927, 5.460, 5.860, 7.059, 8.295 at
q = 2, 3.5, 4, 4.5, 5, 8, 32, against 2.467, 4.318, 4.935, 5.464, 5.860, 7.050,
8.278 from the formula, with w_p = 0 up to q = 4 and w_p = 1.64, 2.17 at
q = 4.5, 5 (formula 1.66, 2.19).  A lighter version is run by check_discrete().
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
PAULI = np.array([SX, SY, SZ])
PHI = np.pi / 2


def gen(w):
    """A = -(i/2) w.sigma."""
    return -0.5j * np.einsum('k,kij->ij', np.asarray(w, float), PAULI)


def target(phi=PHI):
    return expm(-0.5j * phi * SZ)


def geodesic(q, wp, wz, t, th0=0.0):
    """U(t) = exp(-t (q-1) A_k) exp(t (A_p + q A_k)) for an array of times t."""
    Ak = gen([0, 0, wz])
    Ap = gen([wp * np.cos(th0), wp * np.sin(th0), 0])
    return np.array([expm(-s * (q - 1) * Ak) @ expm(s * (Ap + q * Ak)) for s in np.atleast_1d(t)])


def field(q, wp, wz, t, th0=0.0):
    """The control (w_x, w_y, w_z)(t) = coordinates of U'U^{-1}: the plane part turned by -(q-1) w_z t."""
    t = np.atleast_1d(t)
    a = th0 - (q - 1) * wz * t
    return np.stack([wp * np.cos(a), wp * np.sin(a), wz + 0 * t], 1)


def q_branch(phi=PHI):
    return 2 * np.pi / phi


def minimiser(q, phi=PHI):
    """(w_p, w_z) of a minimising g_q-geodesic from I to exp(-i phi sigma_z/2)."""
    if q <= q_branch(phi):
        return 0.0, phi
    wz = (2 * np.pi - phi) / (q - 1)
    return np.sqrt(max(4 * np.pi ** 2 - q ** 2 * wz ** 2, 0.0)), wz


def length(q, phi=PHI):
    """d_q(I, V): the g_q-length of the minimising geodesic on [0, 1]."""
    wp, wz = minimiser(q, phi)
    return np.sqrt(wp ** 2 + q * wz ** 2)


def d_inf(phi=PHI):
    return np.sqrt(phi * (4 * np.pi - phi))


def energies(q, phi=PHI):
    """E_q(gamma_q) = d_q^2/2, E_1(gamma_q), and the violation int g(P_perp, P_perp) dt = w_z^2."""
    wp, wz = minimiser(q, phi)
    return 0.5 * (wp ** 2 + q * wz ** 2), 0.5 * (wp ** 2 + wz ** 2), wz ** 2


def bound(q, phi=PHI):
    """The a priori bound (2 min E_inf - d_g^2)/(q - 1) with d_g = phi."""
    return (d_inf(phi) ** 2 - phi ** 2) / (q - 1)


def all_geodesic_lengths(q, phi=PHI, mmax=4, kmax=3):
    """Lengths of all geodesics from I to V on [0,1] of the families above (direct and twisted)."""
    out = []
    for k in range(-kmax, kmax + 1):
        out.append(("direct", k, np.sqrt(q) * abs(phi + 4 * np.pi * k)))
        for m in range(1, mmax + 1):
            wz = (2 * np.pi * m - phi + 4 * np.pi * k) / (q - 1) if q > 1 else np.inf
            wp2 = 4 * np.pi ** 2 * m ** 2 - q ** 2 * wz ** 2
            if np.isfinite(wz) and wp2 >= 0:
                out.append(("twisted", (m, k), np.sqrt(4 * np.pi ** 2 * m ** 2 - q * (q - 1) * wz ** 2)))
    return out


def rotvec(U):
    """theta n with U = exp(-i theta n.sigma/2), theta in [0, 2 pi]."""
    w = np.real(np.trace(U)) / 2
    v = np.array([-np.imag(np.trace(U @ P)) / 2 for P in PAULI])
    nv = np.linalg.norm(v)
    return 2 * np.arctan2(nv, w) * v / nv if nv > 1e-14 else np.zeros(3)


def ea_residual(q, wp, wz, t, h=1e-4):
    """|M' + [M, Omega]| / (|M| |Omega|) along the closed form (right-invariant Euler-Arnold), by central differences."""
    def omega(s):
        U = geodesic(q, wp, wz, [s - h, s, s + h])
        return (U[2] - U[0]) / (2 * h) @ np.linalg.inv(U[1])
    def mom(O):
        w = np.array([np.real(np.trace(O @ (1j * P))) for P in PAULI])
        return gen([w[0], w[1], q * w[2]])
    O = omega(t)
    M = mom(O)
    dM = (mom(omega(t + h)) - mom(omega(t - h))) / (2 * h)
    return np.linalg.norm(dM + (M @ O - O @ M)) / (np.linalg.norm(M) * np.linalg.norm(O))


def _step(w, dt):
    n = np.linalg.norm(w)
    if n < 1e-14:
        return I2 - 0.5j * dt * np.einsum('k,kij->ij', w, PAULI)
    return np.cos(n * dt / 2) * I2 - 1j * np.sin(n * dt / 2) * np.einsum('k,kij->ij', w, PAULI) / n


def check_discrete(q, phi=PHI, N=16, starts=6, mu=2e4, seed=0):
    """Least discretised penalised energy over N constant steps (L-BFGS, stiff endpoint penalty)."""
    V = target(phi)
    G = np.array([1.0, 1.0, q])
    dt = 1.0 / N

    def obj(x):
        W = x.reshape(N, 3)
        Us = [_step(w, dt) for w in W]
        fw = [I2]
        for U in Us:
            fw.append(U @ fw[-1])
        bw = [I2]
        for U in Us[::-1]:
            bw.append(bw[-1] @ U)
        bw = bw[::-1]
        c = 1 - np.real(np.trace(V.conj().T @ fw[-1])) / 2
        g = W * G * dt
        for k in range(N):
            for j in range(3):
                e = np.zeros(3)
                e[j] = 1e-7
                dU = (_step(W[k] + e, dt) - _step(W[k] - e, dt)) / 2e-7
                g[k, j] -= mu * np.real(np.trace(V.conj().T @ bw[k + 1] @ dU @ fw[k])) / 2
        return 0.5 * np.sum(W ** 2 * G) * dt + mu * c, g.ravel()

    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(starts):
        r = minimize(obj, rng.normal(0, 3, 3 * N), jac=True, method='L-BFGS-B',
                     options={'maxiter': 4000, 'ftol': 1e-15, 'gtol': 1e-10})
        W = r.x.reshape(N, 3)
        U = I2
        for w in W:
            U = _step(w, dt) @ U
        if 1 - np.real(np.trace(V.conj().T @ U)) / 2 < 1e-5:
            best = min(best, 0.5 * np.sum(W ** 2 * G) * dt)
    return best


def checks(discrete=False):
    V = target()
    # the closed forms reach V and solve the Euler-Arnold equation
    for q in (1.0, 2.0, 4.0, 4.5, 8.0, 64.0, 1024.0):
        wp, wz = minimiser(q)
        U = geodesic(q, wp, wz, [1.0], th0=0.7)[0]
        assert np.allclose(U, V, atol=1e-12)
        assert abs(np.sqrt(wp ** 2 + q * wz ** 2) - length(q)) < 1e-12
        if q > 1:
            assert ea_residual(q, wp, wz, 0.4) < 1e-4
    # the minimiser is the shortest of all geodesics from I to V
    for q in np.concatenate([np.linspace(1.01, 4.0, 30), np.linspace(4.0, 40, 60), [100, 1e3, 1e4]]):
        Ls = [L for _, _, L in all_geodesic_lengths(q)]
        assert abs(min(Ls) - length(q)) < 1e-9
    # no twisted route exists below q = 2 pi/phi with m = 1, k = 0; it branches off the direct route there
    assert q_branch() == 4.0 and minimiser(4.0)[0] == 0.0 and minimiser(4.0 + 1e-9)[0] < 1e-3
    # L^2 and its q-derivative are continuous at the branch point (no jump)
    h = 1e-6
    l2 = lambda q: length(q) ** 2
    assert abs(l2(4 - h) - l2(4 + h)) < 1e-5
    assert abs((l2(4) - l2(4 - h)) / h - (l2(4 + h) - l2(4)) / h) < 1e-4
    # the pure branches would cross at q = 7.0
    assert abs((d_inf() / PHI) ** 2 - 7.0) < 0.01
    # limits: d_inf = 4.156, min E_inf = 8.636, bound constant 14.80
    assert abs(d_inf() - 4.156) < 1e-3 and abs(d_inf() ** 2 / 2 - 8.636) < 1e-3
    assert abs(d_inf() ** 2 - PHI ** 2 - 14.80) < 0.01
    assert abs(length(1e8) - d_inf()) < 1e-6
    # energies increase; violation within the bound, at most half of it, and q * violation -> 0
    qs = np.geomspace(2, 1e5, 400)
    Eq = np.array([energies(q)[0] for q in qs])
    E1 = np.array([energies(q)[1] for q in qs])
    viol = np.array([energies(q)[2] for q in qs])
    assert np.all(np.diff(Eq) >= -1e-12) and np.all(E1 <= Eq + 1e-12)
    assert np.all(viol <= bound(qs) + 1e-12) and (viol / bound(qs)).max() <= 0.5 + 1e-12
    assert abs(energies(4.0)[2] / bound(4.0) - 0.5) < 1e-12           # the ratio is largest, 1/2, at the branch point
    assert (qs * viol)[-1] < 1e-3 and abs(viol[-1] * qs[-1] ** 2 / (2 * np.pi - PHI) ** 2 - 1) < 1e-3
    # the field turns at the rate 2 pi - phi for every q above the branch point
    for q in (5.0, 64.0):
        wp, wz = minimiser(q)
        assert abs((q - 1) * wz - (2 * np.pi - PHI)) < 1e-12
    if discrete:
        for q in (3.0, 5.0):
            E = check_discrete(q)
            assert abs(E / energies(q)[0] - 1) < 0.01, (q, E, energies(q)[0])


if __name__ == "__main__":
    checks(discrete=True)
    for q in (1, 2, 4, 4.5, 5, 7, 8, 16, 64, 256, 1024):
        wp, wz = minimiser(q)
        print("q=%7.1f  d_q=%.4f  w_p=%.4f  w_z=%.4f  E_q=%.4f  E_1=%.4f  viol=%.5f  bound=%.4f"
              % (q, length(q), wp, wz, *energies(q), bound(q) if q > 1 else np.inf))
