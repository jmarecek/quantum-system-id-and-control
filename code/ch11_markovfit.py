"""ch11_markovfit.py -- the certified best Markovian model of a qubit record.

Shared by ch11_multiplicative.py, ch11_crosstalk.py and ch11_leakage.py.
Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi), with cvxpy and
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

A record is, for each of a few prepared states, the vector y = (c, x) at the
equally spaced times t_0, ..., t_{2K} (spacing h), where c = Tr rho is the
probability of finding the system in the qubit at all and x_j = Tr(sigma_j rho)
the coherence vector.  A Markovian model of the qubit is a generator

    ydot = G(p) y,     G = [[-k0, -k^T], [beta - k, A - k0 I]],

where (A, beta) are given by the 12 GKSL unknowns (theta, R, S) of
ch11_physical.py (through its matrix M), and (k0, k) are the leak operator
K = k0 I + k.sigma of the extra term -(1/2){K, rho}, which removes probability
from the qubit.  Without leakage (k0 = 0, k = 0, c = 1) this is xdot = A x + beta.
Cumulative Simpson quadrature of the integral form,

    y(t_2m) - y(t_0) = sum_{j<m} (h/3) G (y_2j + 4 y_2j+1 + y_2j+2),

is linear in p, and a semigroup record satisfies it exactly with a generator
that differs from the true one by O(h^4).  The misfit of a record is

    rho = min (1/n) ||D p - d||^2   subject to   [[R, -S], [S, R]] >= 0   (gamma >= 0)
                                   and         [[k0, k^T], [k, k0 I]] >= 0  (K >= 0),

the mean square over the n rows.  With sliding windows (design(window=w))
the rows are y(t_2(m+w)) - y(t_2m) instead; the scripts use windows of half a
Rabi period, since over a whole period the oscillating part of the integrand
vanishes and with it the information about damping.

pysparsepmi.pmi_optimize bounds the misfit from below (a quadratic in the
unknowns, written in whitened unknowns w = Rt (p - p_LS) with D = Q Rt, under
one or two polynomial matrix inequalities, relaxation order 1, which is exact
since the problem is convex), and the first-order pseudo-moments give the
minimiser.  fit() checks every value: the misfit of the minimiser made exactly
feasible by clipping negative eigenvalues (an upper bound) must agree with the
certified value, and so must, with check=True, a direct cvxpy solve and, with
check=2, a local search over factored Kossakowski matrices gamma = B B^dag.
Also here: lindbladian() and shots(), the simulated three-outcome read-out.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvxpy as cp

try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp

import ch11_physical as PH

CARDINAL = [np.array(s, float) for s in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]


def lindbladian(H, Ls):
    """The master-equation superoperator on row-major vec(rho): vec(A rho B) = kron(A, B^T) vec(rho)."""
    d = H.shape[0]
    Id = np.eye(d)
    L = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    for A in Ls:
        AdA = A.conj().T @ A
        L = L + np.kron(A, A.conj()) - 0.5 * (np.kron(AdA, Id) + np.kron(Id, AdA.T))
    return L


def generator(p):
    """The 4x4 matrix G(p) acting on y = (c, x); p has 12 entries, or 16 with the leak (k0, k)."""
    v = PH.M @ p[:12]
    G = np.zeros((4, 4))
    G[1:, 1:], G[1:, 0] = v[:9].reshape(3, 3), v[9:]
    if len(p) > 12:
        k0, k = p[12], p[13:16]
        G[0, 0] -= k0
        G[0, 1:] -= k
        G[1:, 0] -= k
        G[1:, 1:] -= k0 * np.eye(3)
    return G


def design(Ys, h, leak=False, window=None):
    """Rows (D, d) of the integral form, by Simpson's rule, for records Ys[s][i] = y(t_i).

    window=None: y(t_2m) - y(t_0) for m = 1, ..., K (one integral from the start);
    window=w:    y(t_2(m+w)) - y(t_2m) for m = 0, ..., K - w (sliding windows of 2w steps)."""
    npar = 16 if leak else 12
    basis = [generator(e) for e in np.eye(npar)]
    comps = slice(0, 4) if leak else slice(1, 4)
    Ds, ds = [], []
    for Y in Ys:
        Y = np.asarray(Y, float)
        S = Y[0:-2:2] + 4 * Y[1:-1:2] + Y[2::2]                       # Simpson sums of the pairs
        blocks = np.stack([(h / 3) * S @ Gk.T for Gk in basis], -1)    # (pairs, 4, npar)
        cum = np.concatenate([np.zeros((1,) + blocks.shape[1:]), np.cumsum(blocks, 0)])
        Ye = Y[0::2]
        if window is None:
            rows = cum[1:] - cum[0]
            diff = Ye[1:] - Ye[0]
        else:
            rows = cum[window:] - cum[:-window]
            diff = Ye[window:] - Ye[:-window]
        Ds.append(rows[:, comps, :].reshape(-1, npar))
        ds.append(diff[:, comps].ravel())
    return np.vstack(Ds), np.concatenate(ds)


def _pmis(w, zero, leak):
    """gamma >= 0 and K >= 0 for the unknowns given as the polynomials w."""
    R = [[w[3], w[4], w[5]], [w[4], w[6], w[7]], [w[5], w[7], w[8]]]
    S = [[zero, w[9], w[10]], [zero - w[9], zero, w[11]], [zero - w[10], zero - w[11], zero]]
    K = [[None] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            K[i][j] = K[i + 3][j + 3] = R[i][j]
            K[i + 3][j] = S[i][j]
            K[i][j + 3] = zero - S[i][j]
    out = [psp.PolyMatrix(K, nvars=len(w))]
    if leak:
        k0, k = w[12], w[13:16]
        arrow = [[k0] + list(k)] + [[k[i]] + [k0 if j == i else zero for j in range(3)] for i in range(3)]
        out.append(psp.PolyMatrix(arrow, nvars=len(w)))
    return out


def feasible(p):
    """The nearest point with gamma >= 0 and K >= 0 in the obvious sense (eigenvalues clipped)."""
    q = np.array(p, float)
    g = PH.kossakowski(q)
    w, V = np.linalg.eigh(g)
    g = (V * np.clip(w, 0, None)) @ V.conj().T
    for kk, (l, m) in enumerate(PH.PAIRS):
        q[3 + kk] = np.real(g[l, m])
    for kk, (l, m) in enumerate(PH.ANTI):
        q[9 + kk] = np.imag(g[l, m])
    if len(q) > 12:
        k0, k = q[12], q[13:16]
        nk = np.linalg.norm(k)
        if nk > k0:                                   # project (k0, k) onto the cone k0 >= |k|
            a = max((k0 + nk) / 2, 0.0)
            q[12], q[13:16] = a, (a / nk) * k if nk > 0 else k
    return q


def direct(Rt, p_ls, leak=False):
    """The same convex problem solved directly by cvxpy, as a check: min ||w||^2, p = p_LS + Rt^-1 w."""
    Z = cp.Variable((6, 6), symmetric=True)
    th = cp.Variable(3)
    R, S = Z[:3, :3], Z[3:, :3]
    par = [th] + [R[l, m] for l, m in PH.PAIRS] + [S[m, l] for l, m in PH.ANTI]
    cons = [Z >> 0, Z[:3, :3] == Z[3:, 3:], S + S.T == 0]
    if leak:
        k0, k = cp.Variable(1), cp.Variable(3)
        par += [k0, k]
        cons.append(cp.norm(k) <= k0)
    w = cp.Variable(len(p_ls))
    cons.append(cp.hstack(par) == p_ls + np.linalg.solve(Rt, np.eye(len(p_ls))) @ w)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(w)), cons)
    prob.solve(solver="SCS", eps=1e-10, max_iters=200000)
    return prob.value


def factored(Rt, p_ls, starts=4, seed=0):
    """A third value, by local search: gamma = B B^dag (and K = C C^dag) with B, C free, from a few starts."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    npar = len(p_ls)

    def par(x):
        B = (x[3:12] + 1j * x[12:21]).reshape(3, 3)
        g = B @ B.conj().T
        p = [x[:3], [np.real(g[l, m]) for l, m in PH.PAIRS], [np.imag(g[l, m]) for l, m in PH.ANTI]]
        if npar > 12:
            C = np.array([[x[21], 0], [x[22] + 1j * x[23], x[24]]])
            K = C @ C.conj().T                         # K = k0 I + k.sigma
            p.append([np.real(np.trace(K)) / 2, np.real(K[0, 1]), -np.imag(K[0, 1]), np.real(K[0, 0] - K[1, 1]) / 2])
        return np.concatenate(p)

    obj = lambda x: float(np.sum((Rt @ (par(x) - p_ls)) ** 2))
    best = np.inf
    for _ in range(starts):
        x0 = np.concatenate([p_ls[:3], 0.1 * rng.standard_normal(18 + (4 if npar > 12 else 0))])
        best = min(best, minimize(obj, x0, method="BFGS", options=dict(gtol=1e-13, maxiter=20000)).fun)
    return best


def fit(D, d, leak=False, check=True):
    """Certified mean-square misfit of the best Markovian model, with its parameters and the checks.

    ||D p - d||^2 = r_LS + ||w||^2 with D = Q Rt and w = Rt (p - p_LS): the excess over the
    least-squares residual r_LS is minimised over the physical set in the whitened unknowns w.
    Checks: the minimiser, made exactly feasible, gives an upper bound that must agree with the
    certified lower bound; with check=True, so must a direct cvxpy solve and, with check=2, a
    local search over factored Kossakowski matrices."""
    n, npar = D.shape
    Q, Rt = np.linalg.qr(D)
    z = Q.T @ d
    p_ls = np.linalg.solve(Rt, z)
    Ri = np.linalg.solve(Rt, np.eye(npar))
    perp = float(np.sum((d - Q @ z) ** 2))           # the least-squares residual
    v = psp.polyvar(npar)
    zero = psp.Polynomial.zero(npar)
    obj, w = zero, []
    for i in range(npar):
        obj = obj + v[i] * v[i]
    for k in range(npar):
        e = zero + float(p_ls[k])
        for i in range(npar):
            if abs(Ri[k, i]) > 1e-15 * np.abs(Ri).max():
                e = e + v[i] * Ri[k, i]
        w.append(e)
    res = psp.pmi_optimize([[obj]], ineqs=_pmis(w, zero, leak), order=1,
                           solver="SCS", eps=1e-9, max_iters=100000)
    assert res.status == "optimal", res.status
    q = feasible(p_ls + Ri @ np.array(res.minimizer(), float))
    up = float(np.sum((Rt @ (q - p_ls)) ** 2))
    tol = 2e-3 * up + 2e-9
    assert res.value <= up + tol and up - res.value <= tol, (res.value, up)
    out = dict(value=(perp + res.value) / n, excess=res.value / n, status=res.status, p=q, ls=perp / n,
               upper=(perp + up) / n, n=n, G=generator(q))
    if check:
        ref = direct(Rt, p_ls, leak)
        out["direct"] = (perp + ref) / n
        assert abs(res.value - ref) <= tol, (res.value, ref, up)
    if check == 2:
        ref = factored(Rt, p_ls)
        out["factored"] = (perp + ref) / n
        assert ref >= res.value - tol and ref - up <= tol, (res.value, ref, up)
    return out


def shots(Ys, N, rng):
    """Estimates of y from N shots per measurement basis and time: outcomes +1, -1, or 'outside the qubit'."""
    out = []
    for Y in Ys:
        Y = np.asarray(Y, float)
        E = Y.copy()
        cs = []
        for j in range(1, 4):
            pp = np.clip((Y[1:, 0] + Y[1:, j]) / 2, 0, 1)
            pm = np.clip((Y[1:, 0] - Y[1:, j]) / 2, 0, 1)
            npl = rng.binomial(N, pp)
            nmi = rng.binomial(N - npl, np.clip(pm / np.maximum(1 - pp, 1e-300), 0, 1))
            E[1:, j] = (npl - nmi) / N
            cs.append((npl + nmi) / N)
        E[1:, 0] = np.mean(cs, 0)
        out.append(E)                                # the prepared state y(t_0) is known exactly
    return out


def checks():
    # a semigroup record (the qubit of Chapter 5 under a drive) is fitted exactly, with the right rates
    from scipy.linalg import expm
    p = np.zeros(16)
    p[0], p[2] = 1.2, -0.7                           # Hamiltonian 0.6 sigma_x - 0.35 sigma_z, in theta units
    g1, gphi = 0.2, 0.4                              # relaxation to |0> and dephasing
    p[3] = p[6] = g1 / 2                             # amplitude damping: R_xx = R_yy = g1/2, S_xy = -g1/2
    p[9] = -g1 / 2
    p[8] = gphi                                      # dephasing (g/2)(Z rho Z - rho) has gamma_zz = g
    p[12], p[15] = 0.03, -0.02                       # a little leakage, more from |1> than from |0>
    G = generator(p)
    ts = np.linspace(0, 6, 61)
    Ys = [[np.concatenate([[1.0], s]) @ expm(G * t).T for t in ts] for s in CARDINAL]
    for leak in (False, True):
        if not leak:
            Gq = generator(p[:12])
            Ys0 = [[np.concatenate([[1.0], s]) @ expm(Gq * t).T for t in ts] for s in CARDINAL]
            D, d = design(Ys0, ts[1], leak)
            ref = Gq
        else:
            D, d = design(Ys, ts[1], leak)
            ref = G
        out = fit(D, d, leak, check=2)
        assert out["value"] < 1e-10 and out["ls"] < 1e-20 and np.abs(out["G"] - ref).max() < 1e-4, (leak, out["value"], out["ls"], np.abs(out["G"] - ref).max())
    # a record that gains probability cannot be fitted with K >= 0
    Ys2 = [[np.concatenate([[1.0], s]) @ expm(-generator(np.concatenate([np.zeros(12), [0.05, 0, 0, 0]])) * t).T for t in ts] for s in CARDINAL]
    out = fit(*design(Ys2, ts[1], True), leak=True, check=2)
    assert out["value"] > 1e-6


if __name__ == "__main__":
    checks()
    print("ch11_markovfit: checks passed")
