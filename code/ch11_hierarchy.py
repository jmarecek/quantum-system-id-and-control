"""ch11_hierarchy.py -- the hierarchy of Chapter 8, certified.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi), with cvxpy and
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

The four families of Figure ch08:families (code/ch08_families.py): the qubit
of Chapter 5 (a semigroup), Ornstein-Uhlenbeck dephasing with b = 1 and
tau = 0.5 (CP-divisible), the Pauli dynamics of Hall, Cresser, Li and
Andersson (P-divisible) and the qubit with the hidden defect, g = 1 and
kappa = 0.8 (not even P-divisible), over the window [0, 4] of that figure.
Each is given as maps Phi_k = Phi(k dt), dt = 0.08, k = 0, ..., 50, written
as 4x4 affine matrices on Bloch vectors, and its motion as the finite
differences D_k = (Phi_{k+1} - Phi_k)/dt.  (The grid keeps every sample at
least 0.03 from the zero of c(t) of the defect at t = 1.809, where Phi is
singular; with dt = 0.05 a sample falls within 0.009 of it, and SCS returns
only 'optimal_inaccurate' at eps = 1e-9 for that step.)  The distance to a class is the
least mean-square residual (1/K) sum_k ||D_k - G_k Phi_k||_F^2 with
  * semigroups:            one constant GKSL generator G_k = G (gamma >= 0);
  * CP-divisible families: a GKSL generator G_k at each step (gamma_k >= 0);
  * P-divisible families:  at each step a generator G_k = (A_k, beta_k) whose
                           flow keeps the Bloch ball, n.(A_k n + beta_k) <= 0
                           for |n| = 1, which by the S-lemma is the 4x4 PMI
                           [[lambda_k I - (A_k + A_k^T)/2, -beta_k/2],
                            [-beta_k^T/2, -lambda_k]] >= 0.
These are quadratic objectives under PMIs of size 6, 6 and 4.  Where the
forward difference D_k Phi_k^{-1} itself lies in the class (checked by the
eigenvalues of its Kossakowski matrix, or by the S-lemma matrix maximised
over lambda), the distance of that step is 0, attained; elsewhere it is
certified by pysparsepmi.pmi_optimize at order 1 (exact, the problems being
convex; SCS to eps = 1e-9, and only the status 'optimal' is accepted).  The
forward difference respects the hierarchy exactly: (Lambda - id)/dt
generates channels if the transfer map Lambda = Phi_{k+1} Phi_k^{-1} is a
channel, and positive maps if Lambda is positive, so the zeros below are
exact, not small.  The second and third problems split over the steps; for
Hall's dynamics on a coarser grid (dt = 0.16), the steps that leave the
CP-divisible families are also solved as one problem, with the steps as
variable cliques (var_cliques), which gives the sum of the steps.  (The
same joint solve for the defect, or for the P-divisible families, ends
'optimal_inaccurate' at eps = 1e-9, and is not used.)  Every number is
checked against a direct cvxpy solve of the whole problem.  Distances are
reported as the fraction sqrt(rho/rho_0) of the motion
rho_0 = (1/K) sum_k ||D_k||^2.

(a) The distances, in a table whose zeros form the staircase of Figure
    ch08:families: the qubit of Chapter 5 is at distance 0 from all three
    classes; OU dephasing at 0.37 from the semigroups and 0 from the
    others; Hall's dynamics at 0.64 and 0.19 from the first two, and 0 from
    the P-divisible families; the defect at 0.83, 0.42 and 0.41 (for an
    amplitude-damping family positivity and complete positivity of the
    transfer maps fail together, so the last two nearly agree).
(b) Where in time the last two leave their classes: the share of each
    step in the CP and P distances, accumulated over the window.  Hall's
    dynamics leaves the CP-divisible families at every step after the first
    (gamma_3 = -tanh t < 0), though the steps weigh less and less as the
    z-component of the Bloch vector, on which gamma_3 acts, dies away; the defect
    leaves both classes only while the excitation comes back, between the
    zero of c(t) at t = 1.81 and its turning point at 3.21 (shaded).
Writes figures/ch11_hierarchy.tex.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp

import cvxpy as cp
import needham as N
import ch05_spectrum as Q
import ch08_families as FAM
import ch07_defect as D
import ch11_physical as P

T_WIN, DT = 4.0, 0.08
RATE, KAPPA = 0.5, D.E.KAPPA          # long-time OU rate b^2 tau_c; defect decay rate
ROW = {(i, l): (3 * i + l if l < 3 else 9 + i) for i in range(3) for l in range(4)}   # G[i, l] in P.M's output
STATUS = []                                    # (kind, first step, steps, SCS status) of every PMI solve


def affine(M, c):
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = M, c
    return E


def ou(t, tc):
    """OU dephasing with correlation time tc and b^2 = RATE/tc."""
    e = np.exp(-RATE * (t - tc * (1 - np.exp(-t / tc))))
    return affine(np.diag([e, e, 1.0]), np.zeros(3))


def amplitude(t, g, kappa=KAPPA):
    """c(t) of the defect, for any g > 0 (cosh and sinh below kappa/4)."""
    W = np.emath.sqrt(g ** 2 - kappa ** 2 / 16 + 0j)
    if abs(W) < 1e-9:                          # critical coupling g = kappa/4
        return np.exp(-kappa * t / 4) * (1 + kappa * t / 4)
    return np.real(np.exp(-kappa * t / 4) * (np.cos(W * t) + kappa / (4 * W) * np.sin(W * t)))


def defect(t, g):
    c = amplitude(t, g)
    return affine(np.diag([c, c, c * c]), np.array([0.0, 0.0, 1 - c * c]))


def hall(t):
    return affine(np.diag(FAM.hall_lambdas(t)), np.zeros(3))


def regression(family, T, dt=DT):
    """The objective as ||G L - W||^2 + const0, with the unconstrained fit and rho_0."""
    K = int(round(T / dt))
    Ph = [family(k * dt) for k in range(K + 1)]
    Dk = [(Ph[k + 1] - Ph[k])[:3] / dt for k in range(K)]
    S = sum(Ph[k] @ Ph[k].T for k in range(K)) / K
    C = sum(Dk[k] @ Ph[k].T for k in range(K)) / K
    rho0 = sum(np.sum(d ** 2) for d in Dk) / K
    L = np.linalg.cholesky(S)
    W = C @ np.linalg.inv(L).T
    return L, W, rho0 - np.sum(W ** 2), rho0, C @ np.linalg.inv(S)


def certify_fit(L, W, const0):
    """Certified least misfit over GKSL generators, and the minimiser (theta, R, S)."""
    v = psp.polyvar(12)
    zero = psp.Polynomial.zero(12)
    resid = zero + float(const0)
    for i in range(3):
        for j in range(4):
            coef = sum(L[l, j] * P.M[ROW[(i, l)], :] for l in range(4))
            e = zero + float(-W[i, j])
            for k in range(12):
                if abs(coef[k]) > 1e-14:
                    e = e + v[k] * float(coef[k])
            resid = resid + e * e
    R = [[v[3], v[4], v[5]], [v[4], v[6], v[7]], [v[5], v[7], v[8]]]
    S = [[zero, v[9], v[10]], [zero - v[9], zero, v[11]], [zero - v[10], zero - v[11], zero]]
    K = [[None] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            K[i][j] = K[i + 3][j + 3] = R[i][j]
            K[i + 3][j] = S[i][j]
            K[i][j + 3] = zero - S[i][j]
    res = psp.pmi_optimize([[resid]], ineqs=[psp.PolyMatrix(K, nvars=12)], order=1,
                           solver="SCS", eps=1e-9, max_iters=100000)
    return res.value, res.status, np.array(res.minimizer())


def direct_fit(L, W, const0):
    """The same convex problem solved directly by cvxpy, as a check."""
    Z = cp.Variable((6, 6), symmetric=True)
    th = cp.Variable(3)
    R, S = Z[:3, :3], Z[3:, :3]
    par = cp.hstack([th] + [R[l, m] for l, m in P.PAIRS] + [S[m, l] for l, m in P.ANTI])
    ab = P.M @ par
    G = cp.vstack([cp.hstack([ab[3 * i:3 * i + 3], ab[9 + i:10 + i]]) for i in range(3)])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(G @ L - W) + const0),
                      [Z >> 0, Z[:3, :3] == Z[3:, 3:], S + S.T == 0])
    prob.solve(solver="SCS", eps=1e-10, max_iters=200000)
    return prob.value


FAMILIES = [
    ("the qubit of Chapter 5", "a semigroup", lambda t: expm(Q.qubit_A() * t)),
    ("OU dephasing", "CP-divisible", lambda t: ou(t, FAM.TAU)),
    ("Hall et al.", "P-divisible", hall),
    ("the hidden defect", "not P-divisible", lambda t: defect(t, 1.0)),
]


def steps(family, T=T_WIN, dt=DT):
    K = int(round(T / dt))
    Ph = [family(k * dt) for k in range(K + 1)]
    return [((Ph[k + 1] - Ph[k])[:3] / dt, Ph[k]) for k in range(K)]


def gksl_block(v, zero, o, n):
    """The 6x6 PMI gamma >= 0 on the parameters v[o:o+12] (layout of ch11_physical)."""
    R = [[v[o + 3], v[o + 4], v[o + 5]], [v[o + 4], v[o + 6], v[o + 7]], [v[o + 5], v[o + 7], v[o + 8]]]
    S = [[zero, v[o + 9], v[o + 10]], [zero - v[o + 9], zero, v[o + 11]], [zero - v[o + 10], zero - v[o + 11], zero]]
    K = [[None] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            K[i][j] = K[i + 3][j + 3] = R[i][j]
            K[i + 3][j] = S[i][j]
            K[i][j + 3] = zero - S[i][j]
    return psp.PolyMatrix(K, nvars=n)


def ball_block(v, zero, o, n):
    """The 4x4 PMI keeping the Bloch ball, on A = v[o:o+9], beta = v[o+9:o+12], lambda = v[o+12]."""
    G = [[v[o + 3 * i + l] if l < 3 else v[o + 9 + i] for l in range(4)] for i in range(3)]
    lam = v[o + 12]
    B = [[None] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            B[i][j] = (lam if i == j else zero) - (G[i][j] + G[j][i]) * 0.5
        B[i][3] = B[3][i] = zero - G[i][3] * 0.5
    B[3][3] = zero - lam
    return psp.PolyMatrix(B, nvars=n), G


def residual(kind, v, zero, o, Dk, Pk, G=None):
    """sum_ij (D_k - G Phi_k)_ij^2, with G through the GKSL parameters ('cp') or directly ('p')."""
    out = zero
    for i in range(3):
        for j in range(4):
            e = zero + float(-Dk[i, j])
            if kind == "cp":
                coef = sum(Pk[l, j] * P.M[ROW[(i, l)], :] for l in range(4))
                for q in range(12):
                    if abs(coef[q]) > 1e-14:
                        e = e + v[o + q] * float(coef[q])
            else:
                for l in range(4):
                    if abs(Pk[l, j]) > 1e-14:
                        e = e + G[i][l] * float(Pk[l, j])
            out = out + e * e
    return out


def member(kind, Dk, Pk, tol=1e-9):
    """Does the forward difference G = D_k Phi_k^{-1} itself lie in the cone?  Then the distance is 0, attained."""
    Gh = Dk @ np.linalg.inv(Pk)
    A, b = Gh[:, :3], Gh[:, 3]
    scale = max(1.0, np.abs(Gh).max())
    if kind == "cp":
        gam = P.kossakowski(np.linalg.solve(P.M, np.concatenate([A.ravel(), b])))
        return np.linalg.eigvalsh(gam).min() >= -tol * scale
    As = (A + A.T) / 2                         # S-lemma: max_lambda lambda_min of the 4x4 matrix >= 0
    top = np.linalg.eigvalsh(As).max()
    if top > tol * scale:
        return False
    def neg_margin(lam):
        B = np.block([[lam * np.eye(3) - As, -b[:, None] / 2], [-b[None, :] / 2, np.array([[-lam]])]])
        return -np.linalg.eigvalsh(B).min()
    r = minimize_scalar(neg_margin, bounds=(min(top, 0.0), 0.0), method="bounded", options={"xatol": 1e-13})
    return -min(r.fun, neg_margin(0.0), neg_margin(min(top, 0.0))) >= -tol * scale


def solve_pmi(st, kind, ks, joint):
    """Certified least residual over the steps ks, as one PMI problem (the steps as variable cliques if joint)."""
    nv = 12 if kind == "cp" else 13
    n = nv * len(ks)
    v = psp.polyvar(n)
    zero = psp.Polynomial.zero(n)
    obj, ineqs, cliques = zero, [], []
    for a, k in enumerate(ks):
        o = nv * a
        cliques.append(list(range(o, o + nv)))
        if kind == "cp":
            ineqs.append(gksl_block(v, zero, o, n))
            obj = obj + residual(kind, v, zero, o, *st[k])
        else:
            B, G = ball_block(v, zero, o, n)
            ineqs.append(B)
            obj = obj + residual(kind, v, zero, o, *st[k], G=G)
    res = psp.pmi_optimize([[obj]], ineqs=ineqs, order=1, var_cliques=cliques if joint else None,
                           solver="SCS", eps=1e-9, max_iters=300000)
    STATUS.append((kind, ks[0], len(ks), res.status))
    assert res.status == "optimal", (kind, ks, res.status)
    return res.value


def certify_steps(st, kind, joint=False):
    """Per-step distances: 0 where the forward difference is in the cone, else certified by pysparsepmi."""
    if joint:
        return solve_pmi(st, kind, list(range(len(st))), True)
    return np.array([0.0 if member(kind, *st[k]) else solve_pmi(st, kind, [k], False) for k in range(len(st))])


def direct_steps(st, kind):
    """The same per-step problems solved by cvxpy, all at once."""
    total, cons = 0, []
    for Dk, Pk in st:
        if kind == "cp":
            Z = cp.Variable((6, 6), symmetric=True)
            th = cp.Variable(3)
            R, S = Z[:3, :3], Z[3:, :3]
            par = cp.hstack([th] + [R[l, m] for l, m in P.PAIRS] + [S[m, l] for l, m in P.ANTI])
            ab = P.M @ par
            G = cp.vstack([cp.hstack([ab[3 * i:3 * i + 3], ab[9 + i:10 + i]]) for i in range(3)])
            cons += [Z >> 0, Z[:3, :3] == Z[3:, 3:], S + S.T == 0]
        else:
            G = cp.Variable((3, 4))
            lam = cp.Variable()
            A, b = G[:, :3], G[:, 3:]
            Bm = cp.bmat([[lam * np.eye(3) - (A + A.T) / 2, -b / 2], [-b.T / 2, cp.reshape(-lam, (1, 1))]])
            cons += [(Bm + Bm.T) / 2 >> 0]
        total = total + cp.sum_squares(Dk - G @ Pk)
    prob = cp.Problem(cp.Minimize(total), cons)
    prob.solve(solver="SCS", eps=1e-10, max_iters=200000)
    return prob.value


def distances(family):
    st = steps(family)
    K = len(st)
    rho0 = sum(np.sum(Dk ** 2) for Dk, _ in st) / K
    L, W, c0, r0, _ = regression(family, T_WIN, dt=DT)
    assert abs(r0 - rho0) < 1e-12 * max(1, rho0)
    semi = certify_fit(L, W, c0)
    assert semi[1] == "optimal"
    cpv = certify_steps(st, "cp")
    pv = certify_steps(st, "p")
    return dict(st=st, rho0=rho0, semi=semi[0], semi_ls=c0, cp=cpv, p=pv,
                rel=np.sqrt(np.maximum([semi[0], cpv.mean(), pv.mean()], 0) / rho0), L=L, W=W, c0=c0)


def checks(res):
    # the families are those of Chapter 8
    tt = np.linspace(0.1, 4, 40)
    assert np.allclose([ou(t, FAM.TAU)[0, 0] for t in tt], FAM.ou_coherence(tt))
    # certified values agree with direct solves
    for name, cls, fam in FAMILIES:
        r = res[name]
        K = len(r["st"])
        assert abs(r["semi"] - direct_fit(r["L"], r["W"], r["c0"])) < 1e-6 * max(1, r["rho0"])
        for kind in ("cp", "p"):
            ref = direct_steps(r["st"], kind) / K
            assert abs(r[kind].mean() - ref) < 1e-6 * max(1, r["rho0"]), (name, kind, r[kind].mean(), ref)
        # the classes are nested, so the distances can only fall
        assert r["rel"][0] >= r["rel"][1] - 1e-6 and r["rel"][1] >= r["rel"][2] - 1e-6
    rel = np.array([res[name]["rel"] for name, _, _ in FAMILIES])
    zero = rel < 1e-3
    assert (zero == np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], bool)).all(), rel
    assert np.allclose(rel[~zero], [0.367, 0.638, 0.188, 0.831, 0.415, 0.408], atol=0.001), rel[~zero]
    # the zeros are exact: the transfer maps are channels (OU) or positive maps (Hall)
    fams = {n: f for n, _, f in FAMILIES}
    for t in np.arange(0, T_WIN, DT):
        Lam = fams["OU dephasing"](t + DT) @ np.linalg.inv(fams["OU dephasing"](t))
        assert FAM.is_cp(Lam[:3, :3], Lam[:3, 3], tol=1e-12)
        Lam = fams["Hall et al."](t + DT) @ np.linalg.inv(fams["Hall et al."](t))
        assert np.all(np.abs(np.diag(Lam)) <= 1) and (t == 0 or not FAM.is_cp(Lam[:3, :3], Lam[:3, 3]))
    # where in time: the defect leaves CP- and P-divisibility only while |c| grows
    d = res["the hidden defect"]
    t = DT * np.arange(len(d["st"]))
    grow = np.abs(amplitude(t + DT, 1.0)) > np.abs(amplitude(t, 1.0))
    assert np.all(d["p"][~grow] < 1e-7) and np.all(d["p"][grow] > 1e-4)
    assert np.all((t[grow] > 1.75) & (t[grow] < 3.25))
    h = res["Hall et al."]
    assert not any(member("cp", *x) for x in h["st"][1:]) and np.all(h["cp"][1:] > 1e-7) and np.all(h["p"] == 0)
    # the joint problem over the steps that leave the class (coarse grid), with the steps as
    # variable cliques, gives the sum of the steps
    # (for the defect, whose generators reach 30 after the zero of c, SCS ends 'optimal_inaccurate')
    for name, kind in (("Hall et al.", "cp"),):
        st = steps(fams[name], dt=0.16)
        ks = [k for k in range(len(st)) if not member(kind, *st[k])]
        j = solve_pmi(st, kind, ks, True)
        s = sum(solve_pmi(st, kind, [k], False) for k in ks)
        assert len(ks) >= 7 and abs(j - s) < 1e-6 * max(1, s), (name, kind, j, s)


def main():
    res = {name: distances(fam) for name, _, fam in FAMILIES}
    checks(res)
    b = []
    # ---- (a) the table
    cols = [("semigroups", "ndBlue!22"), ("CP-divisible\\\\ families", "ndSky!75"), ("P-divisible\\\\ families", "ndSky!35")]
    x0, cw, rh, ytop = 2.9, 2.3, 0.85, 4.1
    scale = 1.6
    for j, (head, fill) in enumerate(cols):
        xc = x0 + j * cw
        b.append("\\node[font=\\scriptsize, align=center, anchor=south] at (%.3f,%.3f) {%s};" % (xc + cw / 2, ytop + 0.05, head))
    for i, (name, cls, _) in enumerate(FAMILIES):
        y = ytop - (i + 1) * rh
        b.append("\\node[font=\\small, anchor=east, align=right] at (%.3f,%.3f) {%s\\\\[-1pt]{\\scriptsize\\itshape\\color{ndInk!70} %s}};"
                 % (x0 - 0.1, y + rh / 2, name, cls))
        for j, (head, fill) in enumerate(cols):
            xc = x0 + j * cw
            r = res[name]["rel"][j]
            if r < 1e-3:
                b.append("\\fill[%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (fill, xc, y, xc + cw, y + rh))
                b.append("\\node[font=\\small] at (%.3f,%.3f) {$0$};" % (xc + cw / 2, y + rh / 2))
            else:
                b.append("\\fill[ndRose!70] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (xc + 0.12, y + rh / 2 - 0.13, xc + 0.12 + scale * r, y + rh / 2 + 0.13))
                b.append("\\draw[ndRed, line width=0.6pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (xc + 0.12, y + rh / 2 - 0.13, xc + 0.12 + scale * r, y + rh / 2 + 0.13))
                b.append("\\node[font=\\scriptsize, anchor=west] at (%.3f,%.3f) {$%.2f$};" % (xc + 0.14 + scale * r, y + rh / 2, r))
    for i in range(5):
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (x0, ytop - i * rh, x0 + 3 * cw, ytop - i * rh))
    for j in range(4):
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (x0 + j * cw, ytop, x0 + j * cw, ytop - 4 * rh))
    # the staircase: the boundary of the certified zeros
    stair = [(x0, ytop - rh), (x0 + cw, ytop - rh), (x0 + cw, ytop - 2 * rh), (x0 + 2 * cw, ytop - 2 * rh),
             (x0 + 2 * cw, ytop - 3 * rh), (x0 + 3 * cw, ytop - 3 * rh)]
    b.append(N.polyline(stair, "line width=1.3pt, ndInk"))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(a) certified distance to each class, window $[0,4]$};" % (x0 + 1.5 * cw - 1.0, ytop - 4 * rh - 0.3))

    # ---- (b) where in time
    pp = N.Plot((0, 4), (0, 1.0), 3.3, 2.4, origin=(11.0, 1.15))
    b += pp.axes(xlabel="{\\scriptsize $t$}", ylabel="{\\scriptsize share}", xticks=[(2, "$2$"), (4, "$4$")],
                 yticks=[(0.5, "$\\frac12$"), (1, "$1$")])
    b.append("\\fill[ndRose, fill opacity=0.45] %s rectangle %s;" % (pp.pt(1.81, 0), pp.pt(3.21, 1.0)))
    t = DT * np.arange(len(res["Hall et al."]["st"]))
    for name, kind, style, lab, yl in (("Hall et al.", "cp", "line width=1.0pt, ndBlue", "Hall, CP", 0.0),
                                        ("the hidden defect", "cp", "line width=1.0pt, ndRed", "defect, CP", 0.0),
                                        ("the hidden defect", "p", "line width=0.9pt, ndGold!85!ndInk, dash pattern=on 3pt off 2pt", "defect, P", 0.0)):
        v = res[name][kind]
        share = np.cumsum(v) / v.sum()
        b.append(pp.curve(np.concatenate([[0], t + DT]), np.concatenate([[0], share]), style))
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndBlue] at %s {Hall, CP};" % pp.pt(1.0, 0.55))
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndRed, align=left] at %s {defect,\\\\ CP and P};" % pp.pt(3.25, 0.45))
    b.append("\\node[note, anchor=south] at %s {$|c|$ grows};" % pp.pt(2.5, 1.02))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) where it is lost};" % pp.pt(2, -0.3))
    N.write("ch11_hierarchy", b)
    return res


if __name__ == "__main__":
    t0 = time.time()
    res = main()
    for name, _, _ in FAMILIES:
        print("%-24s semigroup %.4f  CP-div %.4f  P-div %.4f" % ((name,) + tuple(res[name]["rel"])))
    print("written figures/ch11_hierarchy.tex (%.0f s)" % (time.time() - t0))
