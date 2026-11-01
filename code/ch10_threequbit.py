"""ch10_threequbit.py -- penalised synthesis of a three-body gate with 2-local controls.

Three qubits, no drift, unit time, the target V = exp(-i theta Z Z Z) with
theta = pi/4, a gate that no single 2-local Hamiltonian makes.  The control is
H(t) = sum_P c_P(t) P/sqrt(8) over the 63 Pauli strings, so that
<A, A> = -Tr(A^2) = Tr(H^2) = sum_P c_P^2 for A = -iH (the directions -iP/sqrt(8)
are orthonormal), and the penalised energy is
    E_q = (1/2) int_0^1 ( sum_{|supp P| <= 2} c_P^2 + q sum_{|supp P| = 3} c_P^2 ) dt.
It is minimised over 24 constant steps with the endpoint U(1) = V imposed
exactly (an augmented Lagrangian on the 128 real entries of V^dag U - I, with
L-BFGS and exact gradients), sweeping q up and then down and keeping the best
of several starts (the direct route, the previous solution, random ones); the
endpoint errors are below 1e-7.

(a) min E_q against q.  For q <= 5 the minimiser is the direct route, the
    constant Hamiltonian theta ZZZ, with E_q = 4 q theta^2 = 2.467 q; between
    q = 5 and q = 6 it jumps to a route that uses mostly 2-local terms, and the
    energies rise to about 17.31 at q = 1024 (17.28 with 32 steps).  An exact
    2-local construction, V = W exp(-i theta Z_1 Y_3) W^dag with
    W = exp(-i (pi/4) Z_2 X_3), has the length 3 sqrt(8) pi/4 and the energy 22.21,
    an upper bound for min E_inf.
(b) The 3-local part int |P_perp A|^2 dt of the optimal control: 8 theta^2 = 4.93
    for the direct route, then 0.91 at q = 8, 0.046 at q = 32 and 4.3e-5 at
    q = 1024, falling roughly like q^-2, well inside the a priori bound
    (2 min E_inf - d_g^2)/(q - 1) of the paper with d_g^2 = 8 theta^2 and min E_inf
    estimated by the energy at q = 1024.
(c) The control at q = 1024: the ten largest Pauli coefficients over time.  They
    come in pairs of 2-local strings that share a qubit, anticommute there and
    are driven in quadrature (c_i^2 + c_j^2 constant in time to 2e-4), each pair
    a 2-local field turning at a steady rate; the bracket of each pair is a multiple of ZZZ.  Which pairs appear
    depends on the start, since the symmetries of the problem (permutations of
    the qubits, local rotations that fix ZZZ) map optimal controls to optimal
    controls; all the starts reach the same energy.
Writes figures/ch10_threequbit.tex.
"""
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import needham as N

PAULI = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0]).astype(complex)}


def ps(s):
    M = np.array([[1.0 + 0j]])
    for c in s:
        M = np.kron(M, PAULI[c])
    return M


STR = ["".join(s) for s in itertools.product("IXYZ", repeat=3) if "".join(s) != "III"]
LOC3 = np.array([sum(c != "I" for c in s) == 3 for s in STR])
PM = np.array([ps(s) for s in STR]) / np.sqrt(8)
NP = len(STR)
TH = np.pi / 4
V = expm(-1j * TH * ps("ZZZ"))
NSTEP = 24
QS = [1, 2, 4, 5, 6, 8, 16, 32, 64, 256, 1024]


def evaluate(x, q, lam, mu, grad=True):
    """Augmented Lagrangian E_q + lam.r + mu/2 |r|^2, r the entries of V^dag U - I, and its gradient."""
    C = x.reshape(NSTEP, NP)
    dt = 1.0 / NSTEP
    w = np.where(LOC3, q, 1.0)
    E = 0.5 * dt * np.sum(C ** 2 * w)
    H = np.einsum("kp,pij->kij", C, PM)
    lamb, Wv = np.linalg.eigh(H)
    a = -1j * dt * lamb
    ea = np.exp(a)
    Us = np.einsum("kia,ka,kja->kij", Wv, ea, Wv.conj())
    pre = [np.eye(8, dtype=complex)]
    for k in range(NSTEP):
        pre.append(Us[k] @ pre[-1])
    U = pre[-1]
    R = V.conj().T @ U - np.eye(8)
    r = np.concatenate([R.real.ravel(), R.imag.ravel()])
    f = E + lam @ r + mu / 2 * r @ r
    if not grad:
        return f, U, r, E
    Lam = (lam[:64] + mu * r[:64]).reshape(8, 8) + 1j * (lam[64:] + mu * r[64:]).reshape(8, 8)
    B = Lam.conj().T @ V.conj().T                              # dS = Re Tr(B dU)
    suf = [np.eye(8, dtype=complex)]
    for k in range(NSTEP - 1, -1, -1):
        suf.append(suf[-1] @ Us[k])
    suf = suf[::-1]
    g = dt * C * w
    for k in range(NSTEP):
        Mt = Wv[k].conj().T @ (pre[k] @ B @ suf[k + 1]) @ Wv[k]
        diff = a[k][:, None] - a[k][None, :]
        small = np.abs(diff) < 1e-12
        Phi = np.where(small, ea[k][:, None] * np.ones((8, 8)),
                       (ea[k][:, None] - ea[k][None, :]) / np.where(small, 1.0, diff))   # divided differences of exp
        Et = np.einsum("ia,pij,jb->pab", Wv[k].conj(), PM, Wv[k]) * (-1j * dt)
        g[k] += np.real(np.einsum("ba,pab,ab->p", Mt, Et, Phi))
    return f, g.ravel(), U, r, E


def solve(x0, q, tol=1e-9):
    lam, mu, x = np.zeros(128), 10.0, x0.copy()
    for _ in range(40):
        x = minimize(lambda z: evaluate(z, q, lam, mu)[:2], x, jac=True, method="L-BFGS-B",
                     options={"maxiter": 20000, "gtol": 1e-9, "maxcor": 30}).x
        _, U, r, E = evaluate(x, q, lam, mu, grad=False)
        lam = lam + mu * r
        if np.linalg.norm(r) < tol:
            break
        mu = min(3 * mu, 1e5)
    return x, E, np.linalg.norm(U - V)


def sweep(seed=1):
    rng = np.random.default_rng(seed)
    xd = np.zeros((NSTEP, NP))
    xd[:, STR.index("ZZZ")] = TH * np.sqrt(8)                     # the direct route
    xd = xd.ravel()
    best, randE = {}, []

    def keep(q, x, E, err):
        assert err < 1e-7, (q, err)
        if q not in best or E < best[q][1] - 1e-9:
            best[q] = (x, E)
    prev = None
    for q in QS:                                                  # upwards: direct route, continuation, random
        starts = [xd + rng.normal(0, 0.05, xd.size)] + ([prev] if prev is not None else [])
        if q in (16, 1024):
            starts += [rng.normal(0, 1, xd.size)]
        for x0 in starts:
            x, E, err = solve(x0, q)
            keep(q, x, E, err)
            if q == 1024 and x0 is starts[-1]:
                randE.append(E)
        prev = best[q][0]
    for i in range(len(QS) - 2, -1, -1):                          # downwards from the large-q solutions
        x, E, err = solve(best[QS[i + 1]][0], QS[i])
        keep(QS[i], x, E, err)
    return best, randE


def violation(x):
    C = x.reshape(NSTEP, NP)
    return np.sum(C[:, LOC3] ** 2) / NSTEP


def sandwich_energy():
    """The exact 2-local construction V = W exp(-i theta Q) W^dag, W = exp(-i pi/4 Z_2 X_3), Q = +-Z_1 Y_3."""
    W = expm(-0.25j * np.pi * ps("IZX"))
    for sgn in (1, -1):
        if np.allclose(W @ expm(-1j * TH * sgn * ps("ZIY")) @ W.conj().T, V):
            L = 3 * np.sqrt(8) * np.pi / 4                         # three Pauli rotations by pi/4 = theta
            return 0.5 * L ** 2, sgn
    raise AssertionError("construction failed")


def checks(best, randE):
    Esw, _ = sandwich_energy()
    assert abs(Esw - 22.207) < 1e-3
    E = np.array([best[q][1] for q in QS])
    viol = np.array([violation(best[q][0]) for q in QS])
    dg2 = 8 * TH ** 2
    for q, e, v in zip(QS, E, viol):
        if q <= 5:                                                # the direct route
            assert abs(e - 4 * q * TH ** 2) < 1e-6 and abs(v - dg2) < 1e-6
        else:
            assert e < 4 * q * TH ** 2 - 1e-3 and v < 0.5 * dg2
    assert np.all(np.diff(E) > -1e-6) and E[-1] < Esw               # nondecreasing, below the construction
    Einf = E[-1]
    for q, v in zip(QS, viol):
        if q >= 2:
            assert v <= (2 * Einf - dg2) / (q - 1) + 1e-9           # the a priori bound
    qv = np.array(QS) * viol
    assert np.all(np.diff(qv[QS.index(8):]) < 0) and qv[-1] < 0.05
    assert max(randE) - min(randE) < 1e-5 and abs(min(randE) - Einf) < 1e-5   # random starts agree
    # the leading terms at q = 1024 pair up: each has a partner among them whose bracket is a multiple of ZZZ
    C = best[1024][0].reshape(NSTEP, NP)
    en = np.sum(C ** 2, 0) / NSTEP
    top = list(np.argsort(-en)[:10])
    zzz = ps("ZZZ")
    for i in top:
        partners = [j for j in top if j != i and np.allclose(*_bracket_vs(ps(STR[i]), ps(STR[j]), zzz))]
        quad = [j for j in partners if np.std(C[:, i] ** 2 + C[:, j] ** 2) < 1e-2 * np.mean(C[:, i] ** 2 + C[:, j] ** 2)]
        assert quad, STR[i]                                       # a partner in quadrature: c_i^2 + c_j^2 constant
    return E, viol, Esw


def _bracket_vs(A, B, Z):
    """(bracket, its projection on Z): equal iff [A, B] is a multiple of Z."""
    Cm = A @ B - B @ A
    return Cm, np.trace(Z @ Cm) / 8 * Z


def main():
    best, randE = sweep()
    E, viol, Esw = checks(best, randE)
    b = []
    # ---- (a) energies against q
    lq = np.log2(QS)
    pa = N.Plot((0, 10.6), (0, 26), 4.4, 3.4, origin=(1.0, 0.9))
    b += pa.axes(xlabel="{\\scriptsize $q$}", ylabel="{\\scriptsize $\\min E_q$}",
                 xticks=[(0, "$1$"), (2, "$4$"), (4, "$16$"), (6, "$64$"), (8, "$256$"), (10, "$1024$")],
                 yticks=[(10, "$10$"), (20, "$20$")])
    qq = np.linspace(0, np.log2(26 / (4 * TH ** 2)), 50)
    b.append(pa.curve(qq, 4 * TH ** 2 * 2 ** qq, "line width=0.9pt, ndGold, dash pattern=on 3pt off 2pt"))
    b.append("\\draw[dashedink] %s -- %s;" % (pa.pt(0, Esw), pa.pt(10.6, Esw)))
    b.append("\\node[note, anchor=north east] at %s {a 2-local construction, $%.1f$};" % (pa.pt(10.6, Esw - 0.3), Esw))
    b.append(pa.curve(lq, E, "line width=1.0pt, ndBlue"))
    for x, e, q in zip(lq, E, QS):
        b.append("\\node[dot, minimum size=3pt, fill=%s] at %s {};" % ("ndGold" if q <= 5 else "ndBlue", pa.pt(x, e)))
    b.append("\\node[note, anchor=east, text=ndGold!55!ndInk] at %s {direct route, $4q\\theta^2$};" % pa.pt(3.15, 24.6))
    b.append("\\node[note, anchor=north, text=ndBlue] at %s {$%.2f$ at $q=1024$};" % (pa.pt(8.6, E[-1] - 0.6), E[-1]))
    b.append("\\node[note, anchor=west] (jp) at %s {the minimiser jumps\\\\ between $q=5$ and $6$};" % pa.pt(4.2, 7.0))
    b.append("\\draw[pointer] (jp.west) to[bend left=20] %s;" % pa.pt(np.log2(5.5) + 0.1, 12.2))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the least penalised energy};" % pa.pt(5.3, -5.0))

    # ---- (b) the 3-local part against q
    pb = N.Plot((0, 10.6), (-5, 1.2), 4.4, 3.4, origin=(7.1, 0.9))
    b += pb.axes(xlabel="{\\scriptsize $q$}", ylabel="{\\scriptsize $\\int\\abs{P_\\perp A}^2\\mathrm dt$}",
                 xticks=[(0, "$1$"), (4, "$16$"), (8, "$256$"), (10, "$1024$")],
                 yticks=[(0, "$1$"), (-2, "$10^{-2}$"), (-4, "$10^{-4}$")])
    qb = np.linspace(1, 10.6, 60)
    bound = (2 * E[-1] - 8 * TH ** 2) / (2 ** qb - 1)
    b.append(pb.curve(qb, np.log10(bound), "line width=0.9pt, ndRed, dash pattern=on 3pt off 2pt"))
    b.append("\\node[note, anchor=south west, text=ndRed] at %s {a priori bound\\\\ $(2\\min E_\\infty-d_g^2)/(q-1)$};" % pb.pt(5.6, 0.0))
    b.append(pb.curve(lq, np.log10(viol), "line width=1.0pt, ndBlue"))
    for x, v, q in zip(lq, viol, QS):
        b.append("\\node[dot, minimum size=3pt, fill=%s] at %s {};" % ("ndGold" if q <= 5 else "ndBlue", pb.pt(x, np.log10(v))))
    b.append("\\node[note, anchor=north east, text=ndBlue] at %s {roughly $q^{-2}$};" % pb.pt(5.5, -2.2))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the 3-local part of the control};" % pb.pt(5.3, -6.1))

    # ---- (c) the control at q = 1024
    C = best[1024][0].reshape(NSTEP, NP)
    en = np.sum(C ** 2, 0) / NSTEP
    order = np.argsort(-en)[:10]
    cmax = np.abs(C[:, order]).max()
    x0, y0, cw, ch = 13.6, 4.2, 0.1, 0.3
    for row, i in enumerate(order):
        s = STR[i]
        lab = "".join("%s_%d" % (c, k + 1) for k, c in enumerate(s) if c != "I")
        y = y0 - row * ch
        b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {$%s$};" % (x0 - 0.05, y - ch / 2, lab))
        for k in range(NSTEP):
            v = C[k, i] / cmax
            col = "ndBlue" if v > 0 else "ndRed"
            b.append("\\fill[%s!%d] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, int(round(100 * abs(v))), x0 + k * cw, y - ch, x0 + (k + 1) * cw, y))
    b.append("\\draw[faint] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0, y0 - 10 * ch, x0 + NSTEP * cw, y0))
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$0$};" % (x0, y0 - 10 * ch - 0.02))
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$t$};" % (x0 + NSTEP * cw / 2, y0 - 10 * ch - 0.02))
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$1$};" % (x0 + NSTEP * cw, y0 - 10 * ch - 0.02))
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {pairs in quadrature;\\\\ each bracket $\\propto ZZZ$};" % (x0 + NSTEP * cw / 2, y0 - 10 * ch - 0.45))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {(c) the control at $q=1024$};" % (x0 + NSTEP * cw / 2 - 0.3, -0.1))
    N.write("ch10_threequbit", b)
    return E, viol, Esw


if __name__ == "__main__":
    E, viol, Esw = main()
    for q, e, v in zip(QS, E, viol):
        print("q = %5d   min E_q = %8.4f   3-local part = %.3e" % (q, e, v))
    print("2-local construction: %.4f" % Esw)
    print("written figures/ch10_threequbit.tex")
