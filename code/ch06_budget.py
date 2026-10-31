"""ch06_budget.py -- the error budget of the identification pipeline.

Each stage of the pipeline of Chapters 4 and 5 multiplies the error of the
stage before by a factor that can be computed for the example (the qubit of
Chapter 5, measured along n, K = 60 samples at tau = 0.25, Hankel matrix with
R = 10 block rows, S = 40 columns, p = 1 observable):
    shots          eps_0 = 1/sqrt(N) per expectation value,
    Hankel matrix  <= sqrt(R S p) eps_0              (Frobenius bound),
    G_tau          <= that / sigma_4(H)              (Ho--Kalman),
    A              <= that * kappa(g) / tau          (eigenvectors, logarithm),
    (theta, K)     <= that * kappa(M)                (reconstruction map).
The reconstruction map M of Chapter 4 is built numerically for one qubit with
the basis F_j = sigma_j / sqrt 2 (Tr F_j F_k = delta_jk): rho = I/2 + sum_j
x_j F_j, H = sum_j theta_j F_j, and the dissipator sum_{kl} K_kl (F_k rho F_l
- {F_l F_k, rho}/2) with K Hermitian, written in the orthonormal basis of
Hermitian 3x3 matrices (E_kk, (E_kl + E_lk)/sqrt 2, i(E_lk - E_kl)/sqrt 2).
M maps the 12 real parameters (theta, K) to the 12 real numbers (A, beta) of
x' = A x + beta.  For one qubit kappa(M) = 2.
The bound is compared, for N = 10^4, with a Monte-Carlo simulation of every
stage (200 repetitions); the errors of G_tau and A are measured on their
eigenvalues, which do not depend on the similarity left free by Ho--Kalman,
and the parameters are read from the poles (gamma_1 = -lambda, Gamma = -Re,
omega = Im, gamma_phi = Gamma - gamma_1/2) and compared in the norm of (theta, K).
Writes figures/ch06_budget.tex.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
import needham as N
from ch05_spectrum import qubit_A, hankel, ho_kalman, X0, CN, TAU, POLES, OMEGA, GAMMA1, GPHI
from ch06_shots import sample_mean

K, R, S, P = 60, 10, 40, 1
NSHOTS = 10 ** 4
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1.0, -1.0]).astype(complex)
F = [s / np.sqrt(2) for s in (SX, SY, SZ)]


def gksl(rho, theta, Kmat):
    H = sum(t * f for t, f in zip(theta, F))
    out = -1j * (H @ rho - rho @ H)
    for k in range(3):
        for l in range(3):
            out = out + Kmat[k, l] * (F[k] @ rho @ F[l] - 0.5 * (F[l] @ F[k] @ rho + rho @ F[l] @ F[k]))
    return out


def a_beta(theta, Kmat):
    A = np.array([[np.trace(F[j] @ gksl(F[k], theta, Kmat)).real for k in range(3)] for j in range(3)])
    beta = np.array([np.trace(F[j] @ gksl(np.eye(2) / 2, theta, Kmat)).real for j in range(3)])
    return A, beta


def herm_basis():
    E = []
    for k in range(3):
        e = np.zeros((3, 3), complex)
        e[k, k] = 1
        E.append(e)
    for k in range(3):
        for l in range(k + 1, 3):
            e = np.zeros((3, 3), complex)
            e[k, l] = e[l, k] = 1 / np.sqrt(2)
            E.append(e)
            e = np.zeros((3, 3), complex)
            e[k, l], e[l, k] = -1j / np.sqrt(2), 1j / np.sqrt(2)
            E.append(e)
    return E


def reconstruction_map():
    E = herm_basis()
    cols = []
    for i in range(3):
        th = np.zeros(3)
        th[i] = 1.0
        A, bt = a_beta(th, np.zeros((3, 3)))
        cols.append(np.r_[A.ravel(), bt])
    for e in E:
        A, bt = a_beta(np.zeros(3), e)
        cols.append(np.r_[A.ravel(), bt])
    return np.array(cols).T, E


def true_parameters(E, omega=OMEGA, gamma1=GAMMA1, gphi=GPHI):
    """(theta, K) of the example: H = (omega/2) sigma_z, L = sqrt(gamma_1) sigma_-, sqrt(gamma_phi/2) sigma_z."""
    theta = np.array([0.0, 0.0, omega / np.sqrt(2)])
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    c = lambda L: np.array([np.trace(f @ L) for f in F])
    Kmat = sum(np.outer(c(L), c(L).conj()) for L in (np.sqrt(gamma1) * sm, np.sqrt(gphi / 2) * SZ))
    return np.r_[theta, [np.trace(e.conj().T @ Kmat).real for e in E]], theta, Kmat


def simulate(y, H, rng, E, p, reps=200):
    mu = np.exp(POLES * TAU)
    eh, eg, ea, ep = [], [], [], []
    for _ in range(reps):
        yn = sample_mean(y, NSHOTS, rng)[:, None]
        eh.append(np.linalg.norm(hankel(yn, R, S) - H, 2))
        Ad, _, _ = ho_kalman(yn, 4, r=R, s=S)
        m = np.linalg.eigvals(Ad)
        D = np.abs(m[:, None] - mu[None, :])
        i, j = linear_sum_assignment(D)
        eg.append(D[i, j].max())
        lam = np.log(m.astype(complex)) / TAU
        D = np.abs(lam[:, None] - POLES[None, :])
        i, j = linear_sum_assignment(D)
        ea.append(D[i, j].max())
        est = lam[i[np.argsort(j)]]                                       # in the order of POLES
        g1, gam, om = -est[1].real, -est[2:].real.mean(), np.abs(est[2:].imag).mean()
        ep.append(np.linalg.norm(true_parameters(E, om, g1, gam - g1 / 2)[0] - p))
    return np.median(eh), np.median(eg), np.median(ea), np.median(ep)


def main():
    M, E = reconstruction_map()
    p, theta, Kmat = true_parameters(E)
    A3, bt = a_beta(theta, Kmat)
    assert np.allclose(A3, qubit_A()[:3, :3]) and np.allclose(bt, qubit_A()[:3, 3] / np.sqrt(2))
    assert np.allclose(M @ p, np.r_[A3.ravel(), bt])
    kM = np.linalg.cond(M)
    assert abs(kM - 2.0) < 1e-9
    _, V = np.linalg.eig(qubit_A())
    kg = np.linalg.cond(V / np.linalg.norm(V, axis=0))
    G = expm(qubit_A() * TAU)
    y = np.array([(CN @ np.linalg.matrix_power(G, k) @ X0).item() for k in range(K)])
    H = hankel(y[:, None], R, S)
    s4 = np.linalg.svd(H, compute_uv=False)[3]
    factors = [np.sqrt(R * S * P), 1 / s4, kg / TAU, kM]
    eps0 = 1 / np.sqrt(NSHOTS)
    bound = eps0 * np.cumprod(np.r_[1.0, factors])
    total = np.prod(factors)
    target = 0.01 * np.linalg.norm(p)
    n_bound = (total / target) ** 2
    rng = np.random.default_rng(4)
    sim = simulate(y, H, rng, E, p)
    assert all(s < b for s, b in zip(sim, bound[1:5]))                      # the bound is pessimistic
    n_sim = (sim[2] * np.sqrt(NSHOTS) / target) ** 2
    print("factors", np.round(factors, 3), "kappa(g) = %.3f, kappa(M) = %.1f" % (kg, kM))
    print("bounds at N = 1e4", np.round(bound, 4), "simulated", np.round(sim, 4))
    print("N for 1%% (norm of parameters %.3f): bound %.2e, simulated eigenvalues of A %.2e" % (np.linalg.norm(p), n_bound, n_sim))

    b = []
    w, h, gap, x0 = 2.15, 1.25, 0.85, 2.3
    names = ["shots\\\\ $\\eps_0=N^{-1/2}$", "Hankel\\\\ matrix", "discrete\\\\ generator $\\mathbf G_\\tau$", "generator\\\\ $\\mathbf A$", "parameters\\\\ $(\\vec\\theta,\\Koss)$"]
    fills = ["ndSky", "ndSky", "ndSky", "ndSky", "ndRose"]
    xc = [x0 + w / 2 + k * (w + gap) for k in range(5)]
    for x, name, fl in zip(xc, names, fills):
        b.append("\\draw[ink, fill=%s, rounded corners=3pt] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (fl, x - w / 2, -h / 2, x + w / 2, h / 2))
        b.append("\\node[font=\\small, align=center] at (%.2f,0) {%s};" % (x, name))
    labels = ["$\\times\\sqrt{RSp}$", "$\\times1/\\sigma_4(\\Hankel)$", "$\\times\\kappa(\\mathbf g)/\\tau$", "$\\times\\kappa(\\mathbf M)$"]
    for k, (lab, f) in enumerate(zip(labels, factors)):
        xa, xb = xc[k] + w / 2 + 0.05, xc[k + 1] - w / 2 - 0.05
        b.append("\\draw[vec, ndRed] (%.2f,0) -- (%.2f,0);" % (xa, xb))
        b.append("\\node[font=\\scriptsize, text=ndRed, align=center, anchor=south] at (%.2f,%.2f) {%s\\\\ $=%s$};"
                 % ((xa + xb) / 2, h / 2 + 0.08, lab, ("%.0f" % f) if f >= 10 else ("%.2f" % f if f < 9.995 else "%.1f" % f)))
    # rows of numbers
    rows = [("bound, $N=10^4$", bound, "ndInk"), ("simulated", [eps0, *sim], "ndBlue")]
    for r, (lab, vals, col) in enumerate(rows):
        yr = -h / 2 - 0.45 - 0.5 * r
        b.append("\\node[note, anchor=east, text=%s] at (%.2f,%.2f) {%s};" % (col, x0 - 0.15, yr, lab))
        for x, v in zip(xc, vals):
            txt = "$%s$" % (("%.3f" % v) if v < 0.1 else ("%.2f" % v) if v < 10 else ("%.1f" % v))
            b.append("\\node[font=\\small, text=%s] at (%.2f,%.2f) {%s};" % (col, x, yr, txt))
    yb = -h / 2 - 1.55
    b.append("\\node[note, anchor=north west, align=left] at (%.2f,%.2f) {By the bound, a relative accuracy of $1\\%%$ in the parameters"
             " needs $N\\approx%s$ shots per expectation value.\\\\ The simulated eigenvalues of $\\mathbf A$ reach the same accuracy,"
             " $%.3f$, at $N\\approx%s$: each factor of the bound\\\\ is a worst case, and the worst cases do not occur together."
             " The simulated errors of $\\mathbf G_\\tau$ and $\\mathbf A$ are those of\\\\ their eigenvalues, which do not depend on"
             " the similarity transformation left free by Ho--Kalman,\\ and the parameters are read from the poles.};"
             % (x0 - 2.2, yb, sci(n_bound), target, sci(n_sim)))
    N.write("ch06_budget", b)
    print("written figures/ch06_budget.tex")


def sci(v):
    e = int(np.floor(np.log10(v)))
    return "%.0f\\cdot10^{%d}" % (v / 10 ** e, e) if v / 10 ** e < 9.5 else "10^{%d}" % (e + 1)


if __name__ == "__main__":
    main()
