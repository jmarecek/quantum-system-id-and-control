"""ch08_families.py -- the hierarchy of families of channels.

A family Phi_t of channels (each Phi_t completely positive and trace
preserving) is P-divisible if Phi_t = Lambda_{t,s} Phi_s with positive,
trace-preserving Lambda_{t,s} for all s <= t, CP-divisible if the Lambda_{t,s}
are channels, and a semigroup if moreover Phi_t = e^{tL} with a constant
generator.  One example of each kind, drawn in nested regions:
  * semigroup: the qubit of Chapter 5 (constant rates gamma1 = 0.2, gphi = 0.4);
  * CP-divisible, not a semigroup: the Ornstein--Uhlenbeck dephasing of
    Chapter 3, coherence exp(-b^2 tau (t - tau(1 - e^{-t/tau}))) with b = 1,
    tau = 0.5, whose rate gamma(t) = b^2 tau (1 - e^{-t/tau}) is positive but
    depends on time;
  * P-divisible, not CP-divisible: the Pauli dynamics of Hall, Cresser, Li and
    Andersson (2014), rates g1 = g2 = 1 and g3(t) = -tanh t, with
    lambda_1 = lambda_2 = (1 + e^{-2t})/2 and lambda_3 = e^{-2t}; every Phi_t
    is a channel, every pair sum g_i + g_j >= 0, but g3 < 0;
  * not even P-divisible: the qubit with the hidden defect of Chapter 4,
    whose trace distance D(t) = c(t)^2 revives.
Writes figures/ch08_families.tex.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import tikzexport as T
import needham as N
import ch05_spectrum as Q
import ch07_defect as D

I2, X, Y, Z = T.pauli()
PAULIS = (X, Y, Z)
B_RMS, TAU = 1.0, 0.5


def choi_affine(M, c):
    """Choi matrix sum_ij |i><j| (x) Phi(|i><j|) of the qubit map r -> M r + c."""
    def Phi(S):
        v = np.array([np.trace(S @ P) for P in PAULIS])
        w = M @ v
        return 0.5 * (np.trace(S) * (I2 + sum(ck * P for ck, P in zip(c, PAULIS))) + sum(wk * P for wk, P in zip(w, PAULIS)))
    J = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            E = np.zeros((2, 2), dtype=complex); E[i, j] = 1
            J += np.kron(E, Phi(E))
    return J


def is_cp(M, c=np.zeros(3), tol=1e-10):
    return np.linalg.eigvalsh(choi_affine(M, c)).min() >= -tol


def ou_coherence(t):
    return np.exp(-B_RMS ** 2 * TAU * (t - TAU * (1 - np.exp(-t / TAU))))


def ou_rate(t):
    return B_RMS ** 2 * TAU * (1 - np.exp(-t / TAU))


def hall_lambdas(t):
    return np.array([(1 + np.exp(-2 * t)) / 2, (1 + np.exp(-2 * t)) / 2, np.exp(-2 * t)])


def hall_rates(t):
    return np.array([1.0, 1.0, -np.tanh(t)])


def checks():
    out = {}
    # the semigroup of Chapter 5: Phi_{t+s} = Phi_t Phi_s, and every intermediate map is a channel
    A = Q.qubit_A()
    for s, t in [(0.3, 1.1), (1.0, 2.5)]:
        assert np.allclose(expm(A * (t + s)), expm(A * t) @ expm(A * s))
        E = expm(A * (t - s))
        assert is_cp(E[:3, :3], E[:3, 3])
    # Ornstein--Uhlenbeck dephasing: rate >= 0 and time dependent; intermediate maps are channels; not a semigroup
    tt = np.linspace(0.01, 4, 400)
    fd = -np.gradient(np.log(ou_coherence(tt)), tt)
    assert np.allclose(fd[2:-2], ou_rate(tt[2:-2]), atol=1e-3) and np.all(ou_rate(tt) >= 0)
    assert abs(ou_coherence(2.0) - ou_coherence(1.0) ** 2) > 1e-2
    for s, t in [(0.2, 0.5), (1.0, 3.0)]:
        f = ou_coherence(t) / ou_coherence(s)
        assert is_cp(np.diag([f, f, 1.0]))
    # Hall et al.: integrate the Pauli master equation, d lambda_j/dt = -(sum_{i != j} g_i) lambda_j
    sol = solve_ivp(lambda t, l: -np.array([hall_rates(t)[[1, 2]].sum(), hall_rates(t)[[0, 2]].sum(),
                                             hall_rates(t)[[0, 1]].sum()]) * l,
                    (0, 4), [1.0, 1.0, 1.0], t_eval=tt, rtol=1e-10, atol=1e-12)
    assert np.allclose(sol.y.T, np.array([hall_lambdas(t) for t in tt]), atol=1e-7)
    assert all(is_cp(np.diag(hall_lambdas(t)), tol=1e-9) for t in tt)         # every Phi_t is a channel
    g = np.array([hall_rates(t) for t in tt])
    assert np.all(g[:, 0] + g[:, 1] >= 0) and np.all(g[:, 0] + g[:, 2] >= 0) and np.all(g[:, 1] + g[:, 2] >= 0)
    assert np.all(g[:, 2] < 0)
    lam = np.array([hall_lambdas(t) for t in tt])
    assert np.all(np.diff(lam, axis=0) <= 1e-12)                                # |lambda_j| never grows: P-divisible
    s, t = 0.5, 1.0
    Lm = np.diag(hall_lambdas(t) / hall_lambdas(s))
    assert np.all(np.abs(np.diag(Lm)) <= 1) and not is_cp(Lm)                 # positive, not completely positive
    out["hall_choi_min"] = np.linalg.eigvalsh(choi_affine(Lm, np.zeros(3))).min()
    # the hidden defect: D(t) = c(t)^2 revives, so not even P-divisible
    Dt = D.amplitude(tt) ** 2
    assert np.any(np.diff(Dt) > 1e-3)
    return out


def inset(b, x0, y0, w, h, ylim, curves, xmax=4.0, zero=None):
    """A small plot with faint axes; curves = [(ts, vs, style), ...]."""
    P = N.Plot((0.0, xmax), ylim, w, h, origin=(x0, y0))
    yz = ylim[0] if zero is None else zero
    b.append("\\fill[white, fill opacity=0.7, rounded corners=2pt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
             % (x0 - 0.12, y0 - 0.12, x0 + w + 0.15, y0 + h + 0.12))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (P.pt(0, yz), P.pt(xmax * 1.05, yz)))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (P.pt(0, ylim[0]), P.pt(0, ylim[1] * 1.05 if ylim[1] > 0 else ylim[1])))
    for ts, vs, style in curves:
        b.append(P.curve(ts, vs, style))
    return P


def main():
    out = checks()
    b = []
    regions = [  # (x1, y1, fill, title)
        (13.3, 6.3, "ndPaper", "families of channels $\\Phi_t$: every $\\Phi_t$ completely positive"),
        (10.2, 5.45, "ndSky!35", "P-divisible: $\\Phi_t=\\Lambda_{t,s}\\Phi_s$ with $\\Lambda_{t,s}$ positive"),
        (7.1, 4.6, "ndSky!75", "CP-divisible: every $\\Lambda_{t,s}$ a channel"),
        (4.0, 3.7, "ndBlue!22", "semigroups $\\mathrm{e}^{t\\mathcal{L}}$"),
    ]
    for k, (x1, y1, fill, title) in enumerate(regions):
        x0 = y0 = 0.27 * k
        b.append("\\filldraw[fill=%s, draw=ndInk!70, line width=0.7pt, rounded corners=8pt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                 % (fill, x0, y0, x1, y1))
        b.append("\\node[anchor=north west, font=\\small] at (%.3f,%.3f) {%s};" % (x0 + 0.12, y1 - 0.08, title))
    small = "font=\\scriptsize\\itshape, text=ndInk!85, align=center, anchor=north"
    yI, hI, yN = 1.95, 1.0, 1.78                           # inset bottom, inset height, top of the note below it
    tt = np.linspace(0, 4, 200)
    # semigroup: constant rates of the Chapter 5 qubit
    inset(b, 1.2, yI, 2.3, hI, (0.0, 0.5),
          [(tt, 0 * tt + Q.GAMMA1, "line width=1.0pt, ndRed"), (tt, 0 * tt + Q.GPHI, "line width=1.0pt, ndBlue")])
    b.append("\\node[font=\\tiny, text=ndBlue, anchor=south east] at (3.5,%.3f) {$\\gamma_\\phi$};" % (yI + hI * Q.GPHI / 0.5))
    b.append("\\node[font=\\tiny, text=ndRed, anchor=south east] at (3.5,%.3f) {$\\gamma_1$};" % (yI + hI * Q.GAMMA1 / 0.5))
    b.append("\\node[%s] at (2.35,%.3f) {the qubit of Chapter 5:\\\\ constant rates};" % (small, yN))
    # CP-divisible: OU dephasing
    inset(b, 4.5, yI, 2.1, hI, (0.0, 0.6), [(tt, ou_rate(tt), "line width=1.0pt, ndBlue")])
    b.append("\\node[font=\\tiny, text=ndBlue, anchor=north east] at (6.6,%.3f) {$\\gamma(t)\\ge0$};" % (yI + hI * 0.5 / 0.6 - 0.04))
    b.append("\\node[%s] at (5.55,%.3f) {dephasing by\\\\ Ornstein--Uhlenbeck\\\\ noise (Chapter 3)};" % (small, yN))
    # P-divisible: Hall et al.
    ylim = (-1.2, 1.3)
    P = inset(b, 7.6, yI, 2.1, hI, ylim, [], zero=0.0)
    g3 = -np.tanh(tt)
    poly = [P.P(0, 0)] + [P.P(t, v) for t, v in zip(tt, g3)] + [P.P(tt[-1], 0)]
    b.append("\\fill[ndRose, fill opacity=0.8] %s -- cycle;" % " -- ".join(N.pt(q) for q in poly))
    b.append(P.curve(tt, 0 * tt + 1, "line width=1.0pt, ndBlue"))
    b.append(P.curve(tt, g3, "line width=1.0pt, ndRed"))
    b.append("\\node[font=\\tiny, text=ndBlue, anchor=south east] at %s {$g_1=g_2=1$};" % P.pt(4.0, 1.02))
    b.append("\\node[font=\\tiny, text=ndRed, anchor=center] at %s {$g_3=-\\tanh t<0$};" % P.pt(2.3, 0.5))
    b.append("\\node[%s] at (8.65,%.3f) {Hall et al.: every\\\\ $g_i+g_j\\ge0$, but $g_3<0$};" % (small, yN))
    # not P-divisible: the hidden defect
    ts = np.linspace(0, 8, 400)
    P = inset(b, 10.7, yI, 2.2, hI, (0.0, 1.0), [(ts, D.amplitude(ts) ** 2, "line width=1.0pt, ndRed")], xmax=8.0)
    b.append("\\node[font=\\tiny, text=ndRed, anchor=south west] at %s {$D(t)$};" % P.pt(0.5, 0.85))
    b.append("\\node[%s] at (11.8,%.3f) {the hidden defect\\\\ (Chapter 4): the\\\\ distance revives};" % (small, yN))
    N.write("ch08_families", b)
    print("Hall et al.: the intermediate map from t = 0.5 to 1 has a Choi eigenvalue %.4f < 0" % out["hall_choi_min"])
    print("written figures/ch08_families.tex")


if __name__ == "__main__":
    main()
