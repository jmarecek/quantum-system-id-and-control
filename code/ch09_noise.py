"""ch09_noise.py -- the geometric error bound for the random Schroedinger equation.

The time-optimal control of the driven qubit (ch09_timeoptimal.py: omega = 1,
|u| <= 1.5, two bangs, T = 2 pi/Omega = 3.485) is disturbed by a noise
Hamiltonian H_1(t, w) whose size is set by a noise metric, as in Lawrence et
al. (2026): in the orthonormal frame P_j = sigma_j/sqrt(2) of su(2) with the
Frobenius inner product,
    ||H_1(t, w)||_F <= Lambda(t) = eta sqrt(g(H_0(t), H_0(t))),   g = diag(1, 1, 0.01),   eta = 0.1,
so that the field u sigma_x/2 is a hundred times noisier, per unit of
strength squared, than the drift.  The noise is H_1 = Lambda(t) f(W_t).P with
W a Wiener process in R^3 (variance 9 t) and f(w) = (2/pi) arctan(|w|) w/|w|,
a bounded continuous image of a Markov process, which satisfies the worst-case
noise condition.  In the interaction picture U_I = U_0^dag U_S obeys
U_I' = -i H_{1,I} U_I, and the theorem says
    ||U_I(t) - I||_F <= d_F(U_I(t), I) <= int_0^t Lambda(s) ds,
with d_F(U, I) = theta/sqrt(2) for U = exp(-i theta n.sigma/2).

(a) The interaction-picture paths U_I(t) of five noise samples, drawn by
    their rotation vectors theta n, inside the ball d_F <= int Lambda, which
    none of them leaves.  The worst-case noise, Lambda(t) times a fixed
    direction in the interaction picture, H_{1,S}(t) = Lambda(t) U_0 (n.P) U_0^dag,
    which turns with the control, drives U_I along a radius (red), a geodesic
    that reaches the surface at T.
(b) d_F(U_I(t), I) against t for 40 samples (grey), the mean over 2000 (blue)
    and the bound (ink), met by the worst-case noise.
(c) The final errors of 2000 samples as fractions of the bound: the mean is
    0.42, the largest 0.86.  The theorem asserts only that every
    neighbourhood of the bound has positive probability.
Writes figures/ch09_noise.tex.
"""
import numpy as np
import tikzexport as T
import needham as N

I2, SX, SY, SZ = T.pauli()
PAULI = np.array([SX, SY, SZ])
FRAME = PAULI / np.sqrt(2)
OMEGA_Q, U0 = 1.0, 1.5
DELTA = -OMEGA_Q
W = np.hypot(DELTA, U0)
T1 = (np.pi - np.arccos((DELTA / U0) ** 2)) / W
TT = 2 * np.pi / W
G = np.array([1.0, 1.0, 0.01])
ETA = 0.1
SIG = 3.0


def H0(t):
    u = -U0 if t < T1 else U0
    return 0.5 * (DELTA * SZ + u * SX)


def coeffs(H):
    return np.real(np.einsum('...ij,kji->...k', H, FRAME))


def Lam(t):
    h = coeffs(H0(t))
    return ETA * np.sqrt(G @ (h * h))


def expm_h(H, dt):
    """exp(-i dt H) for traceless Hermitian 2x2 matrices H (arrays of them)."""
    b = coeffs(H) / np.sqrt(2)
    nb = np.linalg.norm(b, axis=-1)
    n = b / np.maximum(nb, 1e-300)[..., None]
    S = np.einsum('...k,kij->...ij', n, PAULI)
    th = dt * nb
    return np.cos(th)[..., None, None] * I2 - 1j * np.sin(th)[..., None, None] * S


def rotvec(U):
    w = np.clip(np.real(np.trace(U, axis1=-2, axis2=-1)) / 2, -1, 1)
    v = -np.imag(np.einsum('...ij,kji->...k', U, PAULI)) / 2
    nv = np.linalg.norm(v, axis=-1)
    return (2 * np.arccos(w) / np.maximum(nv, 1e-300))[..., None] * v


def d_F(U):
    w = np.clip(np.real(np.trace(U, axis1=-2, axis2=-1)) / 2, -1, 1)
    return 2 * np.arccos(w) / np.sqrt(2)


def simulate(npaths, nsteps=1500, seed=0, worst=False):
    """Interaction-picture propagators along the control, for random or worst-case noise."""
    rng = np.random.default_rng(seed)
    dt = TT / nsteps
    UI = np.broadcast_to(I2, (npaths, 2, 2)).copy()
    U0 = I2.copy()
    Wp = np.zeros((npaths, 3))
    nhat = np.einsum('k,kij->ij', np.array([-0.506, 0.285, 0.814]) / np.linalg.norm([-0.506, 0.285, 0.814]), FRAME)   # a fixed direction, in the plane of the page of (a)
    ts, ds, rv, bound, frob = [0.0], [np.zeros(npaths)], [np.zeros((npaths, 3))], [0.0], [np.zeros(npaths)]
    for k in range(nsteps):
        t = (k + 0.5) * dt
        Um = expm_h(H0(t), dt / 2) @ U0                             # U_0 at the midpoint of the step
        if worst:
            H1I = np.broadcast_to(Lam(t) * nhat, (npaths, 2, 2))
        else:
            Wp += rng.normal(0, SIG * np.sqrt(dt), (npaths, 3))
            r = np.linalg.norm(Wp, axis=1)
            f = (2 / np.pi) * np.arctan(r)[:, None] * Wp / np.maximum(r, 1e-300)[:, None]
            H1S = Lam(t) * np.einsum('nk,kij->nij', f, FRAME)
            H1I = Um.conj().T @ H1S @ Um
        UI = expm_h(H1I, dt) @ UI
        U0 = expm_h(H0(t), dt) @ U0
        ts.append((k + 1) * dt)
        ds.append(d_F(UI))
        rv.append(rotvec(UI))
        bound.append(bound[-1] + Lam(t) * dt)
        frob.append(np.linalg.norm(UI - I2, axis=(1, 2)))
    return np.array(ts), np.array(ds), np.array(rv), np.array(bound), np.array(frob)


def checks():
    ts, ds, rv, bd, fr = simulate(400, seed=1)
    assert np.all(ds <= bd[:, None] + 1e-9) and np.all(fr <= ds + 1e-9)
    tw, dw, rw, bw, _ = simulate(1, worst=True)
    assert np.allclose(dw[:, 0], bw, atol=1e-9)                     # the worst-case noise meets the bound
    assert abs(bd[-1] - ETA * TT * np.sqrt((G[0] * U0 ** 2 + G[2] * DELTA ** 2) / 2)) < 1e-9
    # the worst case is a straight line in rotation-vector coordinates
    dirs = rw[1:, 0] / np.linalg.norm(rw[1:, 0], axis=1)[:, None]
    assert np.allclose(dirs, dirs[0], atol=1e-9)


def main():
    checks()
    ts, ds, rv, bd, fr = simulate(2000, seed=0)
    tw, dw, rw, bw, _ = simulate(1, worst=True)
    ratio = ds[-1] / bd[-1]
    b = []
    # ---- (a) the interaction-picture paths in the ball d_F <= int Lambda
    V = N.View(az=30.0, el=20.0, R=1.0, shift=(2.4, 2.4))
    Rb = np.sqrt(2) * bd[-1]                                          # theta of the bound
    sc = 1.65 / Rb
    b.append("\\shade[ball color=ndSky!60!white, opacity=0.35] %s circle (%.3f);" % (V.pt((0, 0, 0)), sc * Rb))
    b.append("\\draw[ink] %s circle (%.3f);" % (V.pt((0, 0, 0)), sc * Rb))
    eq = np.array([[np.cos(a), np.sin(a), 0] for a in np.linspace(0, 2 * np.pi, 241)]) * Rb * sc
    vis = eq @ V.d >= 0
    for m, sty in [(vis, "faint"), (~vis, "faint, densely dotted")]:
        idx = np.where(m)[0]
        for sg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(sg) > 1:
                b.append(N.polyline([V.P(p) for p in eq[sg]], sty))
    cols = ["ndBlue", "ndGreen", "ndGold!85!ndInk", "ndBlue!55", "ndGreen!60"]
    for j, col in enumerate(cols):
        b.append(N.polyline([V.P(sc * p) for p in rv[::5, j]], "line width=0.7pt, %s" % col))
        b.append("\\node[dot, minimum size=2.4pt, fill=%s] at %s {};" % (col, V.pt(sc * rv[-1, j])))
    b.append("\\draw[line width=1.4pt, ndRed, ->, >=stealth] %s -- %s;" % (V.pt((0, 0, 0)), V.pt(sc * rw[-1, 0])))
    b.append("\\node[dot] at %s {};" % V.pt((0, 0, 0)))
    b.append("\\node[font=\\scriptsize, anchor=north east] at %s {$\\Id$};" % N.pt(V.P((0, 0, 0)) + np.array([-0.03, -0.03])))
    b.append("\\node[note, anchor=west, text=ndRed] (wc) at %s {worst case:\\\\ a geodesic};" % N.pt(V.P((0, 0, 0)) + np.array([1.85, 1.3])))
    b.append("\\draw[pointer] (wc.west) to[bend right=15] %s;" % N.pt(V.P(0.75 * sc * rw[-1, 0]) + np.array([0.05, 0.03])))
    b.append("\\node[note, anchor=west] (bl) at %s {$\\dF\\le\\int_0^T\\Lambda$};" % N.pt(V.P((0, 0, 0)) + np.array([1.85, -1.0])))
    b.append("\\draw[pointer] (bl.west) to[bend left=10] %s;" % N.pt(V.P((0, 0, 0)) + np.array([1.2, -1.15])))
    b.append("\\node[note, anchor=north] at %s {five noise samples};" % N.pt(V.P((0, 0, 0)) + np.array([0.0, -1.75])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) $U_I(t)$ in the ball of the bound};" % N.pt(V.P((0, 0, 0)) + np.array([0.0, -2.2])))

    # ---- (b) the error against time
    pl = N.Plot((0, TT * 1.03), (0, 0.42), 4.4, 3.4, origin=(6.85, 0.55))
    b += pl.axes(xlabel="{\\scriptsize $t$}", ylabel="{\\scriptsize $\\dF(U_I(t),\\Id)$}",
                 xticks=[(T1, "$t_1$"), (TT, "$t^\\star$")], yticks=[(0.1, "$0.1$"), (0.2, "$0.2$"), (0.3, "$0.3$")])
    for j in range(40):
        b.append(pl.curve(ts[::10], ds[::10, j], "line width=0.3pt, ndInk!30"))
    b.append(pl.curve(ts[::10], ds[::10].mean(1), "line width=1.2pt, ndBlue"))
    b.append(pl.curve(ts[::10], bd[::10], "line width=1.3pt, ndInk"))
    b.append(pl.curve(tw[::10], dw[::10, 0], "line width=1.0pt, ndRed, dash pattern=on 4pt off 2pt"))
    b.append("\\node[note, anchor=east] (bb) at %s {bound $\\int_0^t\\Lambda$ $=$\\\\ worst case};" % pl.pt(1.55, 0.33))
    b.append("\\draw[pointer] (bb.east) to[bend left=10] %s;" % pl.pt(2.3, bd[np.searchsorted(ts, 2.3)] + 0.008))
    b.append("\\node[note, anchor=west, text=ndBlue] (mn) at %s {mean of 2000};" % pl.pt(2.25, 0.07))
    b.append("\\draw[pointer] (mn.north) to[bend right=10] %s;" % pl.pt(2.6, ds[np.searchsorted(ts, 2.6)].mean() - 0.005))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the error and its bound};" % pl.pt(TT / 2, -0.11))

    # ---- (c) the histogram of the final errors
    hist, edges = np.histogram(ratio, bins=20, range=(0, 1))
    ph = N.Plot((0, 1.05), (0, hist.max() * 1.15), 3.6, 3.4, origin=(13.0, 0.55))
    b += ph.axes(xlabel="{\\scriptsize $\\dF/\\!\\int\\Lambda$}", xticks=[(0.5, "$0.5$"), (1, "$1$")])
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        if h:
            b.append("\\filldraw[fill=ndSky, draw=ndBlue!70, line width=0.4pt] %s rectangle %s;" % (ph.pt(e0, 0), ph.pt(e1, h)))
    b.append("\\draw[line width=1.0pt, ndRed, dash pattern=on 4pt off 2pt] %s -- %s;" % (ph.pt(1, 0), ph.pt(1, hist.max() * 1.1)))
    b.append("\\draw[line width=0.8pt, ndBlue] %s -- %s;" % (ph.pt(ratio.mean(), 0), ph.pt(ratio.mean(), hist.max() * 1.1)))
    b.append("\\node[note, anchor=south, text=ndBlue] at %s {mean %.2f};" % (ph.pt(ratio.mean(), hist.max() * 1.1), ratio.mean()))
    b.append("\\node[note, anchor=south west, text=ndRed] at %s {bound};" % ph.pt(0.93, hist.max() * 1.1))
    b.append("\\node[note, anchor=north] at %s {largest of 2000: %.2f};" % (ph.pt(0.5, -0.2 * hist.max()), ratio.max()))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the final errors};" % ph.pt(0.5, -0.38 * hist.max()))
    N.write("ch09_noise", b)
    return ratio, bd[-1]


if __name__ == "__main__":
    ratio, bound = main()
    print("bound %.4f; final error / bound: mean %.3f, max %.3f" % (bound, ratio.mean(), ratio.max()))
    print("written figures/ch09_noise.tex")
