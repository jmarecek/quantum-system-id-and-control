"""ch09_noisemetric.py -- a noise metric, and why the fastest control is not the most robust.

The noise metric of Figure ch09:noise, g = diag(1, 1, 0.01) in the frame
sigma_j/sqrt(2) of su(2), for the driven qubit of Chapter 4 with omega = 1:
H = (1/2)(Delta sigma_z + u sigma_x) with Delta = -1 has the coordinates
h = (u/sqrt(2), 0, Delta/sqrt(2)), and the worst-case strength of the noise it
brings is Lambda = eta sqrt(g(h, h)) = eta sqrt((u^2 + 0.01 Delta^2)/2).

(a) The section y = 0 of su(2): the unit ball of the bi-invariant metric (a
    circle, dashed) and that of the noise metric, the ellipse
    h_x^2 + 0.01 h_z^2 <= 1, whose semi-axis along z is 10.  The admissible
    Hamiltonians, |u| <= 1.5, form the segment between H- and H+, the two
    between which the time-optimal control alternates; their noise rates
    sqrt(g(h, h)) are 1.063, against 0.071 for the drift alone.
(b) The bound eta L of Theorem ch09:geometricerror, L = t* sqrt(g(h, h)), for
    the time-optimal control with the bound u0 on the field, as u0 varies
    (eta = 0.1).  Along a bang-bang control sqrt(g(h, h)) is constant, so that
    L = t*(u0) sqrt((u0^2 + 0.01)/2), and t*(u0) is the least time of
    ch09_timeoptimal.py (in units of 1/u0, at k = |Delta|/u0 = 1/u0).  L is
    smallest at u0 = omega, L = pi sqrt(1.01) = 3.157, where t* = 4.443; for
    u0 = 1.5 it is 3.705 with t* = 3.485, and as u0 grows it tends to
    2 pi/sqrt(2) = 4.443.  Since a field bounded by 1 is also bounded by 1.5,
    the time-optimal control of our qubit is not the most robust admissible
    one: a weaker, slower pulse has a smaller worst-case error.  Below u0 = 1
    more bangs are needed and L jumps with t*.  The dots are the mean (filled)
    and the largest (open) error d_F(U_I(T), I) over 800 noise samples, as in
    ch09_noise.py, for u0 = 1, 1.5 and 3: they move with the bound.
Writes figures/ch09_noisemetric.tex.
"""
import numpy as np
import needham as N
import ch09_timeoptimal as TO
import ch09_noise as NO

DELTA = -1.0
G = np.array([1.0, 1.0, 0.01])
ETA = 0.1
U0S = (1.0, 1.5, 3.0)


def rate(u):
    """sqrt(g(h, h)) for H = (1/2)(Delta sigma_z + u sigma_x)."""
    return np.sqrt((G[0] * u ** 2 + G[2] * DELTA ** 2) / 2)


def tstar(u0, bangs=False):
    """The least time with the bound u0, and the number of bangs of the optimal control."""
    k = abs(DELTA) / u0
    if k <= 1:
        t, nb = 2 * np.pi / np.hypot(u0, DELTA), (1 if k == 0 else 2)
    else:
        T, info = TO.least_time(k, mmax=12, ngrid=3000)
        t, nb = T / u0, info[1]
    return (t, nb) if bangs else t


def simulate(u0, npaths=800, nsteps=1500, seed=3):
    """Final d_F(U_I(T), I) along the two-bang control with the bound u0 (k <= 1), noise as in ch09_noise.py."""
    w = np.hypot(u0, DELTA)
    k = abs(DELTA) / u0
    t1, T = (np.pi - np.arccos(k * k)) / w, 2 * np.pi / w
    H0 = lambda t: 0.5 * (DELTA * NO.SZ + (-u0 if t < t1 else u0) * NO.SX)
    rng = np.random.default_rng(seed)
    n1 = max(1, int(round(nsteps * t1 / T)))                          # a grid with a node at the switch
    grid = np.concatenate([np.linspace(0, t1, n1 + 1), np.linspace(t1, T, nsteps - n1 + 1)[1:]])
    UI = np.broadcast_to(NO.I2, (npaths, 2, 2)).copy()
    U0 = NO.I2.copy()
    Wp = np.zeros((npaths, 3))
    for j in range(len(grid) - 1):
        dt = grid[j + 1] - grid[j]
        t = 0.5 * (grid[j] + grid[j + 1])
        lam = ETA * rate(u0)
        Um = NO.expm_h(H0(t), dt / 2) @ U0
        Wp += rng.normal(0, NO.SIG * np.sqrt(dt), (npaths, 3))
        r = np.linalg.norm(Wp, axis=1)
        f = (2 / np.pi) * np.arctan(r)[:, None] * Wp / np.maximum(r, 1e-300)[:, None]
        H1I = Um.conj().T @ (lam * np.einsum('nk,kij->nij', f, NO.FRAME)) @ Um
        UI = NO.expm_h(H1I, dt) @ UI
        U0 = NO.expm_h(H0(t), dt) @ U0
    # the noiseless control reaches |1> up to a phase
    assert abs(abs(U0[1, 0]) - 1) < 1e-9
    return NO.d_F(UI), ETA * rate(u0) * T


def checks():
    assert abs(rate(1.5) - 1.063) < 1e-3 and abs(rate(0.0) - 0.0707) < 1e-4
    assert abs(tstar(1.0) * rate(1.0) - np.pi * np.sqrt(1.01)) < 1e-12
    assert abs(tstar(1.5) * rate(1.5) - 3.705) < 1e-3
    assert abs(tstar(1e4) * rate(1e4) - 2 * np.pi / np.sqrt(2)) < 1e-3
    # the bound of the time-optimal control is least at u0 = omega
    us = np.concatenate([np.linspace(0.3, 0.999, 60), np.linspace(1.001, 5.0, 60)])
    res = [tstar(u, bangs=True) for u in us]
    L = np.array([t * rate(u) for u, (t, nb) in zip(us, res)])
    nb = np.array([r[1] for r in res])
    assert np.all(L > tstar(1.0) * rate(1.0))
    return us, L, nb


def main():
    us, L, nb = checks()
    sims = {u: simulate(u) for u in U0S}
    for u, (d, bound) in sims.items():
        assert np.all(d <= bound + 1e-9)
    b = []
    # ---- (a) the unit balls in the section y = 0
    s = 1.25
    o = np.array([2.6, 2.7])
    P = lambda q: o + s * np.asarray(q)
    X, Z = 1.85, 2.0
    b.append("\\fill[ndSky, fill opacity=0.7] %s rectangle %s;" % (N.pt(P([-1.0, -Z])), N.pt(P([1.0, Z]))))
    zz = np.linspace(-Z, Z, 81)
    for sg in (1, -1):
        b.append(N.polyline([P([sg * np.sqrt(1 - 0.01 * z * z), z]) for z in zz], "ink"))
    b.append("\\fill[ndSky, fill opacity=0.7] %s" % " -- ".join(N.pt(P([np.sqrt(1 - 0.01 * z * z), z])) for z in zz[::4]) + " -- " +
             " -- ".join(N.pt(P([-np.sqrt(1 - 0.01 * z * z), z])) for z in zz[::-4]) + " -- cycle;")
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$h_x$};" % (N.pt(P([-X, 0])), N.pt(P([X + 0.1, 0]))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$h_z$};" % (N.pt(P([0, -Z])), N.pt(P([0, Z + 0.1]))))
    b.append("\\draw[dashedink] %s circle (%.3f);" % (N.pt(P([0, 0])), s))
    hp = np.array([1.5 / np.sqrt(2), DELTA / np.sqrt(2)])
    hd = np.array([0.0, DELTA / np.sqrt(2)])
    b.append("\\draw[line width=2.0pt, ndRed!70] %s -- %s;" % (N.pt(P(hp * [-1, 1])), N.pt(P(hp))))
    for h in (hp, hp * [-1, 1]):
        b.append("\\draw[vec, ndRed] %s -- %s;" % (N.pt(P([0, 0])), N.pt(P(h))))
    b.append("\\draw[vec, ndGreen] %s -- %s;" % (N.pt(P([0, 0])), N.pt(P(hd))))
    b.append("\\node[font=\\scriptsize, anchor=north west, text=ndRed] at %s {$H_+$};" % N.pt(P(hp) + [0.02, -0.02]))
    b.append("\\node[font=\\scriptsize, anchor=north east, text=ndRed] at %s {$H_-$};" % N.pt(P(hp * [-1, 1]) + [-0.02, -0.02]))
    b.append("\\node[note, anchor=north, text=ndRed] at %s {the field: noise rate %.2f};" % (N.pt(P([0, -1.05])), rate(1.5)))
    b.append("\\node[note, anchor=north, text=ndGreen] (dr) at %s {the drift alone: %.2f};" % (N.pt(P([0.0, -1.42])), rate(0.0)))
    b.append("\\draw[pointer] (dr.north east) to[bend right=25] %s;" % N.pt(P(hd) + [0.06, -0.06]))
    b.append("\\node[note, anchor=south west] (nb) at %s {unit ball of $\\gLam$:\\\\ semi-axes $1,1,10$};" % N.pt(P([0.25, 1.3])))
    b.append("\\node[note, anchor=south east] (fb) at %s {unit ball of\\\\ $\\Tr(A^\\dagger B)$};" % N.pt(P([-0.95, 1.25])))
    b.append("\\draw[pointer] (fb.south east) to[bend right=10] %s;" % N.pt(P([-np.cos(1.9), np.sin(1.9)]) * 0 + P([np.cos(2.2), np.sin(2.2)])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the noise metric of the qubit, $h_y=0$};" % N.pt(P([0, -Z - 0.2])))

    # ---- (b) the bound of the time-optimal control against u0
    pl = N.Plot((0.25, 5.0), (0.0, 0.6), 6.0, 4.4, origin=(7.6, 0.5))
    b += pl.axes(xlabel="{\\scriptsize $u_0/\\omega$}", ylabel="{\\scriptsize worst-case error}",
                 xticks=[(1, "$1$"), (1.5, "$1.5$"), (3, "$3$"), (5, "$5$")],
                 yticks=[(0.2, "$0.2$"), (0.4, "$0.4$")])
    b.append("\\draw[dashedink] %s -- %s;" % (pl.pt(0.25, ETA * 2 * np.pi / np.sqrt(2)), pl.pt(5.0, ETA * 2 * np.pi / np.sqrt(2))))
    for n in sorted(set(nb)):                                          # one piece for each number of bangs
        idx = np.where(nb == n)[0]
        for sg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(sg) > 1:
                b.append(pl.curve(us[sg], ETA * L[sg], "line width=1.3pt, ndInk"))
    for u, (d, bound) in sims.items():
        b.append("\\node[dot, fill=ndBlue, minimum size=3.6pt] at %s {};" % pl.pt(u, d.mean()))
        b.append("\\draw[ndBlue, fill=white, line width=0.7pt] %s circle (1.6pt);" % pl.pt(u, d.max()))
    b.append("\\node[dot, fill=ndGold, minimum size=4.2pt] at %s {};" % pl.pt(1.0, ETA * tstar(1.0) * rate(1.0)))
    b.append("\\node[dot, fill=ndRed, minimum size=4.2pt] at %s {};" % pl.pt(1.5, ETA * tstar(1.5) * rate(1.5)))
    b.append("\\node[note, anchor=south, text=ndGold!55!ndInk] (mn) at %s {least at $u_0=\\omega$:\\\\ $t^\\star=4.44$, bound $0.32$};" % pl.pt(0.85, 0.47))
    b.append("\\draw[pointer] (mn.south) to[bend right=10] %s;" % pl.pt(1.0, ETA * tstar(1.0) * rate(1.0) + 0.012))
    b.append("\\node[note, anchor=north west, text=ndRed] (oq) at %s {our qubit, $u_0=1.5$:\\\\ $t^\\star=3.49$, bound $0.37$};" % pl.pt(1.7, 0.31))
    b.append("\\draw[pointer] (oq.north west) to[bend left=15] %s;" % pl.pt(1.52, ETA * tstar(1.5) * rate(1.5) - 0.012))
    b.append("\\node[note, anchor=south east] at %s {$u_0\\to\\infty$: $0.44$};" % pl.pt(5.0, ETA * 2 * np.pi / np.sqrt(2) + 0.005))
    b.append("\\node[note, anchor=north, text=ndBlue] at %s {sampled errors:\\\\ mean (dots), largest (circles)};" % pl.pt(3.7, 0.12))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the bound of the time-optimal control};" % pl.pt(2.6, -0.1))
    N.write("ch09_noisemetric", b)
    return sims


if __name__ == "__main__":
    sims = main()
    for u, (d, bound) in sims.items():
        print("u0 = %.1f: t* = %.3f, bound = %.4f, mean %.4f, max %.4f" % (u, tstar(u), bound, d.mean(), d.max()))
    print("written figures/ch09_noisemetric.tex")
