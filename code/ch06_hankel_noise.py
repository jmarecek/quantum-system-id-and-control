"""ch06_hankel_noise.py -- the Hankel matrix of the example under shot noise.

The set-up of ch05_hankel: the qubit of Chapter 5 measured along n, K = 60
samples at tau = 0.25, Hankel matrix H(0) with R = 10 block rows and S = 40
columns.  Each sample is now the mean of N shots.
  (a) Against N: the exact singular values sigma_1, ..., sigma_4 of H(0)
      (horizontal lines), for the tilted state X0 (solid) and for the weakly
      tilted state at 15 degrees from |1> (dashed), and the median of the
      fifth singular value of the noisy Hankel matrix over 40 repetitions,
      the noise floor, which falls like c / sqrt(N).  The weakest mode rises
      above the floor at N* = (c / sigma_4)^2.
  (b) Root-mean-square error of the poles found by Ho--Kalman (n = 4) over
      150 repetitions for each N, one curve for each pole; above N* the errors
      fall like N^{-1/2}, and the slow relaxation -gamma_1 is the least well
      determined.
Writes figures/ch06_hankel_noise.tex.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
import needham as N
from ch05_spectrum import qubit_A, hankel, ho_kalman, X0, CN, TAU, POLES
from ch06_shots import tilted, sample_mean

K, R, S = 60, 10, 40
WEAK = np.radians(15.0)


def data(x0):
    G = expm(qubit_A() * TAU)
    return np.array([(CN @ np.linalg.matrix_power(G, k) @ x0).item() for k in range(K)])


def svals(y):
    return np.linalg.svd(hankel(np.asarray(y)[:, None], R, S), compute_uv=False)


def floor(y, shots, rng, reps=40):
    return np.median([svals(sample_mean(y, shots, rng))[4] for _ in range(reps)])


def pole_errors(y, shots, rng, reps=150):
    """Root-mean-square error of each Ho--Kalman pole, matched to the true poles."""
    err = []
    for _ in range(reps):
        Ad, _, _ = ho_kalman(sample_mean(y, shots, rng)[:, None], 4, r=R, s=S)
        lam = np.log(np.linalg.eigvals(Ad).astype(complex)) / TAU
        D = np.abs(lam[:, None] - POLES[None, :])
        i, j = linear_sum_assignment(D)
        e = np.empty(4)
        e[j] = D[i, j]
        err.append(e)
    return np.sqrt(np.mean(np.square(err), axis=0))


def main():
    rng = np.random.default_rng(6)
    y0, y1 = data(X0), data(tilted(WEAK))
    s0, s1 = svals(y0)[:5], svals(y1)[:5]
    assert s0[4] < 1e-12 and s1[4] < 1e-12 and s0[3] > 0.5 and s1[3] < 0.3
    grid = np.logspace(1, 6, 11)
    fl0 = np.array([floor(y0, int(n), rng) for n in grid])
    fl1 = np.array([floor(y1, int(n), rng) for n in grid])
    assert np.all(np.abs(np.log(fl1 / fl0)) < 0.15)                       # the floor does not depend on the state
    slope, logc = np.polyfit(np.log10(grid), np.log10(fl0), 1)
    assert abs(slope + 0.5) < 0.05                                         # the floor falls like N^{-1/2}
    c = 10 ** logc
    nstar = [(c / s0[3]) ** 2, (c / s1[3]) ** 2]
    print("floor ~ %.2f N^%.3f; N* = %.0f (tilted), %.0f (15 degrees)" % (c, slope, *nstar))
    assert 100 < nstar[0] < 250 and 600 < nstar[1] < 1200

    pgrid = np.array([3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6])
    perr = np.array([pole_errors(y0, int(n), rng) for n in pgrid])         # columns follow POLES
    top = pgrid >= 3e3
    for k in range(4):
        sl = np.polyfit(np.log10(pgrid[top]), np.log10(perr[top, k]), 1)[0]
        assert -0.65 < sl < -0.4, (k, sl)
    assert np.all(perr[top, 1] > perr[top, 0]) and np.all(perr[top, 1] > perr[top, 2])   # -gamma_1 is the worst
    print("pole errors at N = 1e6:", np.round(perr[-1], 4))

    b = []
    # ---- (a) singular values and the floor
    lo, hi = -2.5, 1.25
    Q = N.Plot((1.0, 6.0), (lo, hi), 6.6, 4.8, origin=(0.0, 0.0))
    b += Q.axes("shots $N$", "singular value", xticks=[(k, "$10^{%d}$" % k) for k in range(1, 7)],
                yticks=[(k, "$10^{%d}$" % k) for k in (-2, -1, 0, 1)])
    for i in range(4):
        b.append("\\draw[line width=0.9pt, ndBlue] %s -- %s;" % (Q.pt(1.0, np.log10(s0[i])), Q.pt(6.0, np.log10(s0[i]))))
        b.append("\\draw[line width=0.9pt, ndGold, dash pattern=on 3pt off 2pt] %s -- %s;" % (Q.pt(1.0, np.log10(s1[i])), Q.pt(6.0, np.log10(s1[i]))))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=west] at %s {$\\sigma_4$};" % Q.pt(6.02, np.log10(s0[3])))
    b.append("\\node[font=\\scriptsize, text=ndGold!80!ndInk, anchor=west] at %s {$\\sigma_4$};" % Q.pt(6.02, np.log10(s1[3])))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=west] at %s {$\\sigma_1$};" % Q.pt(6.02, np.log10(s0[0])))
    lg = np.log10(grid)
    b.append(Q.curve(lg, np.log10(fl0), "line width=1.1pt, ndRed"))
    b += ["\\fill[ndRed] %s circle (1.4pt);" % Q.pt(x, np.log10(v)) for x, v in zip(lg, fl0)]
    for ns, s4, col in [(nstar[0], s0[3], "ndBlue"), (nstar[1], s1[3], "ndGold!80!ndInk")]:
        x = np.log10(ns)
        b.append("\\draw[dashedink] %s -- %s;" % (Q.pt(x, lo), Q.pt(x, np.log10(s4))))
        b.append("\\node[dot, fill=%s, minimum size=4.5pt] at %s {};" % (col, Q.pt(x, np.log10(s4))))
    b.append("\\node[note, anchor=north west, text=ndRed] (f) at %s {the noise floor: the fifth singular\\\\ value, about $%.1f/\\sqrt N$};" % (Q.pt(4.1, -0.7), c))
    b.append("\\draw[pointer] (f.south west) to[bend right=20] %s;" % Q.pt(4.3, np.log10(np.interp(4.3, lg, fl0)) + 0.05))
    b.append("\\node[note, anchor=south] at %s {solid: the tilted state $\\vec x_0$;\\ dashed: tilted by $15^\\circ$ only};" % Q.pt(4.4, hi + 0.12))
    b.append("\\node[note, anchor=north] at %s {the weakest mode rises\\\\ above the floor at\\\\ $N\\approx%d$ and $N\\approx%d$};" % (Q.pt(2.55, -1.55), round(nstar[0], -1), round(nstar[1], -2)))
    b.append("\\node[font=\\small] at %s {(a) the singular values of $\\Hankel(0)$ and the noise};" % N.pt(Q.P(3.5, lo) + np.array([0.0, -0.9])))
    # ---- (b) pole errors
    lo2, hi2 = -3.4, 0.8
    Q2 = N.Plot((1.4, 6.0), (lo2, hi2), 6.2, 4.8, origin=(9.3, 0.0))
    b += Q2.axes("shots $N$", "error of the pole", xticks=[(k, "$10^{%d}$" % k) for k in range(2, 7)],
                 yticks=[(k, "$10^{%d}$" % k) for k in (-3, -2, -1, 0)])
    xs = np.log10(nstar[0])
    b.append("\\fill[ndRose, fill opacity=0.5] %s rectangle %s;" % (Q2.pt(1.4, lo2), Q2.pt(xs, hi2)))
    b.append("\\node[font=\\scriptsize, anchor=south] at %s {$N^*$};" % Q2.pt(xs, hi2))
    lp = np.log10(pgrid)
    curves = [(1, "ndRed", "$-\\gamma_1$"), (2, "ndBlue", "$-\\Gamma\\pm\\imath\\omega$"), (0, "ndInk!60", "$0$")]
    for k, col, lab in curves:
        yv = np.clip(np.log10(perr[:, k]), lo2, hi2)
        b.append(Q2.curve(lp, yv, "line width=1.1pt, %s" % col))
        b += ["\\fill[%s] %s circle (1.4pt);" % (col, Q2.pt(x, v)) for x, v in zip(lp, yv)]
        off = {1: 0.0, 2: 0.13, 0: -0.13}[k]
        b.append("\\node[font=\\scriptsize, text=%s, anchor=west] at %s {%s};" % (col, Q2.pt(6.03, np.log10(perr[-1, k]) + off), lab))
    # slope -1/2 guide
    g0 = np.log10(perr[top, 1][0]) + 0.35
    b.append("\\draw[dashedink] %s -- %s;" % (Q2.pt(np.log10(3e3), g0), Q2.pt(6.0, g0 - 0.5 * (6.0 - np.log10(3e3)))))
    b.append("\\node[note, anchor=south west] at %s {slope $-\\tfrac12$};" % Q2.pt(4.6, g0 - 0.5 * (4.6 - np.log10(3e3)) + 0.05))
    b.append("\\node[note, anchor=north west] (l) at %s {below $N^*$ the weakest\\\\ mode is lost in the noise,\\\\ and the estimates fail};" % Q2.pt(2.3, -2.3))
    b.append("\\draw[pointer] (l.west) to[bend left=20] %s;" % Q2.pt(1.8, -2.5))
    b.append("\\node[font=\\small] at %s {(b) the error of the poles found by Ho--Kalman};" % N.pt(Q2.P(4.0, lo2) + np.array([0.0, -0.9])))
    N.write("ch06_hankel_noise", b)
    print("written figures/ch06_hankel_noise.tex")


if __name__ == "__main__":
    main()
