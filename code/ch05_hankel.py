"""ch05_hankel.py -- reading the model order off the data.

The Hankel matrix H(0) (R = 10 block rows, S = 40 columns) of K = 60 samples
of the example of the chapter (ch05_spectrum).  Its rank is the number of
modes that the preparation excites, the measurement sees, and the sampling
keeps apart (eigenvalues e^{lambda_i tau} distinct).
  (a) Exact data from four experiments: the tilted state X0 measured along n,
      sampled every tau = 0.25, shows all four modes (rank 4); the state |1>
      does not excite the precession (rank 2); sigma_x does not see the
      relaxation (rank 2); sampling every tau = pi/omega merges the two
      precession modes, e^{(-Gamma + i omega) tau} = e^{(-Gamma - i omega) tau}
      (rank 3).
  (b) sigma_n with additive Gaussian noise of standard deviation 1e-3 and
      3e-2: the cliff now ends on a noise floor, which rises with the noise;
      a mode whose singular value lies below the floor is lost.
Writes figures/ch05_hankel.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N
from ch05_spectrum import qubit_A, hankel, X0, CX, CN, TAU, OMEGA

K, R, S = 60, 10, 40
NOISE = (1e-3, 3e-2)


ONE = np.array([0.0, 0.0, -1.0, 1.0])              # |1>, on the z-axis


def data(C, sd=0.0, seed=6, x0=X0, tau=TAU):
    G = expm(qubit_A() * tau)
    y = np.array([C @ np.linalg.matrix_power(G, k) @ x0 for k in range(K)])
    if sd:
        y = y + sd * np.random.default_rng(seed).standard_normal(y.shape)
    return y


def svals(y):
    s = np.linalg.svd(hankel(y, R, S), compute_uv=False)
    return s / s[0]


def stems(Q, s, col, off, lo):
    out = []
    for i, v in enumerate(s, start=1):
        lv = max(np.log10(v), lo)
        out.append("\\draw[line width=1.0pt, %s] %s -- %s;" % (col, Q.pt(i + off, lo), Q.pt(i + off, lv)))
        out.append("\\fill[%s] %s circle (1.5pt);" % (col, Q.pt(i + off, lv)))
    return out


def modal_rank(C, x0, tau):
    """Number of modes with C v != 0 and w^dag x0 != 0, counting equal e^{lambda tau} once."""
    lam, V = np.linalg.eig(qubit_A())
    W = np.linalg.inv(V)
    live = (np.abs(np.asarray(C).reshape(-1) @ V) > 1e-12) & (np.abs(W @ x0) > 1e-12)
    mu = np.exp(lam[live] * tau)
    return len({(round(z.real, 9), round(z.imag, 9)) for z in mu})


def main():
    tau_m = np.pi / OMEGA
    reads = [("excited and seen ($\\vec x_0$, $\\sigma_n$)", CN, X0, TAU, "ndGreen"),
             ("not excited ($\\ket1$, $\\sigma_n$)", CN, ONE, TAU, "ndInk!55"),
             ("not seen ($\\vec x_0$, $\\sigma_x$)", CX, X0, TAU, "ndBlue"),
             ("merged ($\\tau=\\pi/\\omega$)", CN, X0, tau_m, "ndRed")]
    sv = [svals(data(C, x0=x0, tau=tau)) for _, C, x0, tau, _ in reads]
    ranks = [int(np.sum(s > 1e-10)) for s in sv]
    assert ranks == [4, 2, 2, 3]
    assert ranks == [modal_rank(C, x0, tau) for _, C, x0, tau, _ in reads]
    assert all(np.log10(s[r]) < -13 for s, r in zip(sv, ranks))           # the fall
    noisy = [svals(data(CN, sd)) for sd in NOISE]
    floors = [(np.log10(s[4:]).min(), np.log10(s[4:]).max()) for s in noisy]
    for s, (lo_f, hi_f) in zip(noisy, floors):
        assert np.log10(s[3]) > hi_f + 0.3                                 # four modes stay above the floor
    assert floors[1][0] > floors[0][1]                                     # more noise, higher floor

    b = []
    # ---- (a) noiseless
    lo = -17.0
    Q = N.Plot((0.4, R + 0.6), (lo, 0.5), 6.0, 4.6, origin=(0.0, 0.0))
    b += Q.axes("$i$", "$s_i/s_1$", xticks=[(i, "$%d$" % i) for i in (1, 2, 4, 6, 8, 10)],
                yticks=[(0, "$1$"), (-5, "$10^{-5}$"), (-10, "$10^{-10}$"), (-15, "$10^{-15}$")])
    for (name, _, _, _, col), s, off in zip(reads, sv, (-0.3, -0.1, 0.1, 0.3)):
        b += stems(Q, s, col, off, lo)
    xn = Q.P(5.0, 0)[0]
    for j, ((name, _, _, _, col), r) in enumerate(zip(reads, ranks)):
        b.append("\\node[note, anchor=west, text=%s] at (%.2f,%.2f) {%s: rank $%d$};" % (col, xn, Q.P(0, -0.4 - 1.6 * j)[1], name, r))
    b.append("\\node[note, anchor=west] (c) at (%.2f,%.2f) {beyond the rank the values\\\\ fall to rounding error; the rank\\\\ is the number of modes that\\\\ are excited, seen, and kept\\\\ apart by the sampling};" % (xn - 0.15, Q.P(0, -10.6)[1]))
    b.append("\\draw[pointer] (c.west) to[bend left=15] %s;" % Q.pt(4.55, -8.5))
    b.append("\\node[font=\\small] at %s {(a) exact data from four experiments};" % N.pt(Q.P((R + 1) / 2, lo) + np.array([0.9, -0.85])))
    # ---- (b) noisy sigma_n
    lo2 = -4.0
    Q2 = N.Plot((0.4, R + 0.6), (lo2, 0.25), 6.0, 4.6, origin=(8.9, 0.0))
    b += Q2.axes("$i$", "$s_i/s_1$", xticks=[(i, "$%d$" % i) for i in (1, 2, 4, 6, 8, 10)],
                 yticks=[(0, "$1$"), (-1, "$10^{-1}$"), (-2, "$10^{-2}$"), (-3, "$10^{-3}$"), (-4, "$10^{-4}$")])
    cols = ["ndGreen", "ndGold"]
    for (f0, f1), col in zip(floors, cols):
        b.append("\\fill[%s, fill opacity=0.22] %s rectangle %s;" % (col, Q2.pt(0.4, f0 - 0.08), Q2.pt(R + 0.6, f1 + 0.08)))
    for s, col, off in zip(noisy, cols, (-0.13, 0.13)):
        b += stems(Q2, s, col, off, lo2)
    xn2 = Q2.P(R + 0.9, 0)[0]
    for (f0, f1), col, sd in zip(floors, cols, ("10^{-3}", "3\\cdot10^{-2}")):
        b.append("\\node[note, anchor=west, text=%s] at (%.2f,%.2f) {floor for noise $%s$};" % (col, xn2, Q2.P(0, (f0 + f1) / 2)[1], sd))
    b.append("\\node[note, anchor=west] (f) at (%.2f,%.2f) {four singular values lie\\\\ above the floor; a mode\\\\ whose singular value falls\\\\ below it is lost};" % (Q2.P(5.0, 0)[0], Q2.P(0, -0.62)[1]))
    b.append("\\draw[pointer] (f.west) to[bend right=10] %s;" % Q2.pt(4.35, np.log10(noisy[1][3]) + 0.02))
    b.append("\\node[font=\\small] at %s {(b) the same measurement $\\sigma_n$ with noise};" % N.pt(Q2.P((R + 1) / 2, lo2) + np.array([0.9, -0.85])))
    N.write("ch05_hankel", b)
    print("written figures/ch05_hankel.tex; ranks", ranks, "floors", np.round(floors, 2))


if __name__ == "__main__":
    main()
