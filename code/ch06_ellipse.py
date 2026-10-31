"""ch06_ellipse.py -- the error ellipse of two parameters.

The decay rates (gamma_1, gamma_phi) of the qubit of Chapter 5 are estimated,
with omega known, from N = 1000 shots of sigma_n at each of the 40 times
k tau, tau = 0.25, by weighted least squares, over 200 seeded repetitions.
The mean of sigma_n is y = (x + z)/sqrt 2 with
    x = sin(phi) e^{-Gamma t} cos(omega t),  Gamma = gamma_1/2 + gamma_phi,
    z = 1 - (1 + cos phi) e^{-gamma_1 t},
for the preparation at the angle phi from the south pole.  The estimates
scatter within the Cramer-Rao ellipse, the inverse of the Fisher information.
For phi = 15 degrees the precession is weakly excited, and gamma_phi, which
enters the data only through the decay of the precession, is poorly
determined: the ellipse is stretched along gamma_phi.
  (a) phi = 60 degrees;  (b) phi = 15 degrees;  same scales.
Writes figures/ch06_ellipse.tex.
"""
import numpy as np
from scipy.optimize import least_squares
import needham as N
from ch06_shots import tilted, expect, sample_mean, fisher, crb, CN, TAU, OMEGA, GAMMA1, GPHI

TIMES = TAU * np.arange(1, 41)
SHOTS, REPS = 1000, 200
TRUE = np.array([GAMMA1, GPHI])


def mean_n(th, phi):
    g1, gp = th
    G = g1 / 2 + gp
    x = np.sin(phi) * np.exp(-G * TIMES) * np.cos(OMEGA * TIMES)
    z = 1 - (1 + np.cos(phi)) * np.exp(-g1 * TIMES)
    return (x + z) / np.sqrt(2)


def estimates(phi, rng):
    out = []
    for _ in range(REPS):
        yh = sample_mean(mean_n(TRUE, phi), SHOTS, rng)
        res = lambda th: (yh - mean_n(th, phi)) / np.sqrt(np.clip(1 - mean_n(th, phi) ** 2, 1e-9, None))
        out.append(least_squares(res, TRUE, bounds=([1e-4, 1e-4], [2.0, 3.0])).x)
    return np.array(out)


def ellipse(C, k, n=120):
    L = np.linalg.cholesky(C)
    s = np.linspace(0, 2 * np.pi, n)
    return TRUE[None, :] + k * (L @ np.vstack([np.cos(s), np.sin(s)])).T


def main():
    rng = np.random.default_rng(7)
    xl, yl = (0.188, 0.212), (0.12, 0.68)
    b, sds = [], {}
    for j, (deg, col) in enumerate([(60, "ndBlue"), (15, "ndRed")]):
        phi = np.radians(deg)
        assert np.allclose(mean_n(TRUE, phi), expect(np.array([OMEGA, GAMMA1, GPHI]), CN, tilted(phi), TIMES))
        C = crb(fisher(lambda th: mean_n(th, phi), TRUE, SHOTS))
        est = estimates(phi, rng)
        Cs = np.cov(est.T)
        ratio = np.sqrt(np.diag(Cs) / np.diag(C))
        assert np.all(np.abs(ratio - 1) < 0.3), ratio                    # the scatter fills the bound
        sds[deg] = np.sqrt(np.diag(C))
        Pl = N.Plot(xl, yl, 4.9, 4.6, origin=(6.6 * j, 0.0))
        b += Pl.axes("$\\gamma_1$", "$\\gamma_\\phi$", xticks=[(0.19, "$0.19$"), (0.2, "$0.2$"), (0.21, "$0.21$")],
                     yticks=[(0.2, "$0.2$"), (0.4, "$0.4$"), (0.6, "$0.6$")])
        inside = (est[:, 0] > xl[0]) & (est[:, 0] < xl[1]) & (est[:, 1] > yl[0]) & (est[:, 1] < yl[1])
        b += ["\\fill[%s, fill opacity=0.5] %s circle (0.9pt);" % (col, Pl.pt(*e)) for e in est[inside]]
        for k, sty in [(1, "line width=1.0pt, ndInk"), (2, "line width=0.6pt, ndInk, dash pattern=on 2.5pt off 1.5pt")]:
            b.append(N.polyline([Pl.P(*p) for p in ellipse(C, k)] + [Pl.P(*ellipse(C, k)[0])], sty))
        b.append("\\node[dot, fill=ndInk, minimum size=4pt] at %s {};" % Pl.pt(*TRUE))
        b.append("\\node[font=\\small] at %s {(%s) $\\phi=%d^\\circ$};" % (N.pt(Pl.P(0.2, yl[0]) + np.array([0.0, -0.95])), "ab"[j], deg))
    assert sds[15][1] > 2.5 * sds[60][1] and abs(sds[15][0] / sds[60][0] - 1) < 0.4
    xr = 6.6 + 4.9 + 0.9
    b.append("\\node[note, anchor=north west] at (%.2f,4.75) {dots: estimates from $%d$\\\\ repetitions of $%d$ shots\\\\ at each of $%d$ times};" % (xr, REPS, SHOTS, len(TIMES)))
    b.append("\\node[note, anchor=north west] at (%.2f,3.35) {curves: the Cram\\'er--Rao\\\\ ellipses of one and two\\\\ standard deviations};" % xr)
    b.append("\\node[note, anchor=north west, text=ndRed] at (%.2f,1.95) {weakly excited, the\\\\ precession tells little\\\\ about $\\gamma_\\phi$, which\\\\ affects nothing else};" % xr)
    b.append("\\node[note, anchor=north west] at (%.2f,0.1) {the two axes have\\\\ different scales};" % xr)
    N.write("ch06_ellipse", b)
    print("written figures/ch06_ellipse.tex; Cramer-Rao standard deviations (gamma_1, gamma_phi):",
          "60 deg", np.round(sds[60], 4), "15 deg", np.round(sds[15], 4))


if __name__ == "__main__":
    main()
