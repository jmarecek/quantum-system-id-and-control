"""ch13_noise_informed.py -- noise-blind and noise-informed control of a noisy qubit.

Example D revisited, after the experiment of Lawrence et al. (2026a).  A qubit
with H = H_d + sum_j u_j sigma_j + H_n(t), drift H_d = sigma_y/2, controls u_j
polynomials of degree 5 in t on [0, T], T = 1/2, and target
V = exp(-i pi sigma_x / 2).  The noise metric is g_Lambda = diag(1, 1/100, 1/100)
in the basis (sigma_x, sigma_y, sigma_z): a field along x brings ten times the
noise of one along y or z.  Two designs, both by BFGS from the same random
coefficients:
    noise-blind     min ||U(T) - V||_F^2,
    noise-informed  min ||U(T) - V||_F^2 + (eta L)^2,   L = int_0^T sqrt(g_Lambda(u, u)) dt.
Each design is then run with a noise
    H_n(t) = eta sqrt(g_Lambda(u(t), u(t))) sum_j s_j(t) sigma_j,
    s_j = arctan(W_j)/(2 pi),  W_j independent Gaussian processes, Matern 5/2,
so that the noise is bounded by eta times the g_Lambda-speed of the control,
and the operator-norm error ||U_noisy(T) - V|| is recorded.  The settings the
text leaves open (T, V, eta = 0.1 for the comparison, the scale of the process)
follow the authors' code (github.com/llorebaga/RandomSE); the noise term is
made Hermitian, as the random Schrodinger equation requires.
  (a) the errors of 100 pairs of designs, one noise path each: violins of
      log10 of the error, the dots, and the means (dashed);
  (b) the mean error against the worst-case signal-to-noise ratio eta in [0, 2]
      (40 runs per value; the noise-informed design is re-optimised for each
      eta, starting from its solution at the previous value).  The noise-blind
      error grows linearly in eta; the noise-informed error stays far below and
      grows a little faster than linearly, since for larger eta the design
      gives up some accuracy in the absence of noise to shorten its path.
Runs in about two minutes.  Writes figures/ch13_noise_informed.tex.
"""
import os
import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

T, NSTEP, NGRID = 0.5, 100, 1000
LAM = np.array([1.0, 0.01, 0.01])                          # g_Lambda in the basis (sigma_x, sigma_y, sigma_z)
DRIFT = np.array([0.0, 0.5, 0.0])                          # H_d = sigma_y / 2
DEG = 5
TM = (np.arange(NSTEP) + 0.5) * T / NSTEP                   # midpoints of the time steps
BASIS = np.vander(TM, DEG + 1, increasing=True)
NPAR = 3 * (DEG + 1)


# ---- SU(2) as unit quaternions: U = q0 I - i (q1 sigma_x + q2 sigma_y + q3 sigma_z)
def qmul(a, b):
    a0, av, b0, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate([a0 * b0 - np.sum(av * bv, -1, keepdims=True), a0 * bv + b0 * av + np.cross(av, bv)], -1)


def ordered_product(q):
    """q[..., N-1, :] ... q[..., 1, :] q[..., 0, :], by pairwise reduction."""
    while q.shape[-2] > 1:
        if q.shape[-2] % 2:
            one = np.zeros(q.shape[:-2] + (1, 4))
            one[..., 0] = 1.0
            q = np.concatenate([q, one], -2)
        q = qmul(q[..., 1::2, :], q[..., 0::2, :])
    return q[..., 0, :]


def propagator(P, noise=None, eta=0.0):
    """U(T) for coefficient vectors P (B x 18), optionally with the noise path `noise` (NSTEP x 3)."""
    u = np.einsum("nk,bjk->bnj", BASIS, np.atleast_2d(P).reshape(-1, 3, DEG + 1))
    h = u + DRIFT
    if noise is not None:
        h = h + eta * np.sqrt(np.sum(LAM * u ** 2, -1, keepdims=True)) * noise
    nrm = np.linalg.norm(h, axis=-1, keepdims=True)
    phi = nrm * T / NSTEP
    q = np.concatenate([np.cos(phi), np.sin(phi) / np.maximum(nrm, 1e-300) * h], -1)
    return ordered_product(q)


def length(P):
    u = np.einsum("nk,bjk->bnj", BASIS, np.atleast_2d(P).reshape(-1, 3, DEG + 1))
    return np.sum(np.sqrt(np.sum(LAM * u ** 2, -1)), -1) * T / NSTEP


def cost(P, eta):
    # V = -i sigma_x is the quaternion (0, 1, 0, 0), so ||U - V||_F^2 = 4 - 4 q1
    return 4.0 - 4.0 * propagator(P)[:, 1] + (eta * length(P)) ** 2


def error(P, noise, eta):
    """Operator-norm error ||U_noisy(T) - V|| = sqrt(2 - 2 q1) on SU(2)."""
    return np.sqrt(max(2.0 - 2.0 * propagator(P, noise, eta)[0, 1], 0.0))


def design(p0, eta, gtol=1e-6, maxiter=400):
    E = np.eye(NPAR) * 1e-6
    f = lambda x: cost(x[None], eta)[0]
    g = lambda x: (lambda v: (v[:NPAR] - v[NPAR:]) / 2e-6)(cost(np.concatenate([x + E, x - E]), eta))
    return minimize(f, p0, jac=g, method="BFGS", options=dict(gtol=gtol, maxiter=maxiter)).x


def noise_paths(m, rng):
    """m paths of (s_x, s_y, s_z) on the time steps: arctan of Matern-5/2 processes, divided by 2 pi."""
    x = np.linspace(-5.0, 5.0, NGRID)                      # the process on [-5, 5], read on [0, T]
    r = np.abs(x[:, None] - x[None, :])
    K = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(NGRID))
    idx = np.rint(TM / T * (NGRID - 1)).astype(int)
    return [np.arctan(L @ rng.standard_normal((NGRID, 3)))[idx] / (2 * np.pi) for _ in range(m)]


def violin(b, Pl, x, logs, col, wash, half=0.62):
    """A violin of the values `logs` (log10 errors) at abscissa x on the plot Pl."""
    kde = gaussian_kde(logs, bw_method=0.35)
    ys = np.linspace(logs.min(), logs.max(), 120)
    w = kde(ys)
    w = half * w / w.max()
    X0 = Pl.P(x, 0)[0]
    right = [(X0 + wi, Pl.P(0, yi)[1]) for yi, wi in zip(ys, w)]
    left = [(X0 - wi, Pl.P(0, yi)[1]) for yi, wi in zip(ys[::-1], w[::-1])]
    b.append("\\fill[%s] %s -- cycle;" % (wash, " -- ".join(N.pt(p) for p in right + left)))
    b.append("\\draw[line width=0.7pt, %s] %s -- cycle;" % (col, " -- ".join(N.pt(p) for p in right + left)))
    rng = np.random.default_rng(0)
    for yv in logs:
        wy = half * kde(yv)[0] / kde(ys).max()
        b.append("\\fill[%s, fill opacity=0.75] (%.3f,%.3f) circle (0.9pt);" % (col, X0 + rng.uniform(-0.8, 0.8) * wy, Pl.P(0, yv)[1]))


def compute():
    """The two experiments: the errors at eta = 0.1, and the mean errors against eta."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(2801)
    m, eta0 = 100, 0.1
    starts = [rng.standard_normal(NPAR) for _ in range(m)]
    paths = noise_paths(m, rng)
    blind = [design(p, 0.0) for p in starts]
    aware = [design(p, eta0) for p in starts]
    eb = np.array([error(P, z, eta0) for P, z in zip(blind, paths)])
    ea = np.array([error(P, z, eta0) for P, z in zip(aware, paths)])
    lb, la = length(np.array(blind)).mean(), length(np.array(aware)).mean()
    ratio = eb.mean() / ea.mean()
    print("eta = %.2f: mean error blind %.5f, informed %.5f, ratio %.2f (medians %.2f); g-lengths %.3f and %.3f"
          % (eta0, eb.mean(), ea.mean(), ratio, np.median(eb) / np.median(ea), lb, la))
    assert np.all(4.0 - 4.0 * propagator(np.array(blind))[:, 1] < 1e-8)     # the blind designs hit V exactly
    assert ratio > 2.0 and np.mean(ea < eb) > 0.9                              # the informed design is the more robust
    # ---- the scan over eta
    etas = np.linspace(0.0, 2.0, 9)
    ms = 40
    mb, ma = np.zeros(len(etas)), np.zeros(len(etas))
    for i in range(ms):
        P = aware[i]
        for k, eta in enumerate(etas):
            if eta > 0:
                P = design(P, eta)
            mb[k] += error(blind[i], paths[i], eta) / ms
            ma[k] += error(P if eta > 0 else blind[i], paths[i], eta) / ms
    print("eta:", etas, "\nmean error blind:   ", np.round(mb, 4), "\nmean error informed:", np.round(ma, 4))
    assert np.all(ma[1:] < mb[1:])
    assert np.corrcoef(etas, mb)[0, 1] > 0.999                       # the noise-blind error grows linearly
    assert np.all(np.diff(ma, 2) > 0)                                # the noise-informed one a little faster, from a far lower start
    return dict(eta0=eta0, eb=eb, ea=ea, ratio=ratio, etas=etas, mb=mb, ma=ma)


def draw(d):
    eb, ea, ratio, etas, mb, ma = d["eb"], d["ea"], d["ratio"], d["etas"], d["mb"], d["ma"]
    b = []
    # ---- (a) violins on a logarithmic axis
    lo, hi = np.floor(np.log10(min(ea.min(), eb.min())) * 2) / 2, np.ceil(np.log10(max(ea.max(), eb.max())) * 2) / 2
    Pa = N.Plot((0.0, 3.0), (lo, hi), 4.9, 3.8)
    b.append("\\fill[ndPaper] %s rectangle %s;" % (Pa.pt(0.0, lo), Pa.pt(3.0, hi)))
    yt = [(v, "$10^{%d}$" % v) for v in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
    b += Pa.axes("", "error $\\norm{U(T)-V}$", yticks=yt)
    violin(b, Pa, 0.8, np.log10(eb), "ndRed", "ndRose", half=0.5)
    violin(b, Pa, 2.2, np.log10(ea), "ndBlue", "ndSky", half=0.5)
    for x, e, col in [(0.8, eb, "ndRed"), (2.2, ea, "ndBlue")]:
        y = np.log10(e.mean())
        b.append("\\draw[line width=0.9pt, %s, dash pattern=on 3pt off 2pt] %s -- %s;" % (col, Pa.pt(x - 0.6, y), Pa.pt(x + 0.6, y)))
    b.append("\\node[font=\\footnotesize, anchor=north, text=ndRed] at %s {noise-blind};" % Pa.pt(0.8, lo - 0.05))
    b.append("\\node[font=\\footnotesize, anchor=north, text=ndBlue] at %s {noise-informed};" % Pa.pt(2.2, lo - 0.05))
    b.append("\\node[note, anchor=north] at %s {dashed: the means,\\\\ $%.1f$ times apart};" % (Pa.pt(2.1, hi - 0.08), ratio))
    b.append("\\node[font=\\small] at %s {(a) $100$ designs, $\\eta=%.1f$};" % (N.pt(Pa.P(1.5, lo) + np.array([0.0, -0.95])), d["eta0"]))
    # ---- (b) the mean error against eta
    top = 1.1 * mb.max()
    Pb = N.Plot((0.0, 2.0), (0.0, top), 4.4, 3.8, origin=(6.7, 0.0))
    b += Pb.axes("$\\eta$", "mean error", xticks=[(v, "$%g$" % v) for v in (0, 1, 2)],
                 yticks=[(v, "$%g$" % v) for v in np.arange(0, top, 0.1 if top > 0.2 else 0.05)[1:]])
    b.append(Pb.curve(etas, mb, "line width=0.8pt, ndRed"))
    b.append(Pb.curve(etas, ma, "line width=0.8pt, ndBlue"))
    for x, y in zip(etas, mb):
        b.append("\\fill[ndRed] %s ++(-2.2pt,-1.6pt) -- ++(4.4pt,0) -- ++(-2.2pt,3.8pt) -- cycle;" % Pb.pt(x, y))
    b += ["\\node[dot, fill=ndBlue] at %s {};" % Pb.pt(x, y) for x, y in zip(etas, ma)]
    b.append("\\node[note, anchor=west, text=ndRed] at %s {noise-blind};" % Pb.pt(2.1, mb[-1]))
    b.append("\\node[note, anchor=west, align=left, text=ndBlue] at %s {noise-informed,\\\\ re-optimised\\\\ for each $\\eta$};" % Pb.pt(2.1, ma[-1] + 0.06 * top))
    b.append("\\node[font=\\small] at %s {(b) the mean error against $\\eta$};" % N.pt(Pb.P(1.0, 0.0) + np.array([0.0, -0.95])))
    N.write("ch13_noise_informed", b)
    print("written figures/ch13_noise_informed.tex")


def main():
    draw(compute())


if __name__ == "__main__":
    main()
