"""ch06_pipeline.py -- one noise followed through the pipeline.

The set-up of ch06_budget: the qubit of Chapter 5 measured along n, K = 60
samples at tau = 0.25, N = 10^4 shots per expectation value, Hankel matrix with
R = 10 block rows and S = 40 columns, Ho--Kalman of order 4.  The same 150
noisy data sets are followed through every stage, and every panel is drawn to
the same scale, so that one sees where the error grows:
  (a) the errors of the samples, y^_k - y_k, of three data sets, with the band
      of one standard deviation sqrt((1 - y_k^2)/N);
  (b) the errors sigma^_i - sigma_i of the singular values of the Hankel
      matrix (rows i = 1..5; sigma_5 = 0), which by Weyl's inequality never
      exceed ||H^ - H||_2;
  (c) the errors of the eigenvalues mu_i = e^{lambda_i tau} of G_tau;
  (d) the errors of the poles lambda_i = log(mu_i) / tau of A;
  (e) the errors of (gamma_1, gamma_phi), read from the poles:
      gamma_1 = -lambda (the real pole), Gamma = -Re lambda (the complex pair),
      gamma_phi = Gamma - gamma_1/2.
The error of each stage (median over the data sets): eps_0 (root mean square
over the samples), ||H^ - H||_2, the largest error of the mu_i, of the
lambda_i, and of (gamma_1, gamma_phi, omega).  Above each arrow, the factor of
the bound (ch06_budget) and the ratio observed.  The last stage of the bound
passes through the reconstruction map M; here the parameters are read from the
poles, since Ho--Kalman gives A only up to similarity.
Writes figures/ch06_pipeline.tex.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
import needham as N
from ch05_spectrum import qubit_A, hankel, ho_kalman, X0, CN, TAU, POLES, OMEGA, GAMMA1, GPHI
from ch06_shots import sample_mean
from ch06_budget import K, R, S, P, NSHOTS, reconstruction_map

REPS = 150
W = 0.13                                               # half-width of every panel
BOX = 2.8                                              # side of a panel on the page (cm)
GAP = 0.7
COLS = ["ndInk!55", "ndRed", "ndBlue", "ndBlue"]       # 0, -gamma_1, -Gamma +- i omega


def run(y, H, rng):
    mu = np.exp(POLES * TAU)
    out = {"y": [], "sv": [], "mu": [], "lam": [], "par": [], "e0": [], "eh": [], "eg": [], "ea": [], "ep": []}
    sH = np.linalg.svd(H, compute_uv=False)[:5]
    for _ in range(REPS):
        yn = sample_mean(y, NSHOTS, rng)
        Hn = hankel(yn[:, None], R, S)
        Ad, _, _ = ho_kalman(yn[:, None], 4, r=R, s=S)
        m = np.linalg.eigvals(Ad).astype(complex)
        D = np.abs(m[:, None] - mu[None, :])
        i, j = linear_sum_assignment(D)
        mm = np.empty(4, complex)
        mm[j] = m[i]
        lam = np.log(mm) / TAU
        g1, gam, om = -lam[1].real, -lam[2].real, abs(lam[2].imag)
        par = np.array([g1 - GAMMA1, gam - g1 / 2 - GPHI, om - OMEGA])
        out["y"].append(yn - y)
        out["sv"].append(np.linalg.svd(Hn, compute_uv=False)[:5] - sH)
        out["mu"].append(mm - mu)
        out["lam"].append(lam - POLES)
        out["par"].append(par)
        out["e0"].append(np.sqrt(np.mean((yn - y) ** 2)))
        out["eh"].append(np.linalg.norm(Hn - H, 2))
        out["eg"].append(np.abs(mm - mu).max())
        out["ea"].append(np.abs(lam - POLES).max())
        out["ep"].append(np.abs(par).max())
    return {k: np.array(v) for k, v in out.items()}


def frame(x0):
    """Page origin of panel x0 and a map from errors in [-W, W]^2 to the page."""
    c = np.array([x0 + BOX / 2, BOX / 2])
    f = lambda u, v: c + (BOX / (2 * W)) * np.array([u, v])
    return c, f


def main():
    G = expm(qubit_A() * TAU)
    y = np.array([(CN @ np.linalg.matrix_power(G, k) @ X0).item() for k in range(K)])
    H = hankel(y[:, None], R, S)
    sH = np.linalg.svd(H, compute_uv=False)
    _, V = np.linalg.eig(qubit_A())
    kg = np.linalg.cond(V / np.linalg.norm(V, axis=0))
    kM = np.linalg.cond(reconstruction_map()[0])
    factors = [np.sqrt(R * S * P), 1 / sH[3], kg / TAU, kM]
    rng = np.random.default_rng(4)
    d = run(y, H, rng)
    stage = np.array([np.median(d[k]) for k in ("e0", "eh", "eg", "ea", "ep")])
    seen = stage[1:] / stage[:-1]
    assert np.all(np.abs(d["sv"]) <= d["eh"][:, None] + 1e-12)          # Weyl
    assert np.all(seen < np.array(factors))                              # each stage is better than its bound
    assert seen[1] < 0.2 and 3.5 < seen[2] < 1.2 / TAU                   # the eigenvalues average; the logarithm divides by tau
    inside = [np.mean(np.max(np.abs(np.c_[z.real, z.imag]), axis=1) < W) for z in (d["mu"].ravel(), d["lam"].ravel())]
    bias = d["par"].mean(axis=0)
    assert 0.01 < bias[0] < 0.03 and abs(bias[1]) < 0.01           # the relaxation rate comes out too large
    assert min(inside) > 0.99 and np.mean(np.abs(d["par"][:, :2]).max(axis=1) < W) > 0.99
    rms = lambda a: np.sqrt(np.mean(np.abs(a) ** 2, axis=0))
    print("stage errors (median):", np.round(stage, 4))
    print("bound factors", np.round(factors, 2), "observed ratios", np.round(seen, 3))
    print("mean error of (g1, gphi, omega)", np.round(d["par"].mean(axis=0), 4))
    print("rms of mu", np.round(rms(d["mu"]), 4), "of lambda", np.round(rms(d["lam"]), 4), "of (g1, gphi, omega)", np.round(rms(d["par"]), 4))

    b = []
    xs = [k * (BOX + GAP) for k in range(5)]
    titles = ["(a) the samples", "(b) the Hankel matrix", "(c) $\\mathbf G_\\tau$: $\\mu_i=\\ee^{\\lambda_i\\tau}$",
              "(d) $\\mathbf A$: the poles $\\lambda_i$", "(e) $(\\gamma_1,\\gamma_\\phi)$"]
    measures = ["$\\eps_0=%.3f$", "$\\lVert\\hat\\Hankel-\\Hankel\\rVert_2=%.3f$", "$\\max\\lvert\\hat\\mu_i-\\mu_i\\rvert=%.4f$",
                "$\\max\\lvert\\hat\\lambda_i-\\lambda_i\\rvert=%.3f$", "largest error $%.3f$"]
    for x0, t, m, v in zip(xs, titles, measures, stage):
        b.append("\\draw[faint, fill=ndPaper!60] (%.3f,0) rectangle (%.3f,%.3f);" % (x0, x0 + BOX, BOX))
        b.append("\\node[font=\\small, anchor=north] at (%.3f,-0.08) {%s};" % (x0 + BOX / 2, t))
        b.append("\\node[font=\\scriptsize, anchor=north, text=ndInk!80] at (%.3f,-0.55) {%s};" % (x0 + BOX / 2, m % v))
    # ---- (a) errors of the samples against time
    c, f = frame(xs[0])
    t = TAU * np.arange(K)
    tx = lambda tk: xs[0] + 0.12 + (BOX - 0.24) * tk / t[-1]
    sd = np.sqrt((1 - y ** 2) / NSHOTS)
    band = [(tx(a), f(0, s)[1]) for a, s in zip(t, sd)] + [(tx(a), f(0, -s)[1]) for a, s in zip(t[::-1], sd[::-1])]
    b.append("\\fill[ndSky] %s -- cycle;" % " -- ".join(N.pt(q) for q in band))
    b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (xs[0], c[1], xs[0] + BOX, c[1]))
    for k, col in zip(range(3), ("ndRed", "ndGold", "ndGreen")):
        b += ["\\fill[%s] (%.3f,%.3f) circle (0.55pt);" % (col, tx(a), f(0, e)[1]) for a, e in zip(t, d["y"][k])]
    b.append("\\node[font=\\scriptsize, anchor=south east] at (%.3f,0.03) {$t\\to$};" % (xs[0] + BOX - 0.05))
    # ---- (b) errors of the singular values, one row each
    c, f = frame(xs[1])
    rows = np.linspace(BOX - 0.4, 0.4, 5)
    b.append("\\draw[faint] (%.3f,0) -- (%.3f,%.3f);" % (c[0], c[0], BOX))
    for i, yr in enumerate(rows):
        col = "ndInk!70" if i < 4 else "ndRed"
        b += ["\\fill[%s, opacity=0.5] (%.3f,%.3f) circle (0.6pt);" % (col, f(e, 0)[0], yr + 0.1 * (rng.random() - 0.5)) for e in d["sv"][:, i]]
        lab = "$\\sigma_%d$" % (i + 1) if i < 4 else "$\\sigma_5=0$: the floor"
        b.append("\\node[font=\\tiny, anchor=west, text=%s] at (%.3f,%.3f) {%s};" % (col, xs[1] + 0.02, yr + 0.12, lab))
    # ---- (c) and (d): clouds of the errors of the eigenvalues and of the poles
    for x0, key, thumb in ((xs[2], "mu", "z"), (xs[3], "lam", "lam")):
        c, f = frame(x0)
        b.append("\\draw[faint] (%.3f,0) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);" % (c[0], c[0], BOX, x0, c[1], x0 + BOX, c[1]))
        for i in (0, 2, 3, 1):
            b += ["\\fill[%s, opacity=0.55] %s circle (0.6pt);" % (COLS[i], N.pt(f(z.real, z.imag))) for z in d[key][:, i]]
        # a thumbnail of where the four values lie
        o = np.array([x0 + 0.42, BOX - 0.42])
        if thumb == "z":
            b.append("\\draw[faint, fill=white] (%.3f,%.3f) circle (0.32);" % tuple(o))
            pts = [o + 0.32 * np.array([z.real, z.imag]) for z in np.exp(POLES * TAU)]
        else:
            b.append("\\fill[white] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (o[0] - 0.36, o[1] - 0.34, o[0] + 0.12, o[1] + 0.34))
            b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);" % (o[0] - 0.34, o[1], o[0] + 0.1, o[1], o[0], o[1] - 0.32, o[0], o[1] + 0.32))
            pts = [o + np.array([0.55 * z.real, 0.1 * z.imag]) for z in POLES]
        for q, col in zip(pts, COLS):
            b.append("\\fill[%s] %s circle (0.9pt);" % (col, N.pt(q)))
    # ---- (e) the errors of (gamma_1, gamma_phi)
    c, f = frame(xs[4])
    b.append("\\draw[faint] (%.3f,0) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);" % (c[0], c[0], BOX, xs[4], c[1], xs[4] + BOX, c[1]))
    b += ["\\fill[ndGreen, opacity=0.6] %s circle (0.6pt);" % N.pt(f(p[0], p[1])) for p in d["par"]]
    pm = f(*d["par"][:, :2].mean(axis=0))
    b.append("\\draw[ink, line width=0.8pt] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);" % (pm[0] - 0.09, pm[1], pm[0] + 0.09, pm[1], pm[0], pm[1] - 0.09, pm[0], pm[1] + 0.09))
    b.append("\\node[font=\\tiny, anchor=north] at (%.3f,%.3f) {mean};" % (pm[0], pm[1] - 0.45))
    b.append("\\draw[pointer] (%.3f,%.3f) -- (%.3f,%.3f);" % (pm[0], pm[1] - 0.45, pm[0], pm[1] - 0.12))
    b.append("\\node[font=\\tiny, anchor=south east] at (%.3f,%.3f) {$\\gamma_1$};" % (xs[4] + BOX - 0.03, c[1] + 0.02))
    b.append("\\node[font=\\tiny, anchor=north west] at (%.3f,%.3f) {$\\gamma_\\phi$};" % (c[0] + 0.03, BOX - 0.03))
    # ---- arrows, with the factor of the bound and the ratio observed
    for k in range(4):
        xa, xb = xs[k] + BOX + 0.08, xs[k + 1] - 0.08
        b.append("\\draw[vec, ndInk, line width=0.9pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xa, BOX / 2, xb, BOX / 2))
        fv = factors[k]
        ftxt = "%.3g" % fv
        b.append("\\node[font=\\scriptsize, align=center, anchor=south] at (%.3f,%.3f) {\\textcolor{ndRed}{bound $\\times%s$}\\\\ \\textcolor{ndBlue}{seen $\\times%s$}};"
                 % ((xa + xb) / 2, BOX + 0.08, ftxt, ("%.2f" % seen[k]) if seen[k] < 1 else ("%.1f" % seen[k])))
    # scale bar under (a)
    sx = BOX / (2 * W)
    b.append("\\draw[ink, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xs[0], -1.15, xs[0] + 0.05 * sx, -1.15))
    b.append("\\node[note, anchor=west] at (%.3f,%.3f) {$0.05$: every panel is drawn to this scale};" % (xs[0] + 0.05 * sx + 0.08, -1.15))
    N.write("ch06_pipeline", b)
    print("written figures/ch06_pipeline.tex")


if __name__ == "__main__":
    main()
