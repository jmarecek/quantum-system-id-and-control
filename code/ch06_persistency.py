"""ch06_persistency.py -- persistency of excitation, made quantitative.

The third-order system of Chapter 4 (ch04_hankel: a damped rotation and a
decay, sampled at tau = 0.5, one input) now with process noise,
    x_{k+1} = G x_k + B u_k + w_k,     w_k ~ N(0, sigma_w^2 I),  sigma_w = 0.01,
is driven by the three inputs of that figure: one sinusoid, two sinusoids,
and a random input.  Least squares estimates [G B] from K samples of
z_k = (x_k, u_k), and the error is governed by the smallest eigenvalue of the
empirical covariance Sigma_K = (1/K) sum_k z_k z_k^T (eq. ch06:ls).
  (a) The states visited (k = 100, ..., 400), with the ellipsoid of two
      standard deviations of their covariance, and below each the four
      eigenvalues of Sigma_K for K = 10^4.  One sinusoid drives the state
      round a flat ellipse, and Sigma_K has two eigenvalues of the size of
      the noise; two sinusoids fill all four directions, one of them weakly;
      the random input fills them all.
  (b) The root-mean-square error ||G_hat - G||_2 over 100 repetitions against
      K (dots), with sigma_w sqrt((n + m) / (K lambda_min(Sigma_K))) (dashed),
      n + m = 4.  All fall like K^{-1/2} in the end, but at heights set by
      lambda_min: for one sinusoid only the noise itself excites the missing
      directions, and at K = 3 10^4 the error is about twenty-five times that of
      the random input.
Writes figures/ch06_persistency.tex.
"""
import numpy as np
from scipy.spatial import ConvexHull
import needham as N
from ch04_hankel import AD, BD, X0

SW = 0.01
KS = np.array([30, 100, 300, 1000, 3000, 10000, 30000])
REPS = 100
NM = 4                                         # n + m: the dimension of z_k


def inputs(K, rng):
    k = np.arange(K)
    return [("one sinusoid", lambda: np.sin(0.7 * k), "ndBlue"),
            ("two sinusoids", lambda: np.sin(0.7 * k) + np.sin(1.6 * k + 0.4), "ndGold"),
            ("random input", lambda: rng.standard_normal(K), "ndRed")]


def simulate(u, reps, rng):
    """States x_0, ..., x_K for `reps` independent noise sequences (and inputs u of shape (reps, K))."""
    K = u.shape[1]
    X = np.empty((reps, K + 1, 3))
    x = np.tile(X0, (reps, 1))
    for k in range(K):
        X[:, k] = x
        x = x @ AD.T + u[:, k:k + 1] * BD[:, 0] + SW * rng.standard_normal((reps, 3))
    X[:, K] = x
    return X


def fit(X, u, K):
    """Least-squares [G B] from the first K steps; returns the errors of G and lambda_min(Sigma_K)."""
    Z = np.concatenate([X[:, :K], u[:, :K, None]], axis=2)            # (reps, K, 4)
    ZZ = np.einsum("rki,rkj->rij", Z, Z)
    ZY = np.einsum("rki,rkj->rij", Z, X[:, 1:K + 1])
    Th = np.linalg.solve(ZZ, ZY).transpose(0, 2, 1)                   # (reps, 3, 4)
    err = np.array([np.linalg.norm(t[:, :3] - AD, 2) for t in Th])
    lmin = np.linalg.eigvalsh(ZZ / K)[:, 0]
    return err, lmin, np.linalg.eigvalsh(ZZ[0] / K)


def outline(V, centre, cov, scale=2.0, n=40):
    """Page polygon of the outline of the ellipsoid {centre + scale L s : |s| = 1}, cov = L L^T."""
    w, U = np.linalg.eigh(cov)
    L = U * np.sqrt(np.clip(w, 0, None))
    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2))
    s = np.stack([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)], -1).reshape(-1, 3)
    P = np.array([V.P(centre + scale * L @ v) for v in s])
    return [P[i] for i in ConvexHull(P).vertices]


def main():
    rng = np.random.default_rng(4)
    Kmax = KS[-1]
    rows = []
    for name, make, col in inputs(Kmax, rng):
        u = np.array([make() for _ in range(REPS)])
        X = simulate(u, REPS, rng)
        rmse, lam, spec = [], [], None
        for K in KS:
            err, lmin, sp = fit(X, u, K)
            rmse.append(np.sqrt(np.mean(err ** 2)))
            lam.append(np.median(lmin))
            if K == 10000:
                spec = sp
        rmse, lam = np.array(rmse), np.array(lam)
        pred = SW * np.sqrt(NM / (KS * lam))
        rows.append(dict(name=name, col=col, X=X[0], rmse=rmse, lam=lam, pred=pred, spec=spec))
        print("%-14s lambda_min(K=1e4) = %.2e  RMSE: %s  RMSE/pred: %s"
              % (name, lam[5], np.array2string(rmse, precision=4), np.round(rmse / pred, 2)))
    for r in rows:                                                   # the dashed line predicts the error
        assert np.all(np.abs(np.log(r["rmse"] / r["pred"])) < np.log(1.6))
    one, two, rnd = rows
    assert one["spec"][1] < 1e-2 < two["spec"][1]                    # one sinusoid: two directions at noise level
    assert one["lam"][5] < 1e-3 < two["lam"][5] < 1e-2 < 0.2 < rnd["lam"][5]
    assert one["rmse"][-1] > 10 * rnd["rmse"][-1]
    sl = np.polyfit(np.log10(KS[2:]), np.log10(rnd["rmse"][2:]), 1)[0]
    assert abs(sl + 0.5) < 0.06

    b = []
    # ---- (a) the clouds of states, and the eigenvalues of Sigma_K
    V0 = N.View(az=62.0, el=22.0, R=0.62)
    centres = [0.0, 3.05, 6.1]
    blo, bhi, bh = -3.6, 0.6, 1.25                                   # log10 range and height of the bars
    for (r, cx) in zip(rows, centres):
        V = N.View(az=62.0, el=22.0, R=0.62, shift=(cx, 1.15))
        Xs = r["X"][100:401]
        cov = np.cov(Xs.T)
        poly = outline(V, Xs.mean(0), cov)
        b.append("\\fill[%s, fill opacity=0.13] %s -- cycle;" % (r["col"], " -- ".join(N.pt(p) for p in poly)))
        b.append("\\draw[%s, line width=0.6pt] %s -- cycle;" % (r["col"], " -- ".join(N.pt(p) for p in poly)))
        b += ["\\fill[%s] %s circle (0.55pt);" % (r["col"], V.pt(x)) for x in Xs[::2]]
        # bars of the eigenvalues of Sigma_K
        y0 = -1.95
        b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f);" % (cx - 0.85, y0, cx + 0.85, y0))
        for j, v in enumerate(r["spec"]):
            x = cx - 0.6 + 0.4 * j
            h = bh * (np.log10(v) - blo) / (bhi - blo)
            b.append("\\fill[%s!75] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (r["col"], x - 0.12, y0, x + 0.12, y0 + h))
        b.append("\\node[font=\\scriptsize, text=%s] at (%.2f,%.2f) {%s};" % (r["col"], cx, -2.3, r["name"]))
    # one small triad of axes, to the left of the clouds
    V = N.View(az=62.0, el=22.0, R=0.62)
    p0 = np.array([-1.35, 0.7])
    for e, lab in zip(np.eye(3), ("$x_1$", "$x_2$", "$x_3$")):
        q = p0 + 1.0 * (V.P(e) - V.P(0 * e))
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[font=\\scriptsize, text=ndInk!70, pos=1.3] {%s};" % (N.pt(p0), N.pt(q), lab))
    # the scale of the bars, on the left
    yb = lambda v: -1.95 + bh * (np.log10(v) - blo) / (bhi - blo)
    xa = -1.15
    b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f);" % (xa, yb(10 ** blo), xa, yb(10 ** bhi)))
    for v, lab in [(1e-3, "$10^{-3}$"), (1.0, "$1$")]:
        b.append("\\draw[faint] (%.2f,%.3f) -- (%.2f,%.3f) node[left, font=\\scriptsize, text=ndInk] {%s};" % (xa + 0.05, yb(v), xa - 0.05, yb(v), lab))
    b.append("\\node[font=\\scriptsize, anchor=south, align=center] at (%.2f,%.2f) {eigenvalues\\\\ of $\\mathbf\\Sigma_K$};" % (xa - 0.15, yb(10 ** bhi) + 0.02))
    b.append("\\node[note, anchor=south] at (%.2f,%.2f) {one sinusoid keeps the state on a flat ellipse, and two sinusoids\\\\ nearly so; the random input fills every direction};"
             % (centres[1], 2.3))
    b.append("\\node[font=\\small] at (%.2f,%.2f) {(a) the states visited, and the eigenvalues of $\\mathbf\\Sigma_K$};" % (centres[1], -2.95))
    # ---- (b) the error against K
    lo, hi = -4.0, -1.0
    Q = N.Plot((1.3, 4.6), (lo, hi), 5.6, 4.6, origin=(9.6, -2.2))
    b += Q.axes("samples $K$", "error of $\\hat{\\mathbf G}$", xticks=[(k, "$10^{%d}$" % k) for k in (2, 3, 4)],
                yticks=[(k, "$10^{%d}$" % k) for k in (-4, -3, -2, -1)])
    lk = np.log10(KS)
    for r in rows:
        b.append(Q.curve(lk, np.log10(r["pred"]), "dashedink, %s" % r["col"]))
        b += ["\\fill[%s] %s circle (1.5pt);" % (r["col"], Q.pt(x, np.log10(v))) for x, v in zip(lk, r["rmse"])]
    for r, dy in zip(rows, (0.0, 0.0, 0.0)):
        b.append("\\node[font=\\scriptsize, text=%s, anchor=west] at %s {%s};" % (r["col"], Q.pt(lk[-1] + 0.08, np.log10(r["rmse"][-1]) + dy), r["name"]))
    x0, y0 = 1.75, -2.95
    b.append("\\draw[ink, line width=0.5pt] %s -- %s -- %s -- cycle;" % (Q.pt(x0, y0), Q.pt(x0 + 1, y0 - 0.5), Q.pt(x0, y0 - 0.5)))
    b.append("\\node[font=\\scriptsize, anchor=east] at %s {$\\tfrac12$};" % Q.pt(x0, y0 - 0.25))
    b.append("\\node[font=\\scriptsize, anchor=north] at %s {$1$};" % Q.pt(x0 + 0.5, y0 - 0.5))
    b.append("\\node[note, anchor=north west] (d) at %s {dashed: $\\sigma_w\\sqrt{(n+m)/(K\\lambda_{\\min})}$};" % Q.pt(1.35, -0.75))
    b.append("\\node[font=\\small] at %s {(b) the error against the number of samples};" % N.pt(Q.P(2.95, lo) + np.array([0.0, -0.75])))
    N.write("ch06_persistency", b)
    print("written figures/ch06_persistency.tex; lambda_min at K = 1e4:", ["%.1e" % r["lam"][5] for r in rows])


if __name__ == "__main__":
    main()
