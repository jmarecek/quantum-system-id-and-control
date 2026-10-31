"""ch06_reconstruction.py -- the reconstruction magnifies the error by at most kappa(M).

The reconstruction map M of ch06_budget sends the 12 parameters p = (theta, K)
of the qubit of Chapter 5 to the 12 numbers vec(A, beta) of x' = A x + beta.
Its singular values are 2 (the three Hamiltonian coefficients and the trace of
K) and 1 (the rest of K), so that kappa(M) = 2: an error of size rho in vec A
becomes an error between rho/2 and rho in p.
  (a) A circle of errors of size rho = 0.05 in vec A (left) and its image
      under M^{-1}, read in the plane of (gamma_1, gamma_phi) (right): an
      ellipse with semi-axes rho and rho/sqrt 2, between the circles of
      radii rho/sigma_max(M) = rho/2 and rho/sigma_min(M) = rho.  The cloud:
      the images of 400 errors of size rho in random directions of all twelve.
      (gamma_1 and gamma_phi are orthonormal coordinates of p, so that the
      plane is read by orthogonal projection.)
  (b) The bound eq:ch06:error_bound for the error of p, with
      ||vec A~ - vec A|| = rho and ||M~^{-1}|| <= ||M^{-1}|| / (1 - kappa delta),
      against the relative rounding delta = ||Delta M|| / ||M|| of the map;
      its two terms; and the errors of p for random Delta M and random errors
      of A of the same sizes (dots).  The bound holds, and it becomes infinite
      at delta = 1 / kappa(M), where M~ may be singular.
Writes figures/ch06_reconstruction.tex.
"""
import numpy as np
import needham as N
from ch06_budget import reconstruction_map, true_parameters, a_beta
from ch05_spectrum import GAMMA1, GPHI

RHO = 0.05


def main():
    M, E = reconstruction_map()
    p, theta, Kmat = true_parameters(E)
    A, beta = a_beta(theta, Kmat)
    va = np.r_[A.ravel(), beta]
    s = np.linalg.svd(M, compute_uv=False)
    kM = s[0] / s[-1]
    assert np.allclose(s, [2] * 4 + [1] * 8) and abs(kM - 2) < 1e-12
    Mi = np.linalg.inv(M)
    # gamma_1 and gamma_phi as directions in p: K = gamma_1 c c^dag + gamma_phi e_z e_z^T
    u1, u2 = np.zeros(12), np.zeros(12)
    u1[[3, 4]], u1[7], u2[5] = 0.5, np.sqrt(0.5), 1.0
    U = np.c_[u1, u2]
    assert np.allclose(U.T @ U, np.eye(2)) and np.allclose(U.T @ p, [GAMMA1, GPHI])
    L = U.T @ Mi                                                  # vec A -> (gamma_1, gamma_phi)
    Lu, Ls, Lvt = np.linalg.svd(L, full_matrices=False)
    assert np.allclose(sorted(Ls), [np.sqrt(0.5), 1.0])          # semi-axes rho/sqrt2 and rho
    rng = np.random.default_rng(3)
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    ring = [RHO * (np.cos(a) * Lvt[0] + np.sin(a) * Lvt[1]) for a in ang]
    img = np.array([L @ e for e in ring])
    cloud = rng.standard_normal((400, 12))
    cloud = RHO * cloud / np.linalg.norm(cloud, axis=1)[:, None]
    cimg = cloud @ L.T
    assert np.all(np.linalg.norm(cimg, axis=1) <= RHO / s[-1] + 1e-12)
    print("kappa(M) = %.1f; ellipse semi-axes %.4f, %.4f; cloud rms %.4f" % (kM, *(RHO * np.sort(Ls)[::-1]), np.sqrt(np.mean(np.sum(cimg ** 2, 1)))))

    # ---- (b) the bound against the rounding of M
    nA = np.linalg.norm(va)
    deltas = np.logspace(-6, np.log10(0.4999), 400)
    t1 = np.linalg.norm(Mi, 2) * RHO / (1 - kM * deltas)
    t2 = kM / (1 / deltas - kM) * np.linalg.norm(Mi, 2) * nA
    dots = []
    for d in np.logspace(-5.5, np.log10(0.45), 14):
        for _ in range(12):
            D = rng.standard_normal((12, 12))
            D *= d * np.linalg.norm(M, 2) / np.linalg.norm(D, 2)
            e = rng.standard_normal(12)
            e *= RHO / np.linalg.norm(e)
            Mt = M + D
            err = np.linalg.norm(np.linalg.solve(Mt, va + e) - p)
            rhs = np.linalg.norm(np.linalg.inv(Mt), 2) * RHO + kM / (1 / d - kM) * np.linalg.norm(Mi, 2) * nA
            assert err <= rhs and rhs <= np.interp(d, deltas, t1 + t2) * (1 + 1e-9)    # (7.1) holds
            dots.append((d, err))
    dots = np.array(dots)
    cross = RHO / (kM * nA)                                       # where the two terms are equal, roughly
    print("||vec A|| = %.3f; the rounding term overtakes at delta ~ %.1e" % (nA, cross))

    b = []
    # ---- (a) left: a circle of errors of A
    cx, cy, rdisc = 1.15, 2.45, 0.95
    cols = ["ndRed!%d!ndBlue" % int(100 * k / 11) for k in range(12)]
    b.append("\\fill[ndSky, fill opacity=0.5] (%.2f,%.2f) circle (%.2f);" % (cx, cy, rdisc))
    b.append("\\draw[ink] (%.2f,%.2f) circle (%.2f);" % (cx, cy, rdisc))
    b.append("\\node[dot, minimum size=4pt] at (%.2f,%.2f) {};" % (cx, cy))
    for a, col in zip(ang, cols):
        b.append("\\fill[%s] (%.3f,%.3f) circle (1.6pt);" % (col, cx + rdisc * np.cos(a), cy + rdisc * np.sin(a)))
    b.append("\\draw[pointer, <->] (%.2f,%.2f) -- (%.2f,%.2f) node[midway, below, font=\\scriptsize, text=ndInk] {$\\rho$};" % (cx, cy - 0.02, cx + rdisc * np.cos(ang[10]), cy + rdisc * np.sin(ang[10])))
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {errors of size $\\rho$\\\\ in $\\mathrm{vec}\\,\\mathbf A$};" % (cx, cy - rdisc - 0.12))
    b.append("\\draw[vec, ndInk] (%.2f,%.2f) to[bend left=20] node[midway, above, font=\\small] {$\\mathbf M^{-1}$} (%.2f,%.2f);" % (cx + rdisc + 0.15, cy + 0.3, 3.3, cy + 0.3))
    # ---- (a) right: the plane of (gamma_1, gamma_phi)
    w = 0.14
    Pl = N.Plot((GAMMA1 - w / 2, GAMMA1 + w / 2), (GPHI - w / 2, GPHI + w / 2), 4.3, 4.3, origin=(3.55, 0.25))
    sc = Pl.sx
    ctr = Pl.P(GAMMA1, GPHI)
    b.append("\\draw[dashedink, ndGreen] %s circle (%.3f);" % (N.pt(ctr), RHO / s[0] * sc))
    b.append("\\draw[dashedink, ndRed] %s circle (%.3f);" % (N.pt(ctr), RHO / s[-1] * sc))
    for q in cimg:
        b.append("\\fill[ndBlue!45] %s circle (0.55pt);" % Pl.pt(GAMMA1 + q[0], GPHI + q[1]))
    tt = np.linspace(0, 2 * np.pi, 120)
    ell = [Pl.P(GAMMA1 + q[0], GPHI + q[1]) for q in (RHO * (np.outer(np.cos(tt), Lu[:, 0] * Ls[0]) + np.outer(np.sin(tt), Lu[:, 1] * Ls[1])))]
    b.append(N.polyline(ell + [ell[0]], "line width=1.1pt, ndBlue"))
    for q, col in zip(img, cols):
        b.append("\\fill[%s] %s circle (1.6pt);" % (col, Pl.pt(GAMMA1 + q[0], GPHI + q[1])))
    b.append("\\node[dot, minimum size=4pt] at %s {};" % N.pt(ctr))
    b += Pl.axes("$\\gamma_1$", "$\\gamma_\\phi$", xticks=[(0.15, "$0.15$"), (0.2, "$0.2$"), (0.25, "$0.25$")],
                 yticks=[(0.35, "$0.35$"), (0.4, "$0.4$"), (0.45, "$0.45$")])
    b.append("\\node[note, anchor=south west, text=ndRed] (o) at %s {$\\rho/\\sigma_{\\min}(\\mathbf M)$};" % Pl.pt(GAMMA1 + 0.022, GPHI + 0.052))
    b.append("\\node[note, anchor=north east, text=ndGreen] (i) at %s {$\\rho/\\sigma_{\\max}(\\mathbf M)$};" % Pl.pt(GAMMA1 + 0.066, GPHI - 0.05))
    b.append("\\draw[pointer] (i.north) to[bend right=15] %s;" % N.pt(ctr + RHO / s[0] * sc * np.array([np.cos(-0.6), np.sin(-0.6)])))
    b.append("\\node[font=\\small] at %s {(a) a circle of errors becomes an ellipse};" % N.pt(np.array([4.0, -0.95])))
    # ---- (b)
    lo, hi = -2.0, 2.0
    Q = N.Plot((-6.0, 0.0), (lo, hi), 6.0, 4.4, origin=(10.0, 0.0))
    b += Q.axes("$\\delta$", "error of $(\\vec\\theta,\\Koss)$",
                xticks=[(k, "$10^{%d}$" % k) for k in (-6, -4, -2, 0)], yticks=[(k, "$10^{%d}$" % k) for k in (-2, -1, 0, 1, 2)])
    xs = np.log10(deltas)
    clip = lambda v: np.clip(np.log10(v), lo, hi)
    b.append("\\fill[ndRose, fill opacity=0.6] %s rectangle %s;" % (Q.pt(np.log10(1 / kM), lo), Q.pt(0.0, hi)))
    b.append(Q.curve(xs, clip(t1), "line width=0.8pt, ndBlue, dash pattern=on 3pt off 2pt"))
    keep = np.log10(t2) > lo
    b.append(Q.curve(xs[keep], clip(t2[keep]), "line width=0.8pt, ndGold, dash pattern=on 3pt off 2pt"))
    b.append(Q.curve(xs, clip(t1 + t2), "line width=1.3pt, ndInk"))
    for d, e in dots:
        b.append("\\fill[ndRed] %s circle (0.8pt);" % Q.pt(np.log10(d), np.log10(e)))
    b.append("\\draw[line width=0.7pt, ndRed] %s -- %s;" % (Q.pt(np.log10(1 / kM), lo), Q.pt(np.log10(1 / kM), hi)))
    b.append("\\node[note, anchor=north west, text=ndBlue, align=left] at %s {the error of $\\mathbf A$, magnified\\\\ by $\\lVert\\tilde{\\mathbf M}^{-1}\\rVert$};" % Q.pt(-5.95, np.log10(RHO) - 0.06))
    b.append("\\node[note, anchor=north west, text=ndGold!80!ndInk] (r) at %s {the rounding of $\\mathbf M$};" % Q.pt(-5.9, 1.9))
    b.append("\\draw[pointer] (r.south) to[bend right=25] %s;" % Q.pt(-1.5, np.log10(kM / (10 ** 1.5 - kM) * nA) + 0.08))
    b.append("\\node[note, anchor=north west, align=left] at %s {dots: random $\\Delta\\mathbf M$ and\\\\ errors of $\\mathbf A$ of size $\\rho$};" % Q.pt(-5.9, -0.42))
    b.append("\\node[note, anchor=south east, text=ndRed] (k) at %s {$\\kappa(\\mathbf M)\\lVert \\Delta\\mathbf M\\rVert $\\\\ $=\\lVert \\mathbf M\\rVert $};" % Q.pt(-0.42, 1.25))
    b.append("\\draw[pointer] (k.east) to[bend left=20] %s;" % Q.pt(np.log10(1 / kM) - 0.02, 1.5))
    b.append("\\node[font=\\small] at %s {(b) the bound against the rounding $\\delta=\\lVert\\Delta\\mathbf M\\rVert/\\lVert\\mathbf M\\rVert$};" % N.pt(Q.P(-3.0, lo) + np.array([0.0, -0.95])))
    N.write("ch06_reconstruction", b)
    print("written figures/ch06_reconstruction.tex")


if __name__ == "__main__":
    main()
