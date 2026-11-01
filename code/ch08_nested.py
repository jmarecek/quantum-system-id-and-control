"""ch08_nested.py -- a divisible family of channels has nested images.

If Phi_t = Lambda_{t,s} Phi_s with a channel (or merely a positive,
trace-preserving map) Lambda_{t,s} for every s <= t, then the image of the
Bloch ball under Phi_t lies inside its image under Phi_s.

(a) The qubit of Chapter 5 (omega = 3, gamma1 = 0.2, gphi = 0.4), a
    semigroup.  The (x,z) section of the image of the ball under Phi_t is the
    ellipse with semi-axes e^{-Gamma t} and e^{-gamma1 t}, centred at
    (0, 1 - e^{-gamma1 t}); the precession about z does not change it.  The
    ellipses for t = 0.5, 1, 2, 4 are nested and close in on |0>.
(b) The qubit with the hidden defect of Chapter 4 (g = 1, kappa = 0.8, the
    defect in |0>).  In the frame rotating at omega the reduced channel is
    amplitude damping with the real amplitude
        c(t) = e^{-kappa t/4} (cos W t + kappa/(4W) sin W t),  W = sqrt(g^2 - kappa^2/16),
    whose image is the ellipse with semi-axes |c| and c^2 centred at
    (0, 1 - c^2).  At t = 1.81, where c = 0, the image collapses to the point
    |0>; at t = 3.21 it has grown again.  No map whatever can take a point to
    a larger set, so the family is not divisible at all.
(c) The time-local rate gamma(t) = -2 d/dt ln|c(t)| of the defect, infinite at
    the zeros of c and negative while |c| grows, against the constant rate
    gamma1 of the qubit of Chapter 5.
Writes figures/ch08_nested.tex.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
import tikzexport as T
import needham as N
import ch05_spectrum as Q                    # OMEGA = 3, GAMMA1 = 0.2, GPHI = 0.4, GAMMA = 0.5
import ch07_defect as D                      # amplitude(t), defect_generator, evolve, reduced

I2, X, Y, Z = T.pauli()


def affine_map(Phi):
    """(M, c) of a qubit channel given as a function on 2x2 matrices: r -> M r + c."""
    c = np.array([np.real(np.trace(P @ Phi(I2 / 2))) for P in (X, Y, Z)])
    M = np.array([[np.real(np.trace(P @ Phi(S / 2))) for S in (X, Y, Z)] for P in (X, Y, Z)])
    return M, c


def ellipse(a, b, zc, n=181):
    th = np.linspace(0, 2 * np.pi, n)
    return np.stack([a * np.cos(th), zc + b * np.sin(th)], 1)


def inside(pts, a, b, zc, tol=1e-9):
    """Are the points (x, z) inside the ellipse with semi-axes a, b and centre (0, zc)?"""
    if a < 1e-12 or b < 1e-12:
        return np.all(np.abs(pts[:, 0]) < tol) and np.all(np.abs(pts[:, 1] - zc) < tol)
    return np.all((pts[:, 0] / a) ** 2 + ((pts[:, 1] - zc) / b) ** 2 <= 1 + tol)


def rate(t, h=1e-6):
    """gamma(t) = -2 c'(t)/c(t) for the defect, by the closed form of c'."""
    g, k = 1.0, 0.8
    W = np.sqrt(g ** 2 - k ** 2 / 16)
    c = D.amplitude(t)
    dc = -k / 4 * c + np.exp(-k * t / 4) * (-W * np.sin(W * t) + k / 4 * np.cos(W * t))
    return -2 * dc / c


def checks():
    G1, GAM = Q.GAMMA1, Q.GAMMA
    # (a) the semigroup: the affine map of e^{At} has the stated ellipse, and the images are nested
    A = Q.qubit_A()
    for t in (0.5, 1, 2, 4):
        E = expm(A * t)
        M, c = E[:3, :3], E[:3, 3]
        sv = np.linalg.svd(M, compute_uv=False)
        assert np.allclose(sorted(sv), sorted([np.exp(-GAM * t)] * 2 + [np.exp(-G1 * t)]))
        assert np.allclose(c, [0, 0, 1 - np.exp(-G1 * t)])
    ts = np.linspace(0, 6, 61)
    for s, t in zip(ts[:-1], ts[1:]):
        assert inside(ellipse(np.exp(-GAM * t), np.exp(-G1 * t), 1 - np.exp(-G1 * t)),
                      np.exp(-GAM * s), np.exp(-G1 * s), 1 - np.exp(-G1 * s))
    # (b) the defect: the reduced channel is amplitude damping with amplitude c(t)
    Lg = D.defect_generator(omega=0.0)
    tt = np.linspace(0, 4, 401)
    zero = np.array([[1, 0], [0, 0]], dtype=complex)                  # the defect starts in |0>
    out = {}
    for name, rho in [("I", I2 / 2), ("X", X / 2), ("Y", Y / 2), ("Z", Z / 2)]:
        out[name] = [D.reduced(r) for r in D.evolve(Lg, np.kron(rho, zero), tt)]
    for k, t in enumerate(tt):
        M, c0 = affine_map(lambda S: sum(np.trace(S @ P) * out[n][k] for P, n in
                                         [(I2, "I"), (X, "X"), (Y, "Y"), (Z, "Z")]))
        cc = D.amplitude(t)
        assert np.allclose(M, np.diag([cc, cc, cc ** 2]), atol=1e-9)
        assert np.allclose(c0, [0, 0, 1 - cc ** 2], atol=1e-9)
    t0 = brentq(D.amplitude, 1, 2.5)
    t1 = brentq(lambda t: rate(t), 2.5, 4.5)                        # the turning point of c
    assert abs(t0 - 1.81) < 0.005 and abs(t1 - 3.21) < 0.005
    big = ellipse(abs(D.amplitude(t1)), D.amplitude(t1) ** 2, 1 - D.amplitude(t1) ** 2)
    assert not inside(big, 0.0, 0.0, 1.0)                           # larger than the point at t0
    # (c) the rate against a finite difference of -2 ln|c|
    for t in (0.5, 1.2, 2.4, 3.0, 4.0, 6.0):
        fd = -2 * (np.log(abs(D.amplitude(t + 1e-6))) - np.log(abs(D.amplitude(t - 1e-6)))) / 2e-6
        assert abs(fd - rate(t)) < 1e-5 * max(1, abs(fd))
    neg = [t for t in np.linspace(0.01, 10, 2000) if rate(t) < 0]
    assert len(neg) > 0 and min(neg) > t0
    return t0, t1


def disc(C, R):
    """The (x,z) section of the Bloch ball, with its axes and the pole |0>."""
    b = ["\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$x$};"
         % (N.pt(C + [-R - 0.2, 0]), N.pt(C + [R + 0.3, 0])),
         "\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$z$};"
         % (N.pt(C + [0, -R - 0.2]), N.pt(C + [0, R + 0.35])),
         "\\draw[ink] %s circle (%.3f);" % (N.pt(C), R)]
    return b


def main():
    t0, t1 = checks()
    G1, GAM = Q.GAMMA1, Q.GAMMA
    b = []
    R = 1.75
    # ---- (a) the semigroup of Chapter 5
    C = np.array([0.0, 0.0])
    b += ["\\fill[ndSky, fill opacity=0.35] %s circle (%.3f);" % (N.pt(C), R)]
    times = [0.5, 1, 2, 4]
    shades = [30, 45, 60, 80]
    for t, sh in zip(times, shades):
        a, bb, zc = np.exp(-GAM * t), np.exp(-G1 * t), 1 - np.exp(-G1 * t)
        pts = [C + R * p for p in ellipse(a, bb, zc)]
        b.append(N.polyline(pts, "line width=0.7pt, ndBlue, fill=ndBlue!%d, fill opacity=0.5" % (sh // 2)))
    b += disc(C, R)
    for t in times:                                          # labels just above the lowest point of each ellipse
        bb, zc = np.exp(-G1 * t), 1 - np.exp(-G1 * t)
        p = C + R * np.array([0.0, zc - bb])
        b.append("\\node[font=\\scriptsize, text=ndBlue!80!ndInk, anchor=south west, inner sep=1pt] at %s {$%s$};"
                 % (N.pt(p + [0.04, 0.02]), ("t=%g" % t) if t == times[0] else "%g" % t))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % N.pt(C + [0, R]))
    b.append("\\node[font=\\scriptsize, anchor=south east] at %s {$\\ket0$};" % N.pt(C + [-0.05, R + 0.02]))
    b.append("\\node[note, anchor=north] at %s {each image lies inside\\\\ the one before};" % N.pt(C + [0, -R - 0.3]))
    b.append("\\node[font=\\small] at %s {(a) a semigroup (Chapter 5)};" % N.pt(C + [0, -R - 1.3]))

    # ---- (b) the defect
    C = np.array([5.4, 0.0])
    b += ["\\fill[ndSky, fill opacity=0.35] %s circle (%.3f);" % (N.pt(C), R)]
    c08 = D.amplitude(0.8)
    pts = [C + R * p for p in ellipse(abs(c08), c08 ** 2, 1 - c08 ** 2)]
    b.append(N.polyline(pts, "line width=0.7pt, ndBlue, fill=ndBlue!25, fill opacity=0.5"))
    c1 = D.amplitude(t1)
    pts = [C + R * p for p in ellipse(abs(c1), c1 ** 2, 1 - c1 ** 2)]
    b.append(N.polyline(pts, "line width=1.0pt, ndRed, dash pattern=on 3pt off 2pt, fill=ndRose, fill opacity=0.45"))
    b += disc(C, R)
    b.append("\\node[dot, fill=ndRed, minimum size=4.4pt] at %s {};" % N.pt(C + [0, R]))
    b.append("\\node[font=\\scriptsize, anchor=south east] at %s {$\\ket0$};" % N.pt(C + [-0.05, R + 0.02]))
    p08 = C + R * np.array([abs(c08), 1 - c08 ** 2])
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=west, inner sep=1pt] at %s {$t=0.8$};" % N.pt(p08 + [0.1, -0.05]))
    p1 = C + R * np.array([-abs(c1) * np.cos(0.9), 1 - c1 ** 2 + c1 ** 2 * np.sin(0.9)])
    b.append("\\node[note, anchor=east, text=ndRed!85!ndInk] (grow) at %s {$t=%.2f$:\\\\ larger again};" % (N.pt(C + [-0.55, R + 0.55]), t1))
    b.append("\\draw[pointer] (grow.south) to[bend right=20] %s;" % N.pt(p1 + [-0.02, 0.02]))
    b.append("\\node[note, anchor=west, text=ndRed!85!ndInk] (col) at %s {$t=%.2f$: the image\\\\ is the point $\\ket0$};"
             % (N.pt(C + [0.45, R + 0.55]), t0))
    b.append("\\draw[pointer] (col.west) to[bend right=25] %s;" % N.pt(C + [0.1, R + 0.06]))
    b.append("\\node[note, anchor=north] at %s {after the collapse the image\\\\ grows again: not divisible};"
             % N.pt(C + [0, -R - 0.3]))
    b.append("\\node[font=\\small] at %s {(b) the hidden defect (Chapter 4)};" % N.pt(C + [0, -R - 1.3]))

    # ---- (c) the time-local rate
    P = N.Plot((0.0, 10.0), (-3.0, 4.0), 5.0, 4.0, origin=(10.0, -2.05))
    b += P.axes("$t$", "$\\gamma(t)$", xticks=[(k, "$%d$" % k) for k in (2, 4, 6, 8, 10)],
                yticks=[(-2, "$-2$"), (2, "$2$"), (4, "$4$")], yzero=0.0)
    zeros = [brentq(D.amplitude, a, bb) for a, bb in [(1, 2.5), (4, 6), (7.5, 9)]]
    edges = [0.0] + zeros + [10.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        tt = np.linspace(lo + 1e-4, hi - 1e-4, 600)
        g = rate(tt)
        keep = (g > -3.0) & (g < 4.0)
        idx = np.where(keep)[0]
        if len(idx) == 0:
            continue
        for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(seg) < 2:
                continue
            ts, gs = tt[seg], g[seg]
            negm = gs < 0
            if negm.any():                                     # shade where the rate is negative
                j = np.where(negm)[0]
                for sub in np.split(j, np.where(np.diff(j) != 1)[0] + 1):
                    poly = [P.P(ts[sub[0]], 0)] + [P.P(ts[i], gs[i]) for i in sub] + [P.P(ts[sub[-1]], 0)]
                    b.append("\\fill[ndRose, fill opacity=0.5] %s -- cycle;" % " -- ".join(N.pt(q) for q in poly))
            b.append(P.curve(ts, gs, "line width=1.1pt, ndRed"))
    for z0 in zeros:
        b.append("\\draw[ndRed!60, densely dotted, line width=0.5pt] %s -- %s;" % (P.pt(z0, -3.0), P.pt(z0, 4.0)))
    b.append("\\node[font=\\tiny, text=ndRed!80!ndInk, anchor=north] at %s {$c=0$};" % P.pt(zeros[0], -3.05))
    b.append("\\draw[ndBlue, line width=1.0pt, dash pattern=on 3pt off 2pt] %s -- %s;" % (P.pt(0, G1), P.pt(10, G1)))
    b.append("\\node[note, anchor=west, text=ndBlue] (sg) at %s {Chapter 5: $\\gamma_1=0.2$};" % P.pt(5.6, 3.3))
    b.append("\\draw[pointer] (sg.south) to[bend left=15] %s;" % P.pt(6.6, G1 + 0.05))
    b.append("\\node[note, anchor=north, text=ndRed!85!ndInk] at %s {negative while $|c|$ grows};" % P.pt(5.0, -3.25))
    b.append("\\node[font=\\small] at %s {(c) $\\gamma(t)=-2\\,\\tfrac{\\mathrm d}{\\mathrm dt}\\ln|c(t)|$ for the defect};"
             % N.pt(np.array([P.P(5.0, 0)[0], -R - 1.3])))
    N.write("ch08_nested", b)
    print("defect: c = 0 at t = %.3f, turning point t = %.3f with |c| = %.3f; zeros of c at %s"
          % (t0, t1, abs(D.amplitude(t1)), np.round(zeros, 3)))
    print("written figures/ch08_nested.tex")


if __name__ == "__main__":
    main()
