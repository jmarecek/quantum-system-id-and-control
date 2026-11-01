"""ch12_robust.py -- from identification to control: the certified worst case of a pulse over a confidence set.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi) and cvxpy with
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

The driven qubit of Chapter 4, with both quadratures of the field as in
Chapter 10, seen in the frame that turns at the identified frequency: a field
of strength Omega (= 1) and phase phi gives
    H = (1/2) (Delta sigma_z + Omega (1 + eps) (cos phi sigma_x + sin phi sigma_y)),
where Delta is the error in the identified frequency and eps that in the
calibration of the field.  A segment meant to turn the qubit through theta
about an axis on the equator instead turns it through theta (1 + rho) about an
axis tilted out of the equator by chi, where
    1 + rho = sqrt((1 + eps)^2 + (Delta/Omega)^2),   tan chi = Delta / (Omega (1 + eps)).
The confidence set left by the identification is the box |rho| <= 0.1,
|chi| <= chi_m.  Three pulses that take |0> to |1> on the identified model
are compared, by the transfer infidelity |<0|U|0>|^2:
    the plain pi pulse about x;
    the composite pulse 90_x 180_y 90_x of Levitt and Freeman (1979);
    the composite pulse BB1 of Wimperis (1994), 180_phi 360_3phi 180_phi 180_0,
    phi = arccos(-1/4).
Each pulse is a sequence of D equal base steps (D = 1, 4, 5 with the base
angle pi, pi/2, pi), and the base step is the unit quaternion
    V_phi = c - i (p cos phi sigma_x + p sin phi sigma_y + q sigma_z),
    c = cos(theta_b (1 + rho)/2),  p = sin(.) cos chi,  q = sin(.) sin chi,
whatever the phase.  The box is exactly the set c in [cos(theta_b 1.1/2), cos(theta_b 0.9/2)],
p >= 0, q^2 <= tan^2(chi_m) p^2, c^2 + p^2 + q^2 = 1, and the infidelity is a
polynomial of degree 2D in (c, p, q).  Its largest value over the box is
bounded by pysparsepmi.sos_lower_bound at the order D (order 2 for the plain
pulse, for which the order 1 gives only the trivial bound 1); PSD blocks of
order 10, 35 and 56.

The certified worst cases are exact: they agree to 1e-8 with the largest
infidelity found on a 121 x 121 grid of the box refined by local search, the
qubit being evolved by 2x2 exponentials, not by the polynomial; every SCS
status is "optimal".  The worst case lies at a corner of the box.  With
chi_m = 0.1 the worst cases are 0.0342 (plain), 0.0130 (Levitt) and 0.0131
(BB1); along the axis chi = 0 alone they would be 0.0245, 6.0e-4 and 9.2e-6.
The composite pulses cancel the error in the rate, but not the tilt: their
worst cases approach sin^2(chi_m), the infidelity of a pi pulse about an axis
tilted by chi_m (within 35% for BB1 when chi_m >= 0.02, within 50% for
Levitt's pulse when chi_m >= 0.05), and as chi_m shrinks
from 0.1 to 0.01 the certified worst case of BB1 falls from 0.0131 to 1.57e-4
while that of the plain pulse stays near 0.0246.  How much a robust pulse buys
is decided by the confidence set that the identification leaves, here by how
well the frequency is known.  For thinner boxes SCS struggles: at chi_m = 0.005
it returns "optimal_inaccurate" for the plain pulse, with a value 0.0189 below
the true maximum 0.0245, which is no bound at all.

(a) The three pulses, each segment drawn in proportion to its angle.
(b) The infidelity over the box with chi_m = 0.1, on a logarithmic scale of
    colour, with the contour 10^-3 and the certified worst case (star).
(c) The certified worst case against chi_m (dots), on the largest values found
    on the grid (lines).
Writes figures/ch12_robust.tex.
"""
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp
import needham as N

RHO = 0.1
CHIS = [0.01, 0.02, 0.05, 0.1]
PHI1 = np.arccos(-0.25)
PULSES = [("plain", np.pi, [(1, 0.0)]),
          ("Levitt", np.pi / 2, [(1, 0.0), (2, np.pi / 2), (1, 0.0)]),
          ("BB1", np.pi, [(1, PHI1), (2, 3 * PHI1), (1, PHI1), (1, 0.0)])]
COLOURS = {"plain": "ndRed", "Levitt": "ndGold", "BB1": "ndBlue"}
OPTS = dict(solver="SCS", eps=1e-9, max_iters=200000)
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.diag([1.0, -1.0]).astype(complex)


def qmul(A, B):
    """(a1 - i v1.sigma)(a2 - i v2.sigma) for quaternions (a, b, c, d) = a - i(b X + c Y + d Z)."""
    a1, b1, c1, d1 = A
    a2, b2, c2, d2 = B
    return (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + a2 * b1 + c1 * d2 - d1 * c2,
            a1 * c2 + a2 * c1 + d1 * b2 - b1 * d2,
            a1 * d2 + a2 * d1 + b1 * c2 - c1 * b2)


def infidelity_poly(segs, v):
    """|<0|U|0>|^2 = a^2 + d^2 for the pulse, in the base-step variables v = (c, p, q)."""
    c, p, q = v
    Q = None
    for m, ph in segs:
        V = (c, p * np.cos(ph), p * np.sin(ph), q)
        for _ in range(m):
            Q = V if Q is None else qmul(V, Q)                # the later step acts from the left
    return Q[0] * Q[0] + Q[3] * Q[3]


def physical(base, segs, rho, chi):
    """|<0|U|0>|^2 from H with eps and Delta, by 2x2 exponentials; rho and chi may be arrays."""
    rho, chi = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(chi, dtype=float))
    eps, delta = (1 + rho) * np.cos(chi) - 1, (1 + rho) * np.sin(chi)
    U = np.broadcast_to(I2, rho.shape + (2, 2)).copy()
    for m, ph in segs:
        h = np.stack([(1 + eps) * np.cos(ph), (1 + eps) * np.sin(ph), delta], -1)       # H = h.sigma/2
        w = np.linalg.norm(h, axis=-1)[..., None, None]
        hs = (h[..., 0, None, None] * SX + h[..., 1, None, None] * SY + h[..., 2, None, None] * SZ) / w
        t = m * base                                                                    # Omega = 1
        U = np.einsum("...ij,...jk->...ik", np.cos(w * t / 2) * I2 - 1j * np.sin(w * t / 2) * hs, U)
    return np.abs(U[..., 0, 0]) ** 2


def certify(base, segs, chi_m, order=None):
    """Certified max of the infidelity over the box, as minus a lower bound on its negative."""
    v = psp.polyvar(3)
    c, p, q = v
    f = infidelity_poly(segs, v)
    D = sum(m for m, _ in segs)
    ineqs = [c - np.cos(base * (1 + RHO) / 2), np.cos(base * (1 - RHO) / 2) - c, p, np.tan(chi_m) ** 2 * p * p - q * q]
    t = time.perf_counter()
    res = psp.sos_lower_bound(-f, ineqs=ineqs, eqs=[c * c + p * p + q * q - 1], order=order or max(D, 2), **OPTS)
    return res, time.perf_counter() - t


def worst(base, segs, chi_m, n=121):
    """The largest infidelity over the box: a grid, then local search from its best point."""
    R, C = np.meshgrid(np.linspace(-RHO, RHO, n), np.linspace(-chi_m, chi_m, n))
    G = physical(base, segs, R, C)
    i = np.argmax(G)
    r = minimize(lambda x: -physical(base, segs, x[0], x[1]), [R.flat[i], C.flat[i]],
                 bounds=[(-RHO, RHO), (-chi_m, chi_m)], method="L-BFGS-B")
    return max(G.max(), -r.fun), (r.x if -r.fun >= G.max() else (R.flat[i], C.flat[i]))


def compute():
    out = {"cert": {}, "worst": {}}
    for name, base, segs in PULSES:
        for chi_m in CHIS:
            out["cert"][name, chi_m] = certify(base, segs, chi_m)
            out["worst"][name, chi_m] = worst(base, segs, chi_m)
    out["shor"] = certify(np.pi, PULSES[0][2], 0.1, order=1)[0]          # the plain pulse at the order 1
    out["thin"] = certify(np.pi, PULSES[0][2], 0.005)[0], worst(np.pi, PULSES[0][2], 0.005)[0]
    return out


def checks(out):
    rng = np.random.default_rng(0)
    for name, base, segs in PULSES:
        D = sum(m for m, _ in segs)
        assert physical(base, segs, 0.0, 0.0) < 1e-24
        for rho, chi in rng.uniform(-0.2, 0.2, (20, 2)):            # the polynomial is the physics
            psi = base * (1 + rho) / 2
            v = (np.cos(psi), np.sin(psi) * np.cos(chi), np.sin(psi) * np.sin(chi))
            assert abs(infidelity_poly(segs, v) - physical(base, segs, rho, chi)) < 1e-12
        for e, d in ((0.07, 0.0), (0.0, 0.06), (-0.05, 0.04)):      # (rho, chi) from (eps, Delta), checked by expm
            rho, chi = np.hypot(1 + e, d) - 1, np.arctan2(d, 1 + e)
            U = I2
            for m, ph in segs:
                U = expm(-0.5j * (d * SZ + (1 + e) * (np.cos(ph) * SX + np.sin(ph) * SY)) * m * base) @ U
            assert abs(abs(U[0, 0]) ** 2 - physical(base, segs, rho, chi)) < 1e-12
        for chi_m in CHIS:
            res, _ = out["cert"][name, chi_m]
            w, x = out["worst"][name, chi_m]
            assert res.status == "optimal", (name, chi_m, res.status)
            assert abs(-res.value - w) < 1e-8, (name, chi_m, -res.value, w)
            assert abs(abs(x[0]) - RHO) < 1e-9 and abs(abs(x[1]) - chi_m) < 1e-9       # at a corner
            assert max(res.block_sizes) == {1: 10, 4: 35, 5: 56}[D]
    W = lambda name, chi_m: out["worst"][name, chi_m][0]
    assert abs(out["shor"].value + 1) < 1e-6 and out["shor"].status == "optimal"        # order 1: only the trivial bound
    res, w = out["thin"]                                            # a thinner box defeats SCS: no bound at all
    assert res.status == "optimal_inaccurate" and -res.value < w - 5e-3
    for name, v in (("plain", 0.0342), ("Levitt", 0.0130), ("BB1", 0.0131)):
        assert abs(W(name, 0.1) - v) < 5e-5, (name, W(name, 0.1))
    assert all(1 < W("BB1", x) / np.sin(x) ** 2 < 1.35 for x in CHIS if x >= 0.02)
    assert all(1 < W("Levitt", x) / np.sin(x) ** 2 < 1.5 for x in CHIS if x >= 0.05)
    for name, base, segs in PULSES:                                 # along chi = 0 alone
        along = max(physical(base, segs, r, 0.0) for r in np.linspace(-RHO, RHO, 2001))
        assert abs(along - {"plain": 0.0245, "Levitt": 6.0e-4, "BB1": 9.2e-6}[name]) < {"plain": 5e-5, "Levitt": 5e-6, "BB1": 5e-8}[name]
    assert abs(W("BB1", 0.01) - 1.57e-4) < 5e-7 and abs(W("plain", 0.01) - 0.0246) < 5e-5


def colour(v, lo=-6, hi=-1, levels=25):
    """A colour for log10 of the infidelity, in 25 steps: white, then blue, then red."""
    x = np.round(np.clip((v - lo) / (hi - lo), 0, 1) * levels) / levels
    if x <= 0.5:
        return "ndBlue!%d" % int(round(170 * x))
    return "ndRed!%d!ndBlue!85" % int(round(100 * (2 * x - 1)))


def star(x, y, r=0.11):
    pts = [(x + (r if k % 2 == 0 else 0.42 * r) * np.sin(k * np.pi / 5), y + (r if k % 2 == 0 else 0.42 * r) * np.cos(k * np.pi / 5))
           for k in range(10)]
    return "\\filldraw[fill=ndGold, draw=ndInk, line width=0.4pt] %s -- cycle;" % " -- ".join(N.pt(p) for p in pts)


def pulse_row(x0, y, name, base, segs):
    """(a) one pulse, each segment of width proportional to its angle."""
    b = ["\\node[font=\\small, anchor=east] at (%.2f,%.2f) {%s};" % (x0 - 0.1, y + 0.22, name)]
    x = x0
    labels = {0.0: "x", np.pi / 2: "y", PHI1: "\\phi", 3 * PHI1: "3\\phi"}
    for m, ph in segs:
        w = 0.5 * m * base / (np.pi / 2)
        b.append("\\draw[ink, fill=%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % ("ndSky" if ph == 0 else "ndGold!45", x, y, x + w, y + 0.44))
        ang = {np.pi / 2: "90", np.pi: "180", 2 * np.pi: "360"}[m * base]
        b.append("\\node[font=\\tiny] at (%.3f,%.3f) {$%s_{%s}$};" % (x + w / 2, y + 0.22, ang, labels[ph]))
        x += w
    return b


def main():
    out = compute()
    checks(out)
    b = []
    # ---- (a) the pulses
    y0 = 6.3
    for j, (name, base, segs) in enumerate(PULSES):
        b += pulse_row(1.15, y0 - 0.7 * j, name, base, segs)
    b.append("\\node[note, anchor=north west] at (0.4,%.2f) {angles in degrees, axes\\\\ as subscripts, $\\phi=\\arccos(-\\tfrac14)$};" % (y0 - 1.65))
    b.append("\\node[font=\\small, anchor=north] at (2.8,3.75) {(a) three pulses};")

    # ---- (b) the maps
    n = 61
    rr = np.linspace(-RHO, RHO, n)
    cc = np.linspace(-0.1, 0.1, n)
    R, C = np.meshgrid(rr, cc)
    for j, (name, base, segs) in enumerate(PULSES):
        pm = N.Plot((-RHO, RHO), (-0.1, 0.1), 2.6, 2.6, origin=(6.6 + 3.3 * j, 4.4))
        G = np.log10(np.maximum(physical(base, segs, R, C), 1e-12))
        d = 2.6 / (n - 1)
        for a in range(n):                                  # runs of one colour along each row
            cols = [colour(g) for g in G[a]]
            start = 0
            for c_ in range(1, n + 1):
                if c_ == n or cols[c_] != cols[start]:
                    x0, x1 = pm.P(rr[start], 0)[0] - d / 2, pm.P(rr[c_ - 1], 0)[0] + d / 2
                    y = pm.P(0, cc[a])[1]
                    b.append("\\fill[%s] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                             % (cols[start], max(x0, pm.o[0]), max(y - d / 2, pm.o[1]), min(x1, pm.o[0] + 2.6), min(y + d / 2, pm.o[1] + 2.6)))
                    start = c_
        b.append("\\draw[ink] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (pm.o[0], pm.o[1], pm.o[0] + 2.6, pm.o[1] + 2.6))
        b += contour(pm, rr, cc, G, -3)
        w, x = out["worst"][name, 0.1]
        q = pm.P(x[0], x[1])
        b.append(star(q[0], q[1]))
        b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,%.3f) {%s: worst $%s$};" % (pm.o[0] + 1.3, pm.o[1] + 2.62, name, tex(w)))
        for v, lab in ((-RHO, "$-0.1$"), (0, "$0$"), (RHO, "$0.1$")):
            p0 = pm.P(v, -0.1)
            b.append("\\node[font=\\tiny, anchor=north] at (%.3f,%.3f) {%s};" % (p0[0], p0[1] - 0.02, lab))
        b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$\\rho$};" % (pm.o[0] + 1.3, pm.o[1] - 0.25))
        if j == 0:
            for v, lab in ((-0.1, "$-0.1$"), (0, "$0$"), (0.1, "$0.1$")):
                p0 = pm.P(-RHO, v)
                b.append("\\node[font=\\tiny, anchor=east] at (%.3f,%.3f) {%s};" % (p0[0] - 0.02, p0[1], lab))
            b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {$\\chi$};" % (pm.o[0] - 0.45, pm.o[1] + 1.3))
    # the colour scale
    for k in range(50):
        v = -6 + 5 * k / 49
        b.append("\\fill[%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (colour(v), 16.4, 4.4 + 2.6 * k / 50, 16.6, 4.4 + 2.6 * (k + 1) / 50))
    b.append("\\draw[ink, line width=0.4pt] (16.4,4.4) rectangle (16.6,7.0);")
    for v in (-6, -5, -4, -3, -2, -1):
        b.append("\\node[font=\\tiny, anchor=west] at (16.62,%.3f) {$10^{%d}$};" % (4.4 + 2.6 * (v + 6) / 5, v))
    b.append("\\node[font=\\small, anchor=north] at (11.2,3.75) {(b) the infidelity over the box $|\\rho|\\le0.1$, $|\\chi|\\le0.1$, and its certified maximum};")

    # ---- (c) the certified worst case against chi_m
    pc = N.Plot((np.log10(0.004), np.log10(0.13)), (-5, -1), 10.0, 2.6, origin=(4.2, -0.4))
    b += pc.axes(xlabel="{\\scriptsize $\\chi_m$}", ylabel="{\\scriptsize certified worst infidelity}")
    b += log_ticks(pc, [0.005, 0.01, 0.02, 0.05, 0.1], ["$0.005$", "$0.01$", "$0.02$", "$0.05$", "$0.1$"], True)
    b += log_ticks(pc, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], ["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"], False)
    xs = np.geomspace(0.004, 0.13, 30)
    for name, base, segs in PULSES:
        col = COLOURS[name]
        ws = [worst(base, segs, x, n=41)[0] for x in xs]
        b.append(pc.curve(np.log10(xs), np.log10(ws), "line width=0.9pt, %s" % col))
        for chi_m in CHIS:
            q = pc.P(np.log10(chi_m), np.log10(-out["cert"][name, chi_m][0].value))
            b.append("\\fill[%s] (%.3f,%.3f) circle (1.6pt);" % (col, q[0], q[1]))
    b.append("\\node[note, anchor=south west, text=ndRed] at %s {plain: the error in the rate};" % pc.pt(np.log10(0.0042), np.log10(0.03)))
    b.append("\\node[note, anchor=south west, text=ndBlue] at %s {BB1};" % pc.pt(np.log10(0.0045), np.log10(8e-5)))
    b.append("\\node[note, anchor=south west, text=ndGold!80!ndInk] at %s {Levitt};" % pc.pt(np.log10(0.0045), np.log10(7e-4)))
    b.append("\\node[note, anchor=north west] at %s {the composite pulses:\\\\ the tilt, $\\approx\\sin^2\\chi_m$};" % pc.pt(np.log10(0.038), np.log10(1.2e-3)))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) what a robust pulse buys depends on $\\chi_m$, how well the frequency is known};"
             % pc.pt(np.log10(0.004) + 0.5 * np.log10(0.13 / 0.004), -5.5))
    N.write("ch12_robust", b)
    return out


def tex(v):
    e = int(np.floor(np.log10(v)))
    return "%.4f" % v if e >= -2 else "%.1f\\cdot10^{%d}" % (v / 10 ** e, e)


def contour(pl, xs, ys, G, level):
    """The contour G = level by marching squares, as short segments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = plt.contour(xs, ys, G, levels=[level])
    out = []
    for path in cs.allsegs[0]:
        if len(path) > 1:
            out.append(N.polyline([pl.P(p[0], p[1]) for p in path], "ndInk, line width=0.6pt, dash pattern=on 2pt off 1.5pt"))
    plt.close("all")
    return out


def log_ticks(pl, values, labels, x=True):
    out = []
    for v, lab in zip(values, labels):
        if x:
            p = pl.P(np.log10(v), pl.ylim[0])
            out.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[below, font=\\scriptsize, text=ndInk] {%s};"
                       % (p[0], p[1] + 0.06, p[0], p[1] - 0.06, lab))
        else:
            p = pl.P(pl.xlim[0], np.log10(v))
            out.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[left, font=\\scriptsize, text=ndInk] {%s};"
                       % (p[0] + 0.06, p[1], p[0] - 0.06, p[1], lab))
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    out = main()
    for name, base, segs in PULSES:
        for chi_m in CHIS:
            res, t = out["cert"][name, chi_m]
            w, x = out["worst"][name, chi_m]
            print("%-6s chi_m=%.3f  certified %.6e  grid %.6e  at (%.3f, %.3f)  %s  block %d  %.1f s"
                  % (name, chi_m, -res.value, w, x[0], x[1], res.status, max(res.block_sizes), t))
    print("plain at order 1: %.4f (%s)" % (-out["shor"].value, out["shor"].status))
    print("total %.0f s; written figures/ch12_robust.tex" % (time.perf_counter() - t0))
