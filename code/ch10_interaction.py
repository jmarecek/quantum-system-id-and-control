"""ch10_interaction.py -- the geometric interaction picture for the driven qubit.

The driven qubit of Chapter 4 with both quadratures of the field,
    H(t) = -(omega/2) sigma_z + (1/2)(u_x sigma_x + u_y sigma_y),   omega = 1,
is to make the gate V = exp(-i pi sigma_x/4) in the time T = 3 with the least
control energy int_0^T (u_x^2 + u_y^2) dt.  The drift A_d = i (omega/2) sigma_z is
1-local and normalises the control directions span{sigma_x, sigma_y}, so the
interaction picture zeta(t) = exp(-t A_d) U(t), the frame turning with the drift,
removes it: zeta obeys dzeta/dt = Ã_c zeta with the field turned back by omega t,
of the same strength, and must reach the rotated target exp(-T A_d) V =
exp(-i omega T sigma_z/2) V.  The least energy is therefore L^2/T, L the
sub-Riemannian distance of Chapter 9 from I to the rotated target, attained by
the geodesic (eq. ch09:kp) traversed at constant speed: here L = 3.862, the
field has the strength L/T = 1.287 and in the rotating frame turns at the rate
c L/T = 0.707 (c = 0.549); in the laboratory frame it turns at 0.707 + omega =
1.707, and the energy is 4.972 in both frames.  A direct minimisation in the
laboratory frame (120 constant steps, endpoint V or -V exactly by an augmented
Lagrangian, 16 random starts) finds 4.973 from 13 starts, above the geodesic
value by the discretisation error alone, and 5.359 from the other 3: the
geodesic to -V, which is the same gate up to a global phase.

(a) The Bloch vector of |0> in the laboratory frame (blue), which the drift
    swings round the z-axis, and in the rotating frame (gold), which follows a
    sub-Riemannian geodesic; they end at V|0> and at exp(-i omega T sigma_z/2) V|0>.
(b) The field (u_x, u_y) in the two frames: the same strength, turned through
    2.12 and 5.12 radians.
(c) The two curves on SU(2), drawn by their rotation vectors: U(t) from I to V
    and zeta(t) from I to the rotated target.  Only zeta is horizontal; U has a
    component along sigma_z, the drift, which costs nothing.
Writes figures/ch10_interaction.tex.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares, minimize
import tikzexport as T_
import needham as N

I2, SX, SY, SZ = T_.pauli()
PS = np.array([SX, SY, SZ])
OMEGA, TT = 1.0, 3.0
V = expm(-0.25j * np.pi * SX)
TARGET = expm(-0.5j * OMEGA * TT * SZ) @ V                 # exp(-T A_d) V


def geo_end(th0, c, L):
    a, b = np.cos(th0), np.sin(th0)
    return expm(0.5j * c * L * SZ) @ expm(-0.5j * L * (a * SX + b * SY + c * SZ))


def geodesic(starts=80, seed=0):
    """Shortest sub-Riemannian geodesic (Chapter 9, eq. ch09:kp) from I to the rotated target."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        x0 = [rng.uniform(0, 2 * np.pi), rng.normal(0, 2), rng.uniform(0.2, 12)]
        res = lambda x: np.concatenate([(geo_end(*x) - TARGET).real.ravel(), (geo_end(*x) - TARGET).imag.ravel()])
        r = least_squares(res, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.linalg.norm(r.fun) < 1e-10 and r.x[2] > 0 and (best is None or r.x[2] < best[2] - 1e-9):
            best = r.x
    return best


def geodesic_minus(starts=80, seed=0):
    """The shortest geodesic to -TARGET, the same gate up to a global phase."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        x0 = [rng.uniform(0, 2 * np.pi), rng.normal(0, 2), rng.uniform(0.2, 12)]
        res = lambda x: np.concatenate([(geo_end(*x) + TARGET).real.ravel(), (geo_end(*x) + TARGET).imag.ravel()])
        r = least_squares(res, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.linalg.norm(r.fun) < 1e-10 and r.x[2] > 0 and (best is None or r.x[2] < best[2] - 1e-9):
            best = r.x
    return best


def fields(th0, c, L, t):
    """The field in the rotating frame and in the laboratory frame at the times t."""
    ang_rot = th0 - c * L * t / TT
    ang_lab = ang_rot - OMEGA * t
    amp = L / TT
    return amp * np.stack([np.cos(ang_rot), np.sin(ang_rot)], -1), amp * np.stack([np.cos(ang_lab), np.sin(ang_lab)], -1)


def lab_step(ux, uy, dt):
    hx, hy, hz = ux / 2, uy / 2, -OMEGA / 2 * np.ones_like(ux)
    n = np.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
    c, s = np.cos(n * dt), np.sin(n * dt) / n
    U = np.empty(ux.shape + (2, 2), complex)
    U[..., 0, 0], U[..., 1, 1] = c - 1j * s * hz, c + 1j * s * hz
    U[..., 0, 1], U[..., 1, 0] = -1j * s * (hx - 1j * hy), -1j * s * (hx + 1j * hy)
    return U


def lab_minimum(Nst=120, starts=16, seed=3):
    """Least energy in the laboratory frame over piecewise-constant fields, endpoint exactly V (up to sign)."""
    dt = TT / Nst
    h = 1e-6

    def obj(x, lam, mu):
        ux, uy = x[:Nst], x[Nst:]
        Us = lab_step(ux, uy, dt)
        dUx = (lab_step(ux + h, uy, dt) - lab_step(ux - h, uy, dt)) / (2 * h)
        dUy = (lab_step(ux, uy + h, dt) - lab_step(ux, uy - h, dt)) / (2 * h)
        pre = [I2]
        for k in range(Nst):
            pre.append(Us[k] @ pre[-1])
        suf = [I2]
        for k in range(Nst - 1, -1, -1):
            suf.append(suf[-1] @ Us[k])
        suf = suf[::-1]
        U = pre[-1]
        r = np.array([np.imag(np.trace(P @ V.conj().T @ U)) / 2 for P in PS])
        B = sum((l + mu * ri) * P @ V.conj().T for l, ri, P in zip(lam, r, PS)) / 2
        f = np.sum(x ** 2) * dt + lam @ r + mu / 2 * r @ r
        g = 2 * x * dt
        for k in range(Nst):
            M = pre[k] @ B @ suf[k + 1]
            g[k] += np.imag(np.trace(M @ dUx[k]))
            g[Nst + k] += np.imag(np.trace(M @ dUy[k]))
        return f, g, U, r

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(starts):
        x, lam, mu = rng.normal(0, 1.5, 2 * Nst), np.zeros(3), 10.0
        for _ in range(30):
            x = minimize(lambda z: obj(z, lam, mu)[:2], x, jac=True, method="L-BFGS-B",
                         options={"maxiter": 5000, "gtol": 1e-10}).x
            _, _, U, r = obj(x, lam, mu)
            lam = lam + mu * r
            if np.linalg.norm(r) < 1e-11:
                break
            mu = min(3 * mu, 1e6)
        err = min(np.linalg.norm(U - V), np.linalg.norm(U + V))
        out.append((np.sum(x ** 2) * dt, err))
    return out


def rotvec(U):
    w = np.clip(np.real(np.trace(U)) / 2, -1, 1)
    v = np.array([-np.imag(np.trace(U @ P)) / 2 for P in PS])
    nv = np.linalg.norm(v)
    return 2 * np.arccos(w) * v / nv if nv > 1e-14 else np.zeros(3)


def trajectories(th0, c, L, n=1201):
    """zeta(t) from the geodesic, and U(t) = exp(t A_d) zeta(t); also U by integrating the lab field."""
    ts = np.linspace(0, TT, n)
    zeta = [geo_end(th0, c, L * t / TT) for t in ts]
    U = [expm(0.5j * OMEGA * t * SZ) @ z for t, z in zip(ts, zeta)]
    _, ulab = fields(th0, c, L, 0.5 * (ts[1:] + ts[:-1]))
    Ul = [I2]
    for k in range(n - 1):
        Ul.append(lab_step(ulab[k:k + 1, 0], ulab[k:k + 1, 1], ts[1] - ts[0])[0] @ Ul[-1])
    return ts, zeta, U, Ul


def checks():
    th0, c, L = geodesic()
    assert abs(L - 3.8622) < 1e-3 and abs(c - 0.5491) < 1e-3 and abs(L ** 2 / TT - 4.9723) < 1e-3
    ts, zeta, U, Ul = trajectories(th0, c, L)
    assert np.allclose(U[-1], V, atol=1e-9) and np.allclose(zeta[-1], TARGET, atol=1e-9)
    assert np.linalg.norm(Ul[-1] - V) < 1e-5                               # the laboratory field makes V
    for k in (100, 600, 1100):                                             # zeta is horizontal: no sigma_z component
        Hz = 1j * (zeta[k + 1] - zeta[k - 1]) @ zeta[k].conj().T / (ts[2] - ts[0])
        assert abs(np.trace(Hz @ SZ)) < 1e-5
    urot, ulab = fields(th0, c, L, ts)
    assert np.allclose(np.linalg.norm(urot, axis=1), np.linalg.norm(ulab, axis=1))   # the cost identity, pointwise
    lab = lab_minimum()
    Es = np.array([e for e, err in lab if err < 1e-6])
    assert len(Es) == len(lab)                                             # every start converged to +V or -V
    Lm = geodesic_minus()[2]                                               # the geodesic to -V, the same gate
    near = lambda e, E: abs(e - E) < 2e-3 * E
    assert all(near(e, L ** 2 / TT) or near(e, Lm ** 2 / TT) for e in Es)
    assert Es.min() >= L ** 2 / TT - 1e-9 and near(Es.min(), L ** 2 / TT)
    return th0, c, L, Es


def main():
    th0, c, L, Es = checks()
    ts, zeta, U, Ul = trajectories(th0, c, L)
    psi0 = np.array([1, 0], complex)
    bloch = lambda W: np.real([np.conj(W @ psi0) @ P @ (W @ psi0) for P in PS])
    rlab = np.array([bloch(W) for W in U])
    rrot = np.array([bloch(W) for W in zeta])
    b = []
    # ---- (a) Bloch sphere
    Vw = N.View(az=-8.0, el=18.0, R=1.75, shift=(2.1, 2.4))
    b += Vw.sphere()
    b += Vw.great_circle([0, 0, 1])
    for e, lab, off in [((1.3, 0, 0), "x", (-0.1, -0.12)), ((0, 1.3, 0), "y", (0.14, -0.02)), ((0, 0, 1.3), "z", (0.0, 0.13))]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (Vw.pt((0, 0, 0)), Vw.pt(e)))
        b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%s$};" % (N.pt(Vw.P(e) + np.array(off)), lab))
    b += Vw.curve(rrot, "line width=1.3pt, ndGold!90!ndInk", "line width=0.6pt, ndGold!60, densely dotted")
    b += Vw.curve(rlab, "line width=1.3pt, ndBlue", "line width=0.6pt, ndBlue!50, densely dotted")
    b.append("\\node[dot] at %s {};" % Vw.pt((0, 0, 1)))
    b.append("\\node[font=\\scriptsize, anchor=south east] at %s {$\\ket0$};" % N.pt(Vw.P((0, 0, 1)) + np.array([-0.04, 0.03])))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % Vw.pt(rlab[-1]))
    b.append("\\node[dot, fill=ndGold!90!ndInk] at %s {};" % Vw.pt(rrot[-1]))
    c0 = Vw.P((0, 0, 0))
    b.append("\\node[note, anchor=east, text=ndBlue] (lb) at %s {laboratory\\\\ frame, to $V\\ket0$};" % N.pt(c0 + np.array([-1.95, 0.9])))
    kl = np.argmin([Vw.P(p)[0] for p in rlab])
    b.append("\\draw[pointer] (lb.east) to[bend left=10] %s;" % N.pt(Vw.P(rlab[kl]) + np.array([-0.06, 0.0])))
    b.append("\\node[note, anchor=west, text=ndGold!55!ndInk] (rf) at %s {rotating frame:\\\\ a geodesic};" % N.pt(c0 + np.array([1.9, -0.45])))
    kk = np.argmax([Vw.P(p)[0] for p in rrot])
    b.append("\\draw[pointer] (rf.west) to[bend right=10] %s;" % N.pt(Vw.P(rrot[kk]) + np.array([0.06, 0.02])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the state in the two frames};" % N.pt(np.array([c0[0], 0.1])))

    # ---- (b) the field in the two frames
    urot, ulab = fields(th0, c, L, ts)
    amp = L / TT
    for x0, uu, col, ttl, rate in [(6.55, urot, "ndGold!90!ndInk", "rotating frame", "%.2f" % (c * L / TT)),
                                   (9.75, ulab, "ndBlue", "laboratory frame", "%.2f+\\omega" % (c * L / TT))]:
        cc = np.array([x0, 2.75])
        R = 0.95
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$u_x$};" % (N.pt(cc + [-1.3, 0]), N.pt(cc + [1.35, 0])))
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$u_y$};" % (N.pt(cc + [0, -1.3]), N.pt(cc + [0, 1.35])))
        b.append("\\draw[faint] %s circle (%.3f);" % (N.pt(cc), R))
        ang = np.unwrap(np.arctan2(uu[:, 1], uu[:, 0]))
        pts = [cc + R * np.array([np.cos(a), np.sin(a)]) for a in ang]
        b.append(N.polyline(pts, "line width=1.4pt, %s" % col))
        tang = np.sign(ang[-1] - ang[0]) * np.array([-np.sin(ang[-1]), np.cos(ang[-1])])
        b.append("\\draw[->, >=stealth, line width=1.4pt, %s] %s -- %s;" % (col, N.pt(pts[-1] - 0.02 * tang), N.pt(pts[-1])))
        b.append("\\draw[vec, ndInk!70, line width=0.7pt] %s -- %s;" % (N.pt(cc), N.pt(pts[0])))
        b.append("\\node[dot, minimum size=2.6pt] at %s {};" % N.pt(pts[0]))
        b.append("\\node[note, anchor=north] at %s {%s:\\\\ strength $%.3f$,\\\\ rate $%s$,\\\\ through $%.2f$};" % (N.pt(cc + [0, -1.3]), ttl, amp, rate, abs(ang[-1] - ang[0])))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the field in the two frames};" % N.pt(np.array([8.15, 0.1])))

    # ---- (c) the two curves on SU(2)
    W3 = N.View(az=30.0, el=20.0, R=0.72, shift=(14.1, 2.2))
    o = W3.P((0, 0, 0))
    for e, lab, off in [((2.4, 0, 0), "x", (-0.1, -0.12)), ((0, 2.4, 0), "y", (0.13, -0.02)), ((0, 0, 2.6), "z", (0.0, 0.12))]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (W3.pt((0, 0, 0)), W3.pt(e)))
        b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%s$};" % (N.pt(W3.P(e) + np.array(off)), lab))
    cz = np.array([rotvec(W) for W in zeta[::6]])
    cu = np.array([rotvec(W) for W in U[::6]])
    b.append(N.polyline([W3.P(p) for p in cz], "line width=1.3pt, ndGold!90!ndInk"))
    b.append(N.polyline([W3.P(p) for p in cu], "line width=1.3pt, ndBlue"))
    b.append("\\node[dot] at %s {};" % W3.pt((0, 0, 0)))
    b.append("\\node[font=\\scriptsize, anchor=north east] at %s {$\\Id$};" % N.pt(o + np.array([-0.03, -0.03])))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % W3.pt(cu[-1]))
    b.append("\\node[font=\\scriptsize, anchor=north, text=ndBlue] at %s {$V$};" % N.pt(W3.P(cu[-1]) + np.array([0.0, -0.06])))
    b.append("\\node[dot, fill=ndGold!90!ndInk] at %s {};" % W3.pt(cz[-1]))
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndGold!55!ndInk] at %s {$\\ee^{-\\imath\\omega T\\sigma_z/2}V$};" % N.pt(W3.P(cz[-1]) + np.array([0.08, 0.0])))
    b.append("\\node[note, anchor=north] at %s {same energy $%.3f$;\\\\ only the gold curve\\\\ is horizontal};" % (N.pt(o + np.array([0.2, -0.75])), L ** 2 / TT))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the two curves on $\\SU[2]$};" % N.pt(np.array([o[0] + 0.2, 0.1])))
    N.write("ch10_interaction", b)
    return th0, c, L, Es


if __name__ == "__main__":
    th0, c, L, Es = main()
    print("geodesic: L = %.4f, c = %.4f, energy %.4f; laboratory minima:" % (L, c, L ** 2 / TT), np.round(np.sort(Es), 4))
    print("written figures/ch10_interaction.tex")
