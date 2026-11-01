"""ch09_timeoptimal.py -- time-optimal transfer of the driven qubit.

The driven qubit of Chapter 4, H = -(omega/2) sigma_z + (u/2) sigma_x, is
H = (1/2)(Delta sigma_z + u sigma_x) with Delta = -omega.  We take omega = 1
and the bound |u| <= u0 = 1.5, the constant pulse of Figure 7 (ch07_speed.py),
and transfer |0> to |1> (up to a phase) in the least time.  Write
Omega = sqrt(u0^2 + Delta^2) and k = |Delta|/u0.

The maximum principle gives u = u0 sgn H_x with H_x = Im<chi|sigma_x|psi>/2
(no singular arcs), and the first and last bangs are not longer than the
interior ones, which all have one duration t_b.  For 0 < k <= 1 two bangs
suffice and the least time is 2 pi/Omega, with the bangs
    t1 = (pi - arccos k^2)/Omega,   t2 = (pi + arccos k^2)/Omega
in either order and either sign of the first (four solutions), switching on
the equator (Boscain and Mason 2006; Ansel et al. 2024).

(a) The Bloch sphere: two of the four optimal paths (blue), their switching
    points on the equator (red), the meridian from |0> to |1> along which the
    speed limit of Chapter 7 would be met, which the dynamics cannot follow
    (gold), and the constant pulse u = u0, which never reaches |1> (green).
(b) The control u(t) and the switching function H_x(t) of the first path,
    computed by integrating |psi> and the adjoint |chi> under the same
    propagators.  At the switch H = (H_x, H_y, H_z) = (0, 0, 1/Delta): H_P = 0
    fixes H_z, and H must be orthogonal to the Bloch vector, which has a
    non-zero y-component there.  Hence H_x(t) = (u/Omega^2)(1 - cos Omega(t - t1))
    on each bang: it changes sign at t1 with zero slope.
(c) The least time u0 t* against k.  For k = 0 a single bang does it in pi/u0
    (the pi-pulse); for 0 < k <= 1 it is 2 pi/Omega, so that t* jumps at k = 0;
    for k > 1 it is computed by searching the extremals of the maximum
    principle, sequences t1, t_b, ..., t_b, t2 with t1, t2 <= t_b, whose
    endpoint is found in closed form as an intersection of two circles on the
    sphere.  The speed limit pi/Omega of Chapter 7 (dashed) is met only at
    k = 0.  The minima were confirmed independently by bounded gradient ascent
    on 120 piecewise-constant steps (scratch computation): just below t* the
    best transfer probability is < 1, just above it is 1.
Writes figures/ch09_timeoptimal.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N

I2, SX, SY, SZ = T.pauli()
OMEGA_Q, U0 = 1.0, 1.5                       # the driven qubit of Chapter 4 and the bound of Figure 7
DELTA = -OMEGA_Q
W = np.hypot(U0, DELTA)
K = abs(DELTA) / U0
T1 = (np.pi - np.arccos(K ** 2)) / W
T2 = (np.pi + np.arccos(K ** 2)) / W
TSTAR = 2 * np.pi / W


def rot(n, th):
    """Rotation of Bloch vectors by th about the unit vector n (right-handed): r' = h x r."""
    n = np.asarray(n, dtype=float)
    Kx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def bloch(psi):
    return np.real([psi.conj() @ P @ psi for P in (SX, SY, SZ)])


def bang_path(s0, durations, delta=DELTA, u0=U0, n=200):
    """Bloch vectors along a bang-bang control starting with u = s0 u0, from |0>."""
    w = np.hypot(u0, delta)
    r, s, pts = np.array([0, 0, 1.0]), s0, []
    for d in durations:
        ax = np.array([s * u0, 0, delta]) / w
        for th in np.linspace(0, w * d, n):
            pts.append(rot(ax, th) @ r)
        r, s = rot(ax, w * d) @ r, -s
    return np.array(pts), r


def switching(s0, ts):
    """Integrate psi and chi under the bang-bang control (t1 then t2, first sign s0); return u, H_x, H_P."""
    u_of = lambda t: s0 * U0 if t < T1 else -s0 * U0
    Hm = lambda u: 0.5 * (DELTA * SZ + u * SX)
    psi1 = expm(-1j * Hm(s0 * U0) * T1) @ np.array([1, 0], dtype=complex)
    chi1 = -2j * (1 / DELTA) * SZ @ psi1               # (H_x, H_y, H_z)(t1) = (0, 0, 1/Delta)
    us, Hx, HP, Hn = [], [], [], []
    for t in ts:
        if t < T1:
            V = expm(-1j * Hm(s0 * U0) * (t - T1))
        else:
            V = expm(-1j * Hm(-s0 * U0) * (t - T1))
        psi, chi = V @ psi1, V @ chi1
        H = [0.5 * np.imag(chi.conj() @ P @ psi) for P in (SX, SY, SZ)]
        u = u_of(t)
        us.append(u), Hx.append(H[0]), HP.append(DELTA * H[2] + u * H[0] - 1), Hn.append(np.linalg.norm(H))
    return np.array(us), np.array(Hx), np.array(HP), np.array(Hn)


def _rb(n, th):
    Kx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * Kx[None] + (1 - c) * (Kx @ Kx)[None]


def _angle(a, p, q):
    pp, qq = p - (p @ a)[:, None] * a, q - (q @ a)[:, None] * a
    return np.arctan2(np.cross(pp, qq) @ a, np.sum(pp * qq, 1)) % (2 * np.pi)


def least_time(k, u0=1.0, mmax=10, ngrid=6000):
    """Least time from |0> to |1> over the extremals t1, t_b (m times), t2 with t1, t2 <= t_b.

    For each m and t_b the first bang ends on the circle {r : a.r = a.z} about its axis a,
    and the last bang starts on the circle {r : (M^T n).r = -n.z}, M the interior bangs and
    n the last axis; the switching points are the intersections of the two circles."""
    delta = k * u0
    w = np.hypot(delta, u0)
    P = 2 * np.pi / w
    z = np.array([0, 0, 1.0])
    best = (np.inf, None)
    for s0 in (1, -1):
        axes = [np.array([s0 * (-1) ** j * u0, 0, delta]) / w for j in range(mmax + 2)]
        for m in range(mmax + 1):
            a, nl = axes[0], axes[m + 1]
            tb = np.linspace(P / ngrid, P, ngrid) if m > 0 else np.array([P])
            M = np.broadcast_to(np.eye(3), (len(tb), 3, 3)).copy()
            for j in range(1, m + 1):
                M = _rb(axes[j], w * tb) @ M
            bv = np.einsum('nji,j->ni', M, nl)
            al, be, ab = a @ z, -(nl @ z), bv @ a
            det = 1 - ab ** 2
            good = det > 1e-12
            det = np.where(good, det, 1.0)
            r0 = ((al - ab * be) / det)[:, None] * a + ((be - ab * al) / det)[:, None] * bv
            wv = np.cross(np.broadcast_to(a, bv.shape), bv)
            h2 = 1 - np.sum(r0 * r0, 1)
            good &= h2 >= 0
            h = np.sqrt(np.clip(h2, 0, None) / np.maximum(np.sum(wv * wv, 1), 1e-300))
            for sg in (1, -1):
                r = r0 + sg * h[:, None] * wv
                t1 = _angle(a, np.broadcast_to(z, r.shape), r) / w
                t2 = _angle(nl, np.einsum('nij,nj->ni', M, r), np.broadcast_to(-z, r.shape)) / w
                Tt = t1 + m * tb + t2
                ok = good & (t1 <= tb + 1e-9) & (t2 <= tb + 1e-9)
                if ok.any():
                    i = np.argmin(np.where(ok, Tt, np.inf))
                    if Tt[i] < best[0]:
                        best = (Tt[i], (s0, m + 2, t1[i], tb[i], t2[i]))
    return best


def checks():
    # the four two-bang controls reach |1>, switching on the equator; a single bang does not
    for s0 in (1, -1):
        for d in ((T1, T2), (T2, T1)):
            path, end = bang_path(s0, d)
            assert np.allclose(end, [0, 0, -1], atol=1e-9)
            r1 = bang_path(s0, d[:1])[1]
            assert abs(r1[2]) < 1e-9 and abs(r1[1]) > 0.5
    one = max(-bang_path(1, (t,))[1][2] for t in np.linspace(0, 2 * np.pi / W, 2001))
    assert one < 1 - 1e-3 and abs((1 - one) / 2 - (1 - U0 ** 2 / W ** 2)) < 1e-6   # best single bang: u0^2/Omega^2
    # the closed form of the transfer probability of two bangs (Ansel et al.)
    for a, b in [(0.7, 1.9), (T1, T2), (1.3, 0.4)]:
        U = expm(-0.5j * (DELTA * SZ - U0 * SX) * b) @ expm(-0.5j * (DELTA * SZ + U0 * SX) * a)
        pr = U0 ** 2 / W ** 4 * (4 * DELTA ** 2 * np.sin(W * a / 2) ** 2 * np.sin(W * b / 2) ** 2 + W ** 2 * np.sin(W * (a - b) / 2) ** 2)
        assert abs(abs(U[1, 0]) ** 2 - pr) < 1e-12
    # the switching function: sign of u, zero only at t1, H_P = 0 and |H| conserved, closed form
    ts = np.linspace(0, TSTAR, 701)
    u, Hx, HP, Hn = switching(-1, ts)
    assert np.all(np.abs(HP) < 1e-9) and np.ptp(Hn) < 1e-9
    assert np.all(np.sign(Hx[np.abs(ts - T1) > 1e-3]) == np.sign(u[np.abs(ts - T1) > 1e-3]))
    assert np.allclose(Hx, u / W ** 2 * (1 - np.cos(W * (ts - T1))), atol=1e-9)
    # the least times: 2 pi/Omega for k <= 1 (the search agrees), more bangs beyond
    for k in (0.3, 2 / 3, 1.0):
        Tk, info = least_time(k)
        assert abs(Tk - 2 * np.pi / np.hypot(k, 1)) < 1e-6 and info[1] == 2
    for k, nb in ((1.5, 3), (2.0, 4), (3.0, 5)):
        assert least_time(k)[1][1] == nb
    assert abs(least_time(1.1)[0] - 5.031) < 2e-3                    # confirmed by gradient ascent


def main():
    checks()
    b = []
    # ---- (a) the Bloch sphere, seen from the side of +x where all four optimal paths lie
    V = N.View(az=12.0, el=10.0, R=1.75, shift=(2.1, 2.35))
    b += V.sphere()
    b += V.great_circle([0, 0, 1], front="faint", back="faint, densely dotted")
    for e, lab, off in [((1.3, 0, 0), "x", (-0.12, -0.1)), ((0, 1.3, 0), "y", (0.14, 0.0)), ((0, 0, 1.3), "z", (0.0, 0.14))]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (V.pt((0, 0, 0)), V.pt(e)))
        b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%s$};" % (N.pt(V.P(e) + np.array(off)), lab))
    mer = np.array([[np.sin(t) * np.cos(1.22), np.sin(t) * np.sin(1.22), np.cos(t)] for t in np.linspace(0, np.pi, 121)])
    b += V.curve(mer, "line width=1.0pt, ndGold, dash pattern=on 3pt off 2pt", "line width=0.6pt, ndGold!60, densely dotted")
    const = bang_path(-1, (2 * np.pi / W,), n=241)[0]                 # the constant pulse u = -u0: a full circle
    b += V.curve(const, "line width=0.9pt, ndGreen", "line width=0.6pt, ndGreen!60, densely dotted")
    for d, sty in [((T1, T2), "line width=1.5pt, ndBlue")]:
        path = bang_path(-1, d)[0]
        b += V.curve(path, sty, "line width=0.6pt, ndBlue!50, densely dotted")
        b.append("\\node[dot, fill=ndRed] at %s {};" % V.pt(bang_path(-1, d[:1])[1]))
    b.append("\\node[dot] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[dot] at %s {};" % V.pt((0, 0, -1)))
    b.append("\\node[font=\\scriptsize, anchor=south east] at %s {$\\ket0$};" % N.pt(V.P((0, 0, 1)) + np.array([-0.05, 0.03])))
    b.append("\\node[font=\\scriptsize, anchor=north east] at %s {$\\ket1$};" % N.pt(V.P((0, 0, -1)) + np.array([-0.05, -0.03])))
    c = V.P((0, 0, 0))
    sw = bang_path(-1, (T1,))[1]
    b.append("\\node[note, anchor=west, text=ndRed] (sw) at %s {switch on\\\\ the equator};" % N.pt(c + np.array([2.0, 0.55])))
    b.append("\\draw[pointer] (sw.west) to[bend left=10] %s;" % N.pt(V.P(sw) + np.array([0.08, -0.02])))
    b.append("\\node[note, anchor=west, text=ndGreen] (cp) at %s {the constant pulse\\\\ $u=-u_0$ stays\\\\ on this circle};" % N.pt(c + np.array([2.0, -0.85])))
    vis = [q for q in const if q @ V.d > 0.05 and q[2] < -0.12]
    p_c = max(vis, key=lambda q: V.P(q)[0])
    b.append("\\draw[pointer] (cp.west) to[bend right=10] %s;" % N.pt(V.P(p_c) + np.array([0.06, 0.03])))
    b.append("\\node[note, anchor=west, text=ndGold!60!ndInk] (ge) at %s {a geodesic,\\\\ not a trajectory};" % N.pt(c + np.array([2.0, 1.55])))
    pg = mer[30]
    b.append("\\draw[pointer] (ge.west) to[bend right=10] %s;" % N.pt(V.P(pg) + np.array([0.07, 0.02])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the time-optimal paths};" % N.pt(c + np.array([0.0, -2.05])))

    # ---- (b) the control and the switching function
    ts = np.linspace(0, TSTAR, 701)
    u, Hx, HP, Hn = switching(-1, ts)
    sc = 1 / np.abs(Hx).max()
    pl = N.Plot((0, TSTAR * 1.04), (-1.25, 1.25), 4.6, 3.2, origin=(7.25, 0.75))
    b += pl.axes(xlabel="{\\scriptsize $t$}", xticks=[(T1, "$t_1$"), (TSTAR, "$t^\\star$")],
                 yticks=[(-1, "$-u_0$"), (1, "$u_0$")], yzero=0)
    b.append("\\fill[ndBlue!10] %s rectangle %s;" % (pl.pt(0, 0), pl.pt(T1, -1)))
    b.append("\\fill[ndBlue!10] %s rectangle %s;" % (pl.pt(T1, 0), pl.pt(TSTAR, 1)))
    b.append(pl.curve([0, T1, T1, TSTAR], [-1, -1, 1, 1], "line width=1.3pt, ndBlue"))
    b.append(pl.curve(ts, Hx * sc, "line width=1.0pt, ndRed"))
    b.append("\\draw[dashedink] %s -- %s;" % (pl.pt(T1, -1.2), pl.pt(T1, 1.2)))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=south] at %s {$u(t)$};" % pl.pt(2.3, 1.02))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at %s {$H_x(t)$};" % pl.pt(2.95, 0.55))
    b.append("\\node[note, anchor=north] (hx) at %s {$u=u_0\\operatorname{sgn}H_x$;\\\\ $H_x$ changes sign\\\\ with zero slope};" % pl.pt(2.45, -0.22))
    b.append("\\draw[pointer] (hx.west) to[bend left=20] %s;" % pl.pt(T1 + 0.06, -0.05))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) control and switching function};" % pl.pt(TSTAR / 2, -1.55))

    # ---- (c) the least time against the detuning
    ks = np.concatenate([np.linspace(0.004, 1.0, 120), np.linspace(1.0005, 1.05, 30), np.linspace(1.05, 3.0, 200)])
    res = [least_time(k) for k in ks]
    Ts = np.array([r[0] for r in res])
    nb = np.array([r[1][1] for r in res])
    pc = N.Plot((0, 3.0), (0, 7.0), 4.4, 3.2, origin=(13.45, 0.75))
    b += pc.axes(xlabel="{\\scriptsize $\\abs{\\Delta}/u_0$}", ylabel="{\\scriptsize $u_0t^\\star$}",
                 xticks=[(1, "$1$"), (2, "$2$"), (3, "$3$")], yticks=[(np.pi, "$\\pi$"), (2 * np.pi, "$2\\pi$")])
    kk = np.linspace(0, 3, 200)
    b.append(pc.curve(kk, np.pi / np.hypot(kk, 1), "line width=0.8pt, ndGold, dash pattern=on 3pt off 2pt"))
    # pieces of constant number of bangs
    for n in sorted(set(nb)):
        idx = np.where(nb == n)[0]
        for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            b.append(pc.curve(ks[seg], Ts[seg], "line width=1.2pt, ndBlue"))
        mid = idx[len(idx) // 2]
        b.append("\\node[font=\\tiny, text=ndBlue, anchor=south] at %s {%d};" % (pc.pt(ks[mid], Ts[mid] + 0.12), n))
    b.append("\\node[dot, fill=ndBlue, minimum size=3pt] at %s {};" % pc.pt(0, np.pi))
    b.append("\\draw[ndBlue, fill=white, line width=0.7pt] %s circle (1.5pt);" % pc.pt(0, 2 * np.pi))
    b.append("\\node[dot, fill=ndRed, minimum size=3.4pt] at %s {};" % pc.pt(K, U0 * TSTAR))
    b.append("\\node[note, anchor=north, text=ndRed] (ex) at %s {our qubit,\\\\ $u_0t^\\star=5.23$};" % pc.pt(1.75, 7.1))
    b.append("\\draw[pointer] (ex.west) to[bend right=15] %s;" % pc.pt(K + 0.05, U0 * TSTAR + 0.12))
    b.append("\\node[note, anchor=west, text=ndGold!60!ndInk] at %s {speed limit $\\pi u_0/\\Omega$};" % pc.pt(1.25, 0.5))
    b.append("\\node[note, anchor=west] at %s {$\\pi$-pulse};" % pc.pt(0.08, np.pi + 0.4))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the least time};" % pc.pt(1.5, -0.85))
    N.write("ch09_timeoptimal", b)
    return Ts, nb, ks


if __name__ == "__main__":
    Ts, nb, ks = main()
    print("t1 = %.4f, t2 = %.4f, t* = %.4f, u0 t* = %.4f, speed limit %.4f" % (T1, T2, TSTAR, U0 * TSTAR, np.pi / W))
    for n in sorted(set(nb)):
        print("%d bangs for k in [%.3f, %.3f]" % (n, ks[nb == n].min(), ks[nb == n].max()))
    print("written figures/ch09_timeoptimal.tex")
