"""ch09_subriemann.py -- sub-Riemannian geodesics of SU(2).

The qubit with no drift and a field in the x-y plane only,
    H(t) = (1/2)(u_x(t) sigma_x + u_y(t) sigma_y),   u_x^2 + u_y^2 <= 1,
is to make the gate V = exp(-i phi sigma_z/2), a rotation of the Bloch sphere
about the axis that no control turns about.  The directions available at U
are {-i(a sigma_x + b sigma_y)/2 . U}, a field of planes on SU(2) whose
brackets give sigma_z (Chow-Rashevskii), so every V is reachable, and the
least time is the sub-Riemannian distance from I to V.  With 4 = k + p,
k = span{sigma_z}, p = span{sigma_x, sigma_y}, the normal geodesics are
    U(t) = exp(i c t sigma_z/2) exp(-i t (a sigma_x + b sigma_y + c sigma_z)/2),   a^2 + b^2 = 1,
whose velocity U' U^{-1} lies in p and has unit length: the field has
constant strength and turns in the x-y plane at the constant rate c.  The
second factor returns to -I at T = 2 pi/sqrt(1 + c^2), and then U(T) = V with
phi = 2 pi (1 - c/sqrt(1 + c^2)), so that for 0 <= phi <= 2 pi
    T(phi) = sqrt(phi (4 pi - phi)),
against T = phi if sigma_z were available with the same strength (Boscain and
Rossi 2008).  For small phi, T ~ 2 sqrt(pi phi).  Gradient search over
unit-strength piecewise-constant fields (60 steps, 20 starts; scratch
computation) confirms the least times at phi = 0.4, pi/2 and pi: the best
fidelity Re Tr(V^dag U)/2 is below 1 at T(phi) - 0.06 and equal to 1 at
T(phi) + 0.05.

(a) The geodesics from I to V for phi = pi/2, drawn by their rotation vectors
    theta n (U = exp(-i theta n.sigma/2)).  The one-parameter group exp(-i t
    sigma_z/2), the Riemannian geodesic, runs straight up the z-axis (gold),
    but its direction is not available.  The sub-Riemannian geodesics leave
    I horizontally, in the plane of the available directions, and every
    rotation of one of them about the z-axis is another (blue), so that a
    whole circle of geodesics arrives at V: V is a conjugate point.  The small
    patches are the planes of available directions at points of one geodesic,
    to which it is tangent.
(b) The least time T(phi), against phi, with and without sigma_z.
(c) The field (u_x, u_y) of the geodesic drawn bold in (a): it keeps unit
    strength and turns through 2 pi - phi at the rate c = 1.134.
Writes figures/ch09_subriemann.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N

I2, SX, SY, SZ = T.pauli()
PHI = np.pi / 2


def geodesic(phi, theta0=0.0, n=241):
    s = 1 - phi / (2 * np.pi)
    c = s / np.sqrt(1 - s * s)
    Tt = 2 * np.pi / np.sqrt(1 + c * c)
    a, b = np.cos(theta0), np.sin(theta0)
    ts = np.linspace(0, Tt, n)
    Us = [expm(0.5j * c * t * SZ) @ expm(-0.5j * t * (a * SX + b * SY + c * SZ)) for t in ts]
    return ts, Us, Tt, c


def rotvec(U):
    """theta n with U = exp(-i theta n.sigma/2), theta in [0, 2 pi]."""
    w = np.real(np.trace(U)) / 2
    v = np.array([-np.imag(np.trace(U @ P)) / 2 for P in (SX, SY, SZ)])
    nv = np.linalg.norm(v)
    return 2 * np.arctan2(nv, w) * v / nv if nv > 1e-14 else np.zeros(3)


def field(U0, U1, dt):
    """(u_x, u_y, u_z) of the Hamiltonian between two nearby points: U1 U0^dag = exp(-i dt H)."""
    G = 1j * (U1 @ U0.conj().T - U0 @ U1.conj().T) / (2 * dt)        # ~ H, to second order
    return np.array([np.real(np.trace(G @ P)) for P in (SX, SY, SZ)])


def least_time(phi):
    return np.sqrt(phi * (4 * np.pi - phi))


def checks():
    for phi in (0.4, PHI, np.pi, 1.5 * np.pi):
        for th0 in (0.0, 1.0, 2.5):
            ts, Us, Tt, c = geodesic(phi, th0, n=2001)
            V = expm(-0.5j * phi * SZ)
            assert np.allclose(Us[-1], V, atol=1e-10) and abs(Tt - least_time(phi)) < 1e-12
            for k in range(0, 2000, 97):                              # horizontal, unit strength, turning at rate c
                u = field(Us[k], Us[k + 1], ts[1])
                assert abs(u[2]) < 1e-5 and abs(np.hypot(u[0], u[1]) - 1) < 1e-5
            u0 = field(Us[0], Us[1], ts[1])
            u1 = field(Us[1000], Us[1001], ts[1])
            ang = np.arctan2(u1[1], u1[0]) - np.arctan2(u0[1], u0[0])
            assert abs((ang - (-c) * ts[1000] + np.pi) % (2 * np.pi) - np.pi) < 1e-3 or abs((ang - c * ts[1000] + np.pi) % (2 * np.pi) - np.pi) < 1e-3
    # small rotations cost time ~ 2 sqrt(pi phi), large compared with phi
    for phi in (1e-4, 1e-3):
        assert abs(least_time(phi) / (2 * np.sqrt(np.pi * phi)) - 1) < 1e-3
    ts, Us, Tt, c = geodesic(PHI)
    assert abs(c - 1.134) < 1e-3 and max(np.linalg.norm(rotvec(U)) for U in Us) < np.pi


def main():
    checks()
    b = []
    # ---- (a) the geodesics in rotation-vector coordinates
    V = N.View(az=25.0, el=22.0, R=1.55, shift=(2.6, 1.75))
    o = V.P((0, 0, 0))
    for e, lab, off in [((1.55, 0, 0), "x", (-0.1, -0.12)), ((0, 1.55, 0), "y", (0.12, -0.03)), ((0, 0, 2.15), "z", (0.0, 0.12))]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (V.pt((0, 0, 0)), V.pt(e)))
        b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%s$};" % (N.pt(V.P(e) + np.array(off)), lab))
    sq = [(-0.55, -0.55, 0), (0.55, -0.55, 0), (0.55, 0.55, 0), (-0.55, 0.55, 0)]
    b.append("\\fill[ndSky, fill opacity=0.8] %s -- cycle;" % " -- ".join(V.pt(p) for p in sq))
    b.append("\\draw[line width=1.1pt, ndGold, dash pattern=on 3pt off 2pt] %s -- %s;" % (V.pt((0, 0, 0)), V.pt((0, 0, PHI))))
    curves = []
    for k in range(6):
        ts, Us, Tt, c = geodesic(PHI, 2 * np.pi * k / 6)
        curves.append(np.array([rotvec(U) for U in Us]))
    for k, cv in enumerate(curves):
        if k:
            b.append(N.polyline([V.P(p) for p in cv], "line width=0.45pt, ndBlue!40"))
    ts, Us, Tt, c = geodesic(PHI, 0.0)
    main_c = curves[0]
    # the planes of available directions along the bold geodesic
    for k in (45, 100, 160, 205):
        U = Us[k]
        p = rotvec(U)
        e = 1e-5
        tx = (rotvec(expm(-0.5j * e * SX) @ U) - p) / e
        ty = (rotvec(expm(-0.5j * e * SY) @ U) - p) / e
        tx, ty = 0.13 * tx / np.linalg.norm(tx), 0.13 * ty / np.linalg.norm(ty)
        quad = [p - tx - ty, p + tx - ty, p + tx + ty, p - tx + ty]
        b.append("\\filldraw[fill=ndSky, fill opacity=0.85, draw=ndBlue!60, line width=0.3pt] %s -- cycle;" % " -- ".join(V.pt(q) for q in quad))
    b.append(N.polyline([V.P(p) for p in main_c], "line width=1.5pt, ndBlue"))
    k = 150
    d = main_c[k + 3] - main_c[k - 3]
    b.append("\\draw[->, >=stealth, line width=1.5pt, ndBlue] %s -- %s;" % (V.pt(main_c[k]), N.pt(V.P(main_c[k]) + 0.01 * (V.P(main_c[k] + d) - V.P(main_c[k])) / np.linalg.norm(V.P(main_c[k] + d) - V.P(main_c[k])))))
    b.append("\\node[dot] at %s {};" % V.pt((0, 0, 0)))
    b.append("\\node[dot, fill=ndRed] at %s {};" % V.pt((0, 0, PHI)))
    b.append("\\node[font=\\scriptsize, anchor=north east] at %s {$\\Id$};" % N.pt(o + np.array([-0.05, -0.04])))
    b.append("\\node[font=\\scriptsize, anchor=south west, text=ndRed] at %s {$V=\\ee^{-\\imath\\frac\\pi4\\sigma_z}$};" % N.pt(V.P((0, 0, PHI)) + np.array([0.08, 0.04])))
    b.append("\\node[note, anchor=west, text=ndGold!55!ndInk] (rg) at %s {$\\ee^{-\\imath t\\sigma_z/2}$:\\\\ not available};" % N.pt(o + np.array([2.65, 0.75])))
    b.append("\\draw[pointer] (rg.west) to[bend left=10] %s;" % N.pt(V.P((0, 0, 0.55 * PHI)) + np.array([0.06, 0.0])))
    b.append("\\node[note, anchor=east] (pl) at %s {available\\\\ directions at $\\Id$};" % N.pt(o + np.array([-1.25, -0.55])))
    b.append("\\draw[pointer] (pl.east) to[bend right=10] %s;" % N.pt(V.P((-0.3, 0.45, 0)) + np.array([0.0, 0.0])))
    pk = main_c[120]
    b.append("\\node[note, anchor=west, text=ndBlue] (sg) at %s {a circle of\\\\ geodesics};" % N.pt(o + np.array([2.0, 2.0])))
    far = max(curves, key=lambda cv: V.P(cv[100])[0])
    b.append("\\draw[pointer] (sg.west) to[bend right=10] %s;" % N.pt(V.P(far[100]) + np.array([0.05, 0.0])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) geodesics from $\\Id$ to $V$};" % N.pt(o + np.array([0.1, -1.35])))

    # ---- (b) the least time
    phis = np.linspace(0, 2 * np.pi, 241)
    pl = N.Plot((0, 2 * np.pi), (0, 2 * np.pi), 3.6, 3.4, origin=(7.75, 0.55))
    b += pl.axes(xlabel="{\\scriptsize $\\varphi$}", ylabel="{\\scriptsize $T$}",
                 xticks=[(np.pi / 2, "$\\frac\\pi2$"), (np.pi, "$\\pi$"), (2 * np.pi, "$2\\pi$")],
                 yticks=[(np.pi, "$\\pi$"), (2 * np.pi, "$2\\pi$")])
    b.append(pl.curve(phis, phis, "line width=1.0pt, ndGold, dash pattern=on 3pt off 2pt"))
    b.append(pl.curve(phis, least_time(phis), "line width=1.4pt, ndBlue"))
    b.append("\\node[dot, fill=ndRed] at %s {};" % pl.pt(PHI, least_time(PHI)))
    b.append("\\node[note, anchor=south, text=ndBlue] at %s {$x$, $y$ only};" % pl.pt(3.0, 5.75))
    b.append("\\node[note, anchor=west, text=ndGold!55!ndInk] at %s {with $\\sigma_z$};" % pl.pt(4.5, 3.3))
    b.append("\\node[note, anchor=north west] (sm) at %s {small angles:\\\\ $T\\approx2\\sqrt{\\pi\\varphi}$};" % pl.pt(0.35, 1.7))
    b.append("\\draw[pointer] (sm.north west) to[bend left=20] %s;" % pl.pt(0.12, least_time(0.12) - 0.15))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the least time};" % pl.pt(np.pi, -1.05))

    # ---- (c) the field of the bold geodesic
    cc = np.array([13.75, 2.35])
    R = 1.3
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$u_x$};" % (N.pt(cc + [-1.6, 0]), N.pt(cc + [1.65, 0])))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$u_y$};" % (N.pt(cc + [0, -1.6]), N.pt(cc + [0, 1.65])))
    b.append("\\draw[faint] %s circle (%.3f);" % (N.pt(cc), R))
    us = np.array([field(Us[k], Us[k + 1], ts[1]) for k in range(len(Us) - 1)])
    sampled = np.unwrap(np.arctan2(us[:, 1], us[:, 0]))
    assert np.allclose(sampled, -c * (ts[:-1] + ts[1:]) / 2, atol=1e-6)    # the field turns clockwise at the rate c
    ang = -c * ts
    b.append(N.polyline([cc + R * np.array([np.cos(a), np.sin(a)]) for a in ang], "line width=1.5pt, ndBlue"))
    a_end = ang[-1]
    p_end = cc + R * np.array([np.cos(a_end), np.sin(a_end)])
    tang = np.sign(ang[-1] - ang[0]) * np.array([-np.sin(a_end), np.cos(a_end)])
    b.append("\\draw[->, >=stealth, line width=1.5pt, ndBlue] %s -- %s;" % (N.pt(p_end - 0.02 * tang), N.pt(p_end)))
    p0 = cc + R * np.array([np.cos(ang[0]), np.sin(ang[0])])
    b.append("\\draw[vec, ndInk!70, line width=0.8pt] %s -- %s;" % (N.pt(cc), N.pt(p0)))
    b.append("\\node[dot] at %s {};" % N.pt(p0))
    b.append("\\node[font=\\scriptsize, anchor=south west] at %s {$t=0$};" % N.pt(p0 + [0.04, 0.03]))
    b.append("\\node[font=\\scriptsize, anchor=north west] at %s {$t=T$};" % N.pt(p_end + [0.06, -0.02]))
    sweep = abs(ang[-1] - ang[0])
    assert abs(sweep - (2 * np.pi - PHI)) < 1e-3
    b.append("\\node[note, anchor=north] at %s {unit strength, turning\\\\ at the steady rate $c=1.13$\\\\ through $2\\pi-\\varphi$};" % N.pt(cc + [0, -1.75]))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the field};" % N.pt(cc + [0, -2.95]))
    N.write("ch09_subriemann", b)


if __name__ == "__main__":
    main()
    print("written figures/ch09_subriemann.tex")
