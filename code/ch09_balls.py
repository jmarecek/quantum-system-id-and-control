"""ch09_balls.py -- Riemannian against sub-Riemannian balls of SU(2).

A qubit without drift and with a field of strength at most 1.  If the field
may point in any direction, H = (1/2) u.sigma with |u| <= 1, the gates
reachable by the time t are the Riemannian ball {theta n : theta <= t} in the
rotation-vector coordinates theta n of U = exp(-i theta n.sigma/2).  If it
must lie in the xy-plane, H = (1/2)(u_x sigma_x + u_y sigma_y), they form the
sub-Riemannian ball, whose boundary, the sphere of radius t, is swept by the
normal geodesics of ch09_subriemann.py,
    U_c(t) = exp(i c t sigma_z/2) exp(-i t (cos th0 sigma_x + sin th0 sigma_y + c sigma_z)/2),
each up to its cut time 2 pi/sqrt(1 + c^2), where it meets the z-axis
together with the whole circle of its rotations about that axis.  The balls
are solids of revolution about the z-axis.

(a) Their sections by the xz-plane for t = 1, ..., 6 (the whole group is the
    ball theta <= 2 pi).  The Riemannian ones are discs (dashed); the
    sub-Riemannian ones are flattened, apple-shaped, with an inward cusp where
    they meet the z-axis, at the heights +-phi with sqrt(phi (4 pi - phi)) = t,
    i.e. phi = 2 pi - sqrt(4 pi^2 - t^2) (eq. ch09:sr2).  Their greatest height
    is reached off the axis.
(b) The extents against t on logarithmic scales: both balls reach the
    distance t along the x-axis, the Riemannian ball also the height t, but
    the sub-Riemannian ball only the height h(t) ~ t^2/(2 pi) for small t
    (0.159 t^2 at t = 0.1), and phi(t) ~ t^2/(4 pi) on the axis.  Slopes 1
    and 2: the ball-box theorem.

Checks: the sphere of radius t is a simple curve in the half-plane (no two
geodesics meet before their cut time); geodesics continued past their cut
time end inside the ball; the horizontal extent is t; and gradient searches
over unit-strength fields (as in ch09_subriemann.py) find no control reaching
two points of the sphere of radius 3 sooner than t - 0.06, while t + 0.05
suffices.
Writes figures/ch09_balls.tex.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import needham as N
import ch09_subriemann as S

I2, SX, SY, SZ = S.I2, S.SX, S.SY, S.SZ
TS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def endpoint(c, t, th0=0.0):
    a, b = np.cos(th0), np.sin(th0)
    return expm(0.5j * c * t * SZ) @ expm(-0.5j * t * (a * SX + b * SY + c * SZ))


def rz(U):
    v = S.rotvec(U)
    return np.array([np.hypot(v[0], v[1]), v[2]])


def cmax(t):
    return np.sqrt((2 * np.pi / t) ** 2 - 1)


def axis_height(t):
    return 2 * np.pi - np.sqrt(4 * np.pi ** 2 - t * t)


def section(t, n=241):
    """The sphere of radius t in the (rho, z) half-plane, from the lower cusp to the upper one."""
    cm = cmax(t)
    cs = np.sinh(np.linspace(-np.arcsinh(cm), np.arcsinh(cm), n))      # denser near c = 0
    return np.array([rz(endpoint(c, t)) for c in cs])


def height(t):
    return np.abs(section(t, 601)[:, 1]).max()


def segments_cross(P):
    """Does the polyline P intersect itself (non-adjacent segments)?"""
    def cross(a, b, c, d):
        o = lambda p, q, r: np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
        return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0
    for i in range(len(P) - 1):
        for j in range(i + 2, len(P) - 1):
            if cross(P[i], P[i + 1], P[j], P[j + 1]):
                return True
    return False


def inside(p, curve):
    """Is the point p of the half-plane inside the region bounded by the curve and the z-axis? (ray casting)"""
    poly = np.vstack([curve, [[0.0, curve[-1, 1]], [0.0, curve[0, 1]]]])
    x, y = p
    c = False
    for k in range(len(poly)):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            c = not c
    return c


def best_fidelity(V, T, n=50, starts=8, seed=0):
    """max Re Tr(V^dag U)/2 over unit-strength xy fields, piecewise constant on n steps."""
    rng = np.random.default_rng(seed)

    def f(th):
        dt = T / n
        Us = [np.cos(dt / 2) * I2 - 1j * np.sin(dt / 2) * (np.cos(a) * SX + np.sin(a) * SY) for a in th]
        dU = [-1j * np.sin(dt / 2) * (-np.sin(a) * SX + np.cos(a) * SY) for a in th]
        fw = [I2]
        for U in Us:
            fw.append(U @ fw[-1])
        bw = [I2]
        for U in Us[::-1]:
            bw.append(bw[-1] @ U)
        bw = bw[::-1]
        val = np.real(np.trace(V.conj().T @ fw[-1])) / 2
        g = np.array([np.real(np.trace(V.conj().T @ bw[k + 1] @ dU[k] @ fw[k])) / 2 for k in range(n)])
        return -val, -g
    best = -1
    for _ in range(starts):
        r = minimize(f, rng.uniform(0, 2 * np.pi, n), jac=True, method="BFGS", options={"gtol": 1e-12, "maxiter": 3000})
        best = max(best, -r.fun)
    return best


def checks():
    for t in TS:
        sec = section(t)
        assert np.all(sec[:, 0] >= -1e-12)
        assert np.allclose(sec[0], [0, -axis_height(t)], atol=1e-7) and np.allclose(sec[-1], [0, axis_height(t)], atol=1e-7)
        assert abs(np.sqrt(axis_height(t) * (4 * np.pi - axis_height(t))) - t) < 1e-12
        assert abs(sec[:, 0].max() - t) < 1e-9                                # horizontal extent t (c = 0)
        assert np.all(np.linalg.norm(sec, axis=1) <= t + 1e-9)               # inside the Riemannian ball
        assert not segments_cross(sec)
        # the rotations of a geodesic about z give the same section
        for c in (-0.4 * cmax(t), 0.3 * cmax(t)):
            assert np.allclose(rz(endpoint(c, t, 1.3)), rz(endpoint(c, t)), atol=1e-10)
        # geodesics continued past their cut time end inside the ball of radius t
        for c in np.linspace(1.05 * cmax(t), 3 * cmax(t) + 2, 12):
            for sgn in (1, -1):
                p = rz(endpoint(sgn * c, t))
                assert inside(p, sec), (t, c, p)
    # small t: the height grows like t^2/(2 pi), the cusp like t^2/(4 pi)
    assert abs(height(0.1) / 0.01 - 1 / (2 * np.pi)) < 2e-3
    assert abs(axis_height(0.01) / 1e-4 - 1 / (4 * np.pi)) < 1e-5
    # minimality: points of the sphere of radius 3 are not reached sooner
    for c in (0.6, 1.5):
        V = endpoint(c, 3.0)
        assert best_fidelity(V, 3.0 - 0.06) < 1 - 1e-6
        assert best_fidelity(V, 3.0 + 0.05) > 1 - 1e-8


def main():
    checks()
    b = []
    # ---- (a) the sections
    s = 0.43
    o = np.array([3.0, 3.0])
    P = lambda q: o + s * np.asarray(q)
    R = 2 * np.pi
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$x$};" % (N.pt(P([-R - 0.3, 0])), N.pt(P([R + 0.5, 0]))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$z$};" % (N.pt(P([0, -R - 0.3])), N.pt(P([0, R + 0.5]))))
    b.append("\\draw[ink] %s circle (%.3f);" % (N.pt(P([0, 0])), s * R))
    shades = {1.0: 80, 2.0: 65, 3.0: 50, 4.0: 38, 5.0: 26, 6.0: 15}
    for t in reversed(TS):
        sec = section(t)
        full = np.vstack([sec, sec[::-1] * [-1, 1]])
        b.append("\\filldraw[fill=ndBlue!%d, draw=ndBlue!70!ndInk, line width=0.5pt] %s -- cycle;"
                 % (shades[t], " -- ".join(N.pt(P(q)) for q in full[::2])))
    for t in TS:
        b.append("\\draw[dashedink] %s circle (%.3f);" % (N.pt(P([0, 0])), s * t))
        for sg in (1, -1):
            b.append("\\node[dot, minimum size=2.2pt, fill=ndRed] at %s {};" % N.pt(P([0, sg * axis_height(t)])))
        b.append("\\node[font=\\tiny, anchor=north west, text=ndInk!80] at %s {%d};" % (N.pt(P([t, 0]) + [0.0, -0.02]), int(t)))
    b.append("\\node[note, anchor=west] (rb) at %s {Riemannian:\\\\ discs $\\theta\\le t$};" % N.pt(P([4.2, 5.6])))
    b.append("\\draw[pointer] (rb.south west) to[bend right=10] %s;" % N.pt(P([3.0 * np.cos(1.0), 3.0 * np.sin(1.0)])))
    b.append("\\node[note, anchor=west, text=ndBlue] (sb) at %s {$x$, $y$ only:\\\\ flattened apples};" % N.pt(P([4.2, -5.4])))
    b.append("\\draw[pointer] (sb.north west) to[bend left=10] %s;" % N.pt(P([3.3, -1.9])))
    b.append("\\node[note, anchor=east, text=ndRed] (cu) at %s {cusps on the\\\\ axis, at $\\pm\\varphi(t)$};" % N.pt(P([-5.0, 6.2])))
    b.append("\\draw[pointer] (cu.south east) to[bend right=15] %s;" % N.pt(P([-0.06, axis_height(4.0) + 0.05])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the balls of radius $t=1,\\dots,6$, cut through the $z$-axis};" % N.pt(P([0, -R - 0.75])))

    # ---- (b) the extents against t, on logarithmic scales
    tt = np.exp(np.linspace(np.log(0.05), np.log(6.0), 60))
    hh = np.array([height(t) for t in tt])
    ph = axis_height(tt)
    lx = (np.log10(0.05), np.log10(8.0))
    ly = (np.log10(2e-4), np.log10(8.0))
    pl = N.Plot(lx, ly, 5.2, 5.4, origin=(9.3, 0.3))
    b += pl.axes(xlabel="{\\scriptsize $t$}", ylabel="{\\scriptsize extent}",
                 xticks=[(np.log10(v), lab) for v, lab in [(0.1, "$0.1$"), (1, "$1$")]],
                 yticks=[(np.log10(v), lab) for v, lab in [(1e-3, "$10^{-3}$"), (1e-2, "$10^{-2}$"), (1e-1, "$10^{-1}$"), (1, "$1$")]])
    b.append(pl.curve(np.log10(tt), np.log10(tt), "line width=1.3pt, ndInk"))
    b.append(pl.curve(np.log10(tt), np.log10(hh), "line width=1.3pt, ndBlue"))
    b.append(pl.curve(np.log10(tt), np.log10(ph), "line width=0.9pt, ndRed, dash pattern=on 3pt off 2pt"))
    def along(tv, fn, txt, col, anchor):
        """A label written along a curve, rotated to its slope on the page."""
        p0, p1 = pl.P(np.log10(tv), np.log10(fn(tv))), pl.P(np.log10(tv * 1.1), np.log10(fn(tv * 1.1)))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        return "\\node[note, rotate=%.1f, anchor=%s, text=%s] at %s {%s};" % (ang, anchor, col, N.pt(p0), txt)
    b.append(along(0.12, lambda t: t, "width $t$ (both); height $t$ (Riemannian)", "ndInk!85", "south west"))
    b.append(along(0.07, lambda t: t * t / (2 * np.pi), "height, $x$, $y$ only: $\\approx t^2/2\\pi$", "ndBlue", "south west"))
    b.append(along(0.2, lambda t: t * t / (4 * np.pi), "on the axis: $\\varphi\\approx t^2/4\\pi$", "ndRed", "north west"))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) width and height of the balls};" % pl.pt((lx[0] + lx[1]) / 2, ly[0] - 0.75))
    N.write("ch09_balls", b)


if __name__ == "__main__":
    main()
    print("height(0.1)/0.01 = %.4f, 1/(2 pi) = %.4f" % (height(0.1) / 0.01, 1 / (2 * np.pi)))
    for t in TS:
        print("t = %g: cusp phi = %.4f, height = %.4f" % (t, axis_height(t), height(t)))
    print("written figures/ch09_balls.tex")
