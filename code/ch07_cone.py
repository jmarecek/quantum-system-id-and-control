"""ch07_cone.py -- a three-dimensional section of the qutrit state space.

In the coherence-vector coordinates of Chapter 2, rho = I/3 + sum_j x_j F_j with
F_j = lambda_j / sqrt 2 (normalised Gell-Mann matrices), we keep the three
coordinates (x1, x3, x8) along
    F1 = (|0><1| + |1><0|)/sqrt2,  F3 = diag(1,-1,0)/sqrt2,  F8 = diag(1,1,-2)/sqrt6
and set the other five to zero.  The states in this 3-plane form a solid
circular cone with axis x8:
  * apex: the pure state |2><2| at x8 = -sqrt(2/3);
  * base: the disc x8 = 1/sqrt6 of radius 1/sqrt2, made of the states supported
    on span{|0>,|1>}, a face of the state space and a copy of the (x, z) disc of
    a qubit's Bloch ball;
  * lateral surface: radius r(x8) = sqrt2/3 + x8/sqrt3, half-angle 30 degrees;
  * the inscribed sphere (radius 1/sqrt6) touches the base at its centre and
    the lateral surface; the circumscribed sphere (radius sqrt(2/3)) passes
    through the apex and the rim;
  * the axial section x1 = 0 is the triangle of diagonal states of Chapter 2.
The script finds the boundary by bisection on the radius in many directions,
checks it against the cone, and checks the ranks: one at the apex and on the
rim, two on the lateral surface and on the base, three inside.
It also estimates, by Monte Carlo, the probability that a pure state plus a
small traceless GUE error eps X remains a state (the compression of X to the
complement of the pure state must be positive semidefinite): 1/2 for N = 2,
about 0.03 for N = 3 and about 1e-4 for N = 4.  This is not drawn.
Writes figures/ch07_cone.tex.
"""
import numpy as np
from scipy.spatial import ConvexHull
import needham as N

S2, S3, S6 = np.sqrt(2.0), np.sqrt(3.0), np.sqrt(6.0)
F1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex) / S2
F3 = np.diag([1.0, -1.0, 0.0]).astype(complex) / S2
F8 = np.diag([1.0, 1.0, -2.0]).astype(complex) / S6
X8_APEX, X8_BASE, R_BASE = -np.sqrt(2.0 / 3.0), 1.0 / S6, 1.0 / S2
R_IN, R_OUT = 1.0 / S6, np.sqrt(2.0 / 3.0)


def rho(x):
    """The density matrix with coherence coordinates x = (x1, x3, x8)."""
    return np.eye(3) / 3 + x[0] * F1 + x[1] * F3 + x[2] * F8


def radius(x8):
    """Radius of the section of the cone at height x8."""
    return S2 / 3 + x8 / S3


def boundary(u):
    """The point where the ray from the centre in the direction u leaves the state space."""
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.linalg.eigvalsh(rho(mid * u)).min() >= -1e-13:
            lo = mid
        else:
            hi = mid
    return lo * u


def rank(x, tol=1e-9):
    return int((np.linalg.eigvalsh(rho(x)) > tol).sum())


def check():
    rng = np.random.default_rng(7)
    U = rng.normal(size=(4000, 3))
    U /= np.linalg.norm(U, axis=1)[:, None]
    B = np.array([boundary(u) for u in U])
    rr, h = np.hypot(B[:, 0], B[:, 1]), B[:, 2]
    on_base = np.abs(h - X8_BASE) < 1e-9
    on_side = np.abs(rr - radius(h)) < 1e-9
    assert np.all(on_base | on_side)                          # the boundary is the cone
    assert np.all(rr[on_base] <= R_BASE + 1e-9) and np.all((h[on_side] >= X8_APEX - 1e-9) & (h[on_side] <= X8_BASE + 1e-9))
    nb = np.linalg.norm(B, axis=1)
    assert abs(nb.min() - R_IN) < 2e-3 and nb.max() <= R_OUT + 1e-9      # inscribed and circumscribed spheres
    assert abs(radius(X8_APEX)) < 1e-12 and abs(radius(X8_BASE) - R_BASE) < 1e-12
    assert abs(np.degrees(np.arctan(R_BASE / (X8_BASE - X8_APEX))) - 30.0) < 1e-9   # half-angle
    # the inscribed sphere touches the lateral surface: distance from the centre to a generator
    apex, rim = np.array([0, X8_APEX]), np.array([R_BASE, X8_BASE])
    g = (rim - apex) / np.linalg.norm(rim - apex)
    assert abs(abs(apex[0] * g[1] - apex[1] * g[0]) - R_IN) < 1e-12
    # ranks
    assert rank([0, 0, X8_APEX]) == 1
    for phi in np.linspace(0, 2 * np.pi, 13):
        assert rank([R_BASE * np.cos(phi), R_BASE * np.sin(phi), X8_BASE]) == 1        # rim: pure
        assert rank([0.5 * R_BASE * np.cos(phi), 0.5 * R_BASE * np.sin(phi), X8_BASE]) == 2   # base
        s = 0.3
        assert rank([radius(s) * np.cos(phi), radius(s) * np.sin(phi), s]) == 2       # lateral surface
        assert rank([0.5 * radius(s) * np.cos(phi), 0.5 * radius(s) * np.sin(phi), s]) == 3   # inside
    # the base is a face: its states are supported on span{|0>,|1>}
    for x in ([0.2, -0.4, X8_BASE], [0.0, 0.0, X8_BASE]):
        assert abs(rho(x)[2, 2]) < 1e-12
    # the axial triangle: |0>, |1>, |2> at the expected coordinates
    for k, x in [(0, [0, 1 / S2, 1 / S6]), (1, [0, -1 / S2, 1 / S6]), (2, [0, 0, X8_APEX])]:
        e = np.zeros(3); e[k] = 1
        assert np.allclose(rho(x), np.outer(e, e))
    print("cone: apex x8 = %.4f, base x8 = %.4f, base radius %.4f, half-angle 30 deg, "
          "r_in = %.4f (min boundary distance %.4f), r_out = %.4f (max %.4f)"
          % (X8_APEX, X8_BASE, R_BASE, R_IN, nb.min(), R_OUT, nb.max()))


def perturbed_pure(n_samples=400000, seed=11):
    """Probability that rho + eps X stays a state for small eps, rho = |0><0| pure on C^N and X a
    traceless GUE matrix: the compression of X to the complement of |0> must be positive semidefinite."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in (2, 3, 4):
        A = rng.normal(size=(n_samples, n, n)) + 1j * rng.normal(size=(n_samples, n, n))
        X = (A + np.conj(np.swapaxes(A, 1, 2))) / 2                         # GUE
        X -= np.trace(X, axis1=1, axis2=2)[:, None, None].real / n * np.eye(n)   # traceless
        out[n] = np.mean(np.linalg.eigvalsh(X[:, 1:, 1:]).min(axis=1) >= 0)
    assert abs(out[2] - 0.5) < 0.01 and 0.02 <= out[3] <= 0.04 and 2e-5 <= out[4] <= 3e-4
    print("pure state + eps X (traceless GUE) stays a state with probability %.4f (N=2), %.4f (N=3), %.2e (N=4)"
          % (out[2], out[3], out[4]))
    return out


def main():
    check()
    perturbed_pure()
    az, el = 32.0, 17.0
    V = N.View(az=az, el=el, R=2.6)
    d = V.d
    apex = np.array([0.0, 0.0, X8_APEX])
    phi = np.linspace(0, 2 * np.pi, 241)
    rim = np.stack([R_BASE * np.cos(phi), R_BASE * np.sin(phi), np.full_like(phi, X8_BASE)], 1)
    b = []
    # the outer, circumscribed sphere, faintly
    b.append("\\draw[faint, densely dashed] %s circle (%.3f);" % (V.pt([0, 0, 0]), V.R * R_OUT))
    # the solid cone: silhouette washed, the base a stronger wash
    P2 = np.array([V.P(apex)] + [V.P(p) for p in rim])
    hull = ConvexHull(P2)
    b.append("\\fill[ndSky, fill opacity=0.45] %s -- cycle;" % " -- ".join(N.pt(P2[i]) for i in hull.vertices))
    b.append("\\fill[ndSky!85!ndBlue, fill opacity=0.55] %s -- cycle;" % " -- ".join(V.pt(p) for p in rim[:-1]))
    # generators facing the viewer: outward normal (cos p, sin p, -tan 30) . d > 0
    t30 = np.tan(np.radians(30.0))
    for p in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        nrm = np.array([np.cos(p), np.sin(p), -t30])
        if nrm @ d > 0.08:
            b.append("\\draw[faint] %s -- %s;" % (V.pt(apex), V.pt([R_BASE * np.cos(p), R_BASE * np.sin(p), X8_BASE])))
    # silhouette generators: normal perpendicular to the viewing direction
    a, e = np.radians(az), np.radians(el)
    for s in (+1, -1):
        p = a + s * np.arccos(t30 * np.tan(e))
        assert abs(np.array([np.cos(p), np.sin(p), -t30]) @ d) < 1e-9
        b.append("\\draw[ink] %s -- %s;" % (V.pt(apex), V.pt([R_BASE * np.cos(p), R_BASE * np.sin(p), X8_BASE])))
    # the inscribed sphere, touching the base at its centre and the sides
    b.append("\\draw[ndGreen, line width=0.6pt, dash pattern=on 2.5pt off 1.8pt] %s circle (%.3f);" % (V.pt([0, 0, 0]), V.R * R_IN))
    # the axial triangle of diagonal states (x1 = 0)
    k0, k1 = np.array([0, 1 / S2, 1 / S6]), np.array([0, -1 / S2, 1 / S6])
    b.append("\\fill[ndBlue, fill opacity=0.13] %s -- %s -- %s -- cycle;" % (V.pt(k0), V.pt(k1), V.pt(apex)))
    for q in (k0, k1):                                        # edges on the lateral surface: front or behind
        nrm = np.array([q[0], q[1], -t30 * R_BASE]) / R_BASE
        style = "line width=0.8pt, ndBlue" if nrm @ d > 0 else "line width=0.6pt, ndBlue, densely dashed"
        b.append("\\draw[%s] %s -- %s;" % (style, V.pt(apex), V.pt(q)))
    b.append("\\draw[line width=0.8pt, ndBlue] %s -- %s;" % (V.pt(k0), V.pt(k1)))
    # the rim, in front of everything
    b.append(N.polyline([V.P(p) for p in rim], "ink"))
    # axis x8
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (V.pt(apex), V.pt([0, 0, X8_BASE])))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, text=ndInk, font=\\small] {$x_8$};"
             % (V.pt([0, 0, X8_BASE]), V.pt([0, 0, X8_BASE + 0.42])))
    # points
    b.append("\\node[dot, fill=ndRed] at %s {};" % V.pt(apex))
    b.append("\\node[dot, fill=ndRed] at %s {};" % V.pt(k0))
    b.append("\\node[dot, fill=ndRed] at %s {};" % V.pt(k1))
    b.append("\\node[dot, minimum size=3pt] at %s {};" % V.pt([0, 0, 0]))
    b.append("\\node[dot, fill=ndGreen, minimum size=2.6pt] at %s {};" % V.pt([0, 0, X8_BASE]))
    b.append("\\node[font=\\small, anchor=north] at %s {$\\ket2$};" % N.pt(V.P(apex) + np.array([0, -0.08])))
    b.append("\\node[font=\\small, anchor=west] at %s {$\\ket0$};" % N.pt(V.P(k0) + np.array([0.08, 0.02])))
    b.append("\\node[font=\\small, anchor=east] at %s {$\\ket1$};" % N.pt(V.P(k1) + np.array([-0.08, 0.02])))
    b.append("\\node[font=\\small, anchor=west] at %s {$\\tfrac13\\Id$};" % N.pt(V.P([0, 0, 0]) + np.array([0.06, -0.12])))
    # notes
    ph = np.radians(240.0)                                   # a point of the base behind the diameter
    base_pt = V.P([0.6 * R_BASE * np.cos(ph), 0.6 * R_BASE * np.sin(ph), X8_BASE])
    b.append("\\node[note, anchor=east] (fa) at %s {the base is a face:\\\\ the states on $\\mathrm{span}\\{\\ket0,\\ket1\\}$,\\\\ a copy of the disc of a qubit};"
             % N.pt(V.P(k1) + np.array([-0.55, 0.85])))
    b.append("\\draw[pointer] (fa.east) to[bend left=10] %s;" % N.pt(base_pt))
    side = V.P([radius(-0.3) * np.cos(a + 0.75), radius(-0.3) * np.sin(a + 0.75), -0.3])
    b.append("\\node[note, anchor=west] (si) at %s {lateral surface:\\\\ rank two};" % N.pt(V.P(k0) + np.array([0.6, -1.15])))
    b.append("\\draw[pointer] (si.west) to[bend right=12] %s;" % N.pt(side))
    b.append("\\node[note, anchor=west, text=ndRed] (pu) at %s {pure states: the\\\\ apex and the rim only};"
             % N.pt(V.P(apex) + np.array([0.55, 0.05])))
    b.append("\\draw[pointer] (pu.west) to[bend left=10] %s;" % N.pt(V.P(apex) + np.array([0.09, 0.02])))
    b.append("\\node[note, anchor=east] (fr) at %s {full rank inside};" % N.pt(V.P(k1) + np.array([-0.5, -1.25])))
    b.append("\\draw[pointer] (fr.east) to[bend right=10] %s;" % N.pt(V.P([0, -0.1, -0.12])))
    b.append("\\node[note, anchor=north east, text=ndInk!55] at %s {outer sphere,\\\\ radius $\\sqrt{2/3}$};"
             % N.pt(V.P([0, 0, 0]) + V.R * R_OUT * np.array([np.cos(3.75), np.sin(3.75)])))
    b.append("\\node[note, anchor=west, text=ndGreen] (ins) at %s {inscribed sphere,\\\\ radius $1/\\sqrt6$};"
             % N.pt(V.P(k0) + np.array([0.6, -0.35])))
    b.append("\\draw[pointer] (ins.west) to[bend right=10] %s;" % N.pt(V.P([0, 0, 0]) + V.R * R_IN * np.array([np.cos(0.25), np.sin(0.25)])))
    N.write("ch07_cone", b)
    print("written figures/ch07_cone.tex")


if __name__ == "__main__":
    main()
