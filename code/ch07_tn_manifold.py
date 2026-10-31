"""ch07_tn_manifold.py -- the manifold of matrix product states, and leaving it.

(a) A cartoon of the set M_D of matrix product states of bond dimension D as a
    curved sheet in the space of states.  At a point psi of the sheet the
    Schroedinger velocity -i H psi has a part along the sheet and a part off
    it.  The time-dependent variational principle (Dirac--Frenkel) keeps the
    part along the sheet, the orthogonal projection onto the tangent plane, so
    that the residual is orthogonal to the tangent plane; the variational
    trajectory stays on the sheet, while the exact trajectory leaves it, and
    truncation by singular values steps back onto the sheet.
(b) A quench of the controlled Ising chain H = sum Z_i Z_{i+1} + u sum X_i,
    N = 12, u = 1, from the product state |0...0> (bond dimension one).  Above:
    the entanglement entropy S(t) of the middle cut, which grows roughly
    linearly.  Below: the Fubini--Study distance between the exact state and
    its TT-SVD truncation to the bond dimension D = 2, 4, 8.  Each stays near
    zero until S(t) comes within about a bit of log2 D, and then rises.  The
    truncation is within a factor sqrt(N - 1) of the nearest point of M_D
    (Oseledets 2011), so that each curve bounds the distance from M_D above.
Writes figures/ch07_tn_manifold.tex.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply
import needham as N
from ch07_tn_entropy import ising, schmidt, entropy_bits, tt_svd, mps_vector


def f(x, y):
    """The sheet M_D of the cartoon: a gently curved graph over the (x, y) plane."""
    return -0.15 * x ** 2 - 0.10 * y ** 2 + 0.04 * x * y


def grad(x, y):
    return np.array([-0.30 * x + 0.04 * y, -0.20 * y + 0.04 * x])


def on_sheet(x, y):
    return np.array([x, y, f(x, y)])


def frame(x, y):
    """Orthonormal tangent vectors t1, t2 and the unit normal of the sheet at (x, y)."""
    fx, fy = grad(x, y)
    e1, e2 = np.array([1.0, 0.0, fx]), np.array([0.0, 1.0, fy])
    t1 = e1 / np.linalg.norm(e1)
    t2 = e2 - (e2 @ t1) * t1
    t2 /= np.linalg.norm(t2)
    n = np.cross(t1, t2)
    return t1, t2, n / np.linalg.norm(n)


def quench(n=12, u=1.0, T=3.0, num=121, Ds=(2, 4, 8)):
    H = ising(n, 1.0, u)
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0                                                       # |0...0>, a product state
    ts = np.linspace(0.0, T, num)
    states = expm_multiply(-1j * H, psi0, start=0.0, stop=T, num=num, endpoint=True)
    S = np.array([entropy_bits(schmidt(p, n // 2, n)) for p in states])
    dist = {}
    for D in Ds:
        d = []
        for p in states:
            q = mps_vector(tt_svd(p, n, D))
            d.append(np.arccos(min(1.0, abs(np.vdot(p, q)) / np.linalg.norm(q))))
        dist[D] = np.array(d)
    return ts, S, dist


def main():
    ts, S, dist = quench()
    assert S[0] < 1e-9 and all(d[0] < 1e-9 for d in dist.values())      # a product state lies on every M_D
    assert np.all(np.diff(S[ts <= 3.0]) > -1e-3)                        # the entropy grows over the window
    first = {}
    for D, d in dist.items():
        assert np.all(d[S < np.log2(D) - 1.2] < 0.03)                  # near zero while S is well below log2 D
        i = int(np.argmax(d > 0.1))
        first[D] = (ts[i], S[i])
    assert first[2][0] < first[4][0] < first[8][0]
    assert dist[2][-1] > dist[4][-1] > dist[8][-1]
    print("S(t) at t = 0, 1, 2, 3: %s bits" % np.round(S[[0, 40, 80, 120]], 3))
    for D in dist:
        print("D = %d: distance first exceeds 0.1 at t = %.3f, where S = %.2f bits (log2 D = %d); final distance %.3f"
              % (D, first[D][0], first[D][1], np.log2(D), dist[D][-1]))

    # ---- (a) the cartoon: geometry first, with its checks
    x0, y0 = -0.75, -0.35
    psi = on_sheet(x0, y0)
    t1, t2, nrm = frame(x0, y0)
    v = 1.15 * t1 + 0.20 * t2 + 0.55 * nrm                             # -i H psi, partly off the sheet
    Pv = (v @ t1) * t1 + (v @ t2) * t2                                  # its projection on the tangent plane
    res = v - Pv
    assert abs(res @ t1) < 1e-12 and abs(res @ t2) < 1e-12              # Dirac--Frenkel: residual orthogonal
    assert np.linalg.norm(Pv) < np.linalg.norm(v)
    s = np.linspace(0.0, 1.45, 70)
    exact = np.array([psi + si * v + si ** 2 * (0.06 * nrm + 0.10 * t1) for si in s])
    a1, a2 = Pv[0], Pv[1]
    var = np.array([on_sheet(x0 + si * a1 - 0.05 * si ** 2, y0 + si * a2 + 0.10 * si ** 2) for si in s])
    tangent0 = (var[1] - var[0]) / (s[1] - s[0])
    assert np.allclose(var[:, 2], f(var[:, 0], var[:, 1]))              # the variational path stays on the sheet
    assert np.linalg.norm(np.cross(tangent0 / np.linalg.norm(tangent0), Pv / np.linalg.norm(Pv))) < 0.05
    e0 = (exact[1] - exact[0]) / (s[1] - s[0])
    assert np.linalg.norm(np.cross(e0 / np.linalg.norm(e0), v / np.linalg.norm(v))) < 0.05
    end = exact[-1]                                                     # truncation: the nearest point of the sheet
    foot_xy = minimize(lambda z: np.sum((on_sheet(*z) - end) ** 2), end[:2]).x
    foot = on_sheet(*foot_xy)
    f1, f2, _ = frame(*foot_xy)
    assert abs((end - foot) @ f1) < 1e-4 and abs((end - foot) @ f2) < 1e-4   # it leaves along the normal

    V = N.View(az=-62.0, el=26.0, R=1.75, shift=(0.0, -0.2))
    b = []
    X, Y = 2.0, 1.25
    edge = ([on_sheet(x, -Y) for x in np.linspace(-X, X, 40)] + [on_sheet(X, y) for y in np.linspace(-Y, Y, 30)]
            + [on_sheet(x, Y) for x in np.linspace(X, -X, 40)] + [on_sheet(-X, y) for y in np.linspace(Y, -Y, 30)])
    b.append("\\fill[ndSky, fill opacity=0.5] %s -- cycle;" % " -- ".join(V.pt(p) for p in edge))
    for x in np.linspace(-X, X, 9)[1:-1]:
        b.append(N.polyline([V.P(on_sheet(x, y)) for y in np.linspace(-Y, Y, 30)], "line width=0.3pt, ndBlue!30"))
    for y in np.linspace(-Y, Y, 6)[1:-1]:
        b.append(N.polyline([V.P(on_sheet(x, y)) for x in np.linspace(-X, X, 40)], "line width=0.3pt, ndBlue!30"))
    b.append("\\draw[ndBlue, line width=0.8pt] %s -- cycle;" % " -- ".join(V.pt(p) for p in edge))
    # the tangent plane at psi
    h1, h2 = 0.85, 0.65
    corners = [psi + a * h1 * t1 + c * h2 * t2 for a, c in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    b.append("\\filldraw[fill=ndGold!30, fill opacity=0.8, draw=ndGold!70!ndInk, line width=0.6pt] %s -- cycle;" % " -- ".join(V.pt(p) for p in corners))
    # the trajectories, the truncation, the velocity and its projection
    b.append(N.polyline([V.P(p) for p in var], "line width=1.4pt, ndBlue"))
    b.append(N.polyline([V.P(p) for p in exact], "line width=1.0pt, ndRed, densely dashed"))
    b.append("\\draw[->, >=stealth, line width=1.0pt, ndGreen] %s -- %s;" % (V.pt(end), V.pt(foot)))
    b.append("\\node[dot, fill=ndRed, minimum size=3pt] at %s {};" % V.pt(end))
    b.append("\\draw[vec, ndRed] %s -- %s;" % (V.pt(psi), V.pt(psi + v)))
    b.append("\\draw[vec, ndBlue] %s -- %s;" % (V.pt(psi), V.pt(psi + Pv)))
    b.append("\\draw[ndInk!70, line width=0.6pt, dash pattern=on 2pt off 1.5pt] %s -- %s;" % (V.pt(psi + Pv), V.pt(psi + v)))
    b.append("\\node[dot] at %s {};" % V.pt(psi))
    P0, tip, ptip = V.P(psi), V.P(psi + v), V.P(psi + Pv)
    b.append("\\node[font=\\small, anchor=north east] at %s {$\\psi$};" % N.pt(P0 + np.array([-0.03, -0.03])))
    b.append("\\node[font=\\small, text=ndRed, anchor=south east] at %s {$-\\imath H\\psi$};" % N.pt(tip + np.array([-0.02, 0.02])))
    b.append("\\node[note, anchor=north, align=center, text=ndBlue] (pv) at %s {its projection:\\\\ the residual (dashed)\\\\ is orthogonal to $T_\\psi\\mathcal M_D$};" % N.pt(ptip + np.array([0.0, -0.75])))
    b.append("\\draw[pointer] (pv.north) -- %s;" % N.pt(ptip + np.array([0.0, -0.08])))
    c3 = V.P(corners[3])
    b.append("\\node[font=\\scriptsize, text=ndGold!55!ndInk, anchor=south west] at %s {$T_\\psi\\mathcal M_D$};" % N.pt(c3 + np.array([0.02, 0.02])))
    ep = V.P(end)
    b.append("\\node[note, anchor=south, text=ndRed] at %s {exact};" % N.pt(ep + np.array([0.0, 0.08])))
    b.append("\\node[note, anchor=west, align=left, text=ndGreen] at %s {truncation:\\\\ back to $\\mathcal M_D$};" % N.pt(0.5 * (ep + V.P(foot)) + np.array([0.22, 0.3])))
    vp = V.P(var[-1])
    b.append("\\node[note, anchor=north west, text=ndBlue] at %s {variational};" % N.pt(vp + np.array([0.05, -0.05])))
    corner = min((V.P(p) for p in edge), key=lambda q: q[0])            # the leftmost point of the sheet
    b.append("\\node[note, anchor=north west, align=left, text=ndBlue] at %s {$\\mathcal M_D$: the states\\\\ of bond dimension $D$};" % N.pt(corner + np.array([-0.05, -0.3])))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) the time-dependent variational principle};" % (0.0, -3.35))

    # ---- (b) the quench: entropy above, distances below
    cols = {2: "ndRed", 4: "ndGold", 8: "ndGreen"}
    ox = 5.3
    P1 = N.Plot((0.0, 3.0), (0.0, 3.2), 5.6, 1.9, origin=(ox, 0.75))
    b += P1.axes("", "$S(t)$, bits", yticks=[(1, "$1$"), (2, "$2$"), (3, "$3$")])
    for D in cols:
        b.append("\\draw[%s, densely dotted, line width=0.7pt] %s -- %s node[right, font=\\scriptsize, text=%s] {$\\log_2%d$};"
                 % (cols[D], P1.pt(0, np.log2(D)), P1.pt(3.0, np.log2(D)), cols[D], D))
    b.append(P1.curve(ts, S, "line width=1.3pt, ndBlue"))
    b.append("\\node[note, anchor=south east, text=ndBlue] at %s {middle cut};" % P1.pt(1.75, 1.45))
    P2 = N.Plot((0.0, 3.0), (0.0, 0.9), 5.6, 1.9, origin=(ox, -2.05))
    b += P2.axes("$t$", "distance", xticks=[(t, "$%d$" % t) for t in (1, 2, 3)], yticks=[(0.4, "$0.4$"), (0.8, "$0.8$")])
    for D in cols:
        b.append(P2.curve(ts, dist[D], "line width=1.3pt, %s" % cols[D]))
        b.append("\\node[font=\\scriptsize, text=%s, anchor=west] at %s {$D=%d$};" % (cols[D], P2.pt(3.02, dist[D][-1]), D))
        tf = first[D][0]
        b.append("\\draw[%s, line width=0.5pt, densely dashed] %s -- %s;" % (cols[D], P2.pt(tf, 0.1), P1.pt(tf, first[D][1])))
    b.append("\\node[note, anchor=north west, align=left] at %s {Fubini--Study distance\\\\ from the truncation};" % P2.pt(0.05, 0.86))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) a quench of the Ising chain, $N=12$, $u=1$};" % (ox + 2.8, -3.35))
    N.write("ch07_tn_manifold", b)
    print("written figures/ch07_tn_manifold.tex")


if __name__ == "__main__":
    main()
