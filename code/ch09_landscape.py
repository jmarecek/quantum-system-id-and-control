"""ch09_landscape.py -- where the traps of a control landscape come from.

The driven qubit of Chapter 4 (omega = 1), H = (1/2)(-omega sigma_z + u sigma_x),
with a control of two constant steps u1, u2, each of duration tau = 1.6, and
the landscape F(u1, u2) = |<1|U(2 tau)|0>|^2 = (1 - z)/2, z the height of the
final Bloch vector.  F = F_kin o E, where E: (u1, u2) -> r(2 tau) is the
endpoint map and F_kin(r) = (1 - z)/2 the kinematic landscape on the sphere,
whose only critical points are its maximum |1> and its minimum |0>.  At a
control where the differential of E is invertible, E is a local
diffeomorphism, so that a critical point of F with 0 < F < 1 can only occur
where E is singular, det dE = r.(d1 r x d2 r) = 0 (red curves).

(a) The landscape on [-3.6, 3.6]^2 (contours at F = 0.1, ..., 0.9), with its
    four global maxima (F = 1, gold), its two traps (local maxima with
    F = 0.885, red crosses), at the constant pulses u1 = u2 = +-2.788, and its
    saddles (diamonds), all of which lie on the red curves.  The dashed square
    is the bound |u_i| <= 1.5 of Figure 9.2: no control in it reaches |1>, and
    the maxima of F on it are four points of its edge, with F = 0.847 (open
    circles).
(b) The fold behind the trap.  The images of the lines u1 = const (blue) and
    u2 = const (green) through a square of side 1.4 about the trap, drawn in
    longitude and height about the trap's endpoint.  They fill a region whose
    lower edge (red) is the image of the singular curve through the trap,
    along which E folds over; the region touches the latitude through the trap (gold) from above,
    so that no control nearby brings the state further south.
Writes figures/ch09_landscape.tex.
"""
import numpy as np
from scipy.optimize import root, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import needham as N

OMEGA_Q, TAU, U0 = 1.0, 1.6, 1.5
DELTA = -OMEGA_Q
L = 3.6


def endpoint(u1, u2):
    """Final Bloch vectors for arrays of the two step amplitudes (vectorised Rodrigues rotations)."""
    u1, u2 = np.broadcast_arrays(np.asarray(u1, float), np.asarray(u2, float))
    r = np.zeros(u1.shape + (3,))
    r[..., 2] = 1.0
    for u in (u1, u2):
        w = np.sqrt(u ** 2 + DELTA ** 2)
        n = np.stack([u / w, np.zeros_like(u), DELTA / w], -1)
        th = (w * TAU)[..., None]
        r = r * np.cos(th) + np.cross(n, r) * np.sin(th) + n * np.sum(n * r, -1, keepdims=True) * (1 - np.cos(th))
    return r


def F(u):
    return (1 - endpoint(u[0], u[1])[..., 2]) / 2


def grad(u, e=1e-6):
    return np.array([(F(u + e * v) - F(u - e * v)) / (2 * e) for v in np.eye(2)])


def hess(u, e=1e-4):
    return np.array([(grad(u + e * v) - grad(u - e * v)) / (2 * e) for v in np.eye(2)])


def det_dE(u1, u2, e=1e-6):
    r = endpoint(u1, u2)
    d1 = (endpoint(u1 + e, u2) - endpoint(u1 - e, u2)) / (2 * e)
    d2 = (endpoint(u1, u2 + e) - endpoint(u1, u2 - e)) / (2 * e)
    return np.sum(r * np.cross(d1, d2), -1)


def critical_points():
    pts = []
    for a in np.linspace(-L, L, 37):
        for b in np.linspace(-L, L, 37):
            s = root(grad, [a, b], tol=1e-13)
            if s.success and np.all(np.abs(s.x) <= L) and np.linalg.norm(grad(s.x)) < 1e-8:
                if not any(np.linalg.norm(s.x - p) < 1e-4 for p, _ in pts):
                    ev = np.linalg.eigvalsh(hess(s.x))
                    kind = "max" if ev.max() < -1e-6 else ("min" if ev.min() > 1e-6 else "saddle")
                    pts.append((s.x, kind))
    return pts


def box_maxima():
    """Local maxima of F on the square |u_i| <= U0: L-BFGS-B from a grid of starts, kept if the KKT conditions hold."""
    found = []
    for a in np.linspace(-U0, U0, 13):
        for c in np.linspace(-U0, U0, 13):
            x = minimize(lambda u: -F(u), [a, c], method="L-BFGS-B", bounds=[(-U0, U0)] * 2,
                         options={"ftol": 1e-15, "gtol": 1e-12}).x
            gr = grad(x)
            act = np.isclose(np.abs(x), U0, atol=1e-7)
            kkt = all((gr[i] * np.sign(x[i]) >= -1e-7) if act[i] else abs(gr[i]) < 1e-5 for i in range(2))
            if kkt and act.any() and not any(np.linalg.norm(x - p) < 1e-3 for p in found):
                found.append(x)
    return found


def checks(pts, edge):
    maxima = [p for p, k in pts if k == "max"]
    glob = [p for p in maxima if F(p) > 1 - 1e-9]
    traps = [p for p in maxima if F(p) < 1 - 1e-6]
    assert len(glob) == 4 and len(traps) == 2
    for p in traps:
        assert abs(p[0] - p[1]) < 1e-6 and abs(abs(p[0]) - 2.788) < 1e-3 and abs(F(p) - 0.8854) < 1e-4
    d = np.linspace(-L, L, 7201)                                     # the traps are the best constant pulses
    assert abs(F((d, d)).max() - F(traps[0])) < 1e-6
    # the kinematic argument: critical points with 0 < F < 1 are singular points of E, the global extrema are not
    for p, k in pts:
        f = F(p)
        if 1e-6 < f < 1 - 1e-6:
            assert abs(det_dE(*p)) < 1e-6, (p, k, f)
        elif k == "max" or (k == "min" and np.linalg.norm(p) < 2):
            assert abs(det_dE(*p)) > 0.1
    # inside the bound |u_i| <= 1.5 nothing reaches |1>; the maxima there lie on the edge, at F = 0.847
    g = np.linspace(-U0, U0, 301)
    assert abs(F(np.meshgrid(g, g)).max() - 0.84696) < 1e-4
    assert len(edge) == 4 and all(np.isclose(np.abs(x).max(), U0) and abs(F(x) - 0.84696) < 1e-4 for x in edge)
    return glob, traps


def contour_segments(Z, g, level):
    cs = plt.contour(g, g, Z, levels=[level])
    segs = cs.allsegs[0]
    plt.close("all")
    return segs


def main():
    pts = critical_points()
    edge = box_maxima()
    glob, traps = checks(pts, edge)
    b = []
    # ---- (a) the landscape
    S = 0.78
    oa = np.array([3.0, 3.0])
    P = lambda v: oa + S * np.asarray(v)
    g = np.linspace(-L, L, 361)
    U1, U2 = np.meshgrid(g, g)
    Z = (1 - endpoint(U1, U2)[..., 2]) / 2
    Dt = det_dE(U1, U2)
    b.append("\\fill[ndPaper] %s rectangle %s;" % (N.pt(P([-L, -L])), N.pt(P([L, L]))))
    for lev in np.arange(0.1, 0.95, 0.1):
        shade = int(15 + 75 * lev)
        for seg in contour_segments(Z, g, lev):
            if len(seg) > 2:
                b.append(N.polyline([P(q) for q in seg[::2]] + [P(seg[-1])], "line width=0.45pt, ndBlue!%d" % shade))
    for seg in contour_segments(Dt, g, 0.0):
        if len(seg) > 2:
            b.append(N.polyline([P(q) for q in seg[::2]] + [P(seg[-1])], "line width=1.0pt, ndRed"))
    b.append("\\draw[ink] %s rectangle %s;" % (N.pt(P([-L, -L])), N.pt(P([L, L]))))
    b.append("\\draw[dashedink, line width=0.7pt] %s rectangle %s;" % (N.pt(P([-U0, -U0])), N.pt(P([U0, U0]))))
    for p, k in pts:
        if k == "saddle":
            q = P(p)
            b.append("\\draw[ndInk, fill=white, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- cycle;"
                     % (q[0] - 0.07, q[1], q[0], q[1] + 0.07, q[0] + 0.07, q[1], q[0], q[1] - 0.07))
    for p in glob:
        b.append("\\node[dot, fill=ndGold, minimum size=4.2pt] at %s {};" % N.pt(P(p)))
    for p in traps:
        q = P(p)
        b.append("\\draw[ndRed, line width=1.4pt] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
                 % (q[0] - 0.09, q[1] - 0.09, q[0] + 0.09, q[1] + 0.09, q[0] - 0.09, q[1] + 0.09, q[0] + 0.09, q[1] - 0.09))
    for x in edge:
        b.append("\\draw[ndInk, fill=white, line width=0.7pt] %s circle (1.8pt);" % N.pt(P(x)))
    for v, lab in [(-3, "$-3$"), (0, "$0$"), (3, "$3$")]:
        b.append("\\node[font=\\scriptsize, anchor=north] at %s {%s};" % (N.pt(P([v, -L]) + [0, -0.03]), lab))
        b.append("\\node[font=\\scriptsize, anchor=east] at %s {%s};" % (N.pt(P([-L, v]) + [-0.03, 0]), lab))
    b.append("\\node[font=\\scriptsize] at %s {$u_1$};" % N.pt(P([L, -L]) + [0.25, -0.22]))
    b.append("\\node[font=\\scriptsize] at %s {$u_2$};" % N.pt(P([-L, L]) + [-0.3, 0.2]))
    t = traps[0] if traps[0][0] > 0 else traps[1]
    b.append("\\node[note, anchor=south, text=ndRed] (tr) at %s {a trap: the best\\\\ constant pulse};" % N.pt(P([2.0, L + 0.15])))
    b.append("\\draw[pointer] (tr.south) to[bend left=10] %s;" % N.pt(P(t) + [-0.06, 0.1]))
    gm = max(glob, key=lambda p: p[0] - p[1])
    b.append("\\node[note, anchor=north, text=ndGold!55!ndInk] (gm) at %s {$F=1$: the state\\\\ reaches $\\ket1$};" % N.pt(P([2.2, -L - 0.75])))
    b.append("\\draw[pointer] (gm.north) to[bend right=10] %s;" % N.pt(P(gm) + [0.02, -0.09]))
    b.append("\\node[note, anchor=north, text=ndRed] (dt) at %s {$\\det\\mathrm dE=0$};" % N.pt(P([-2.6, -L - 0.75])))
    allq = np.vstack(contour_segments(Dt, g, 0.0))
    tgt = min(allq, key=lambda q: np.linalg.norm(q - np.array([-2.9, -1.6])))
    b.append("\\draw[pointer] (dt.north) to[bend left=20] %s;" % N.pt(P(tgt)))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the landscape $F(u_1,u_2)$};" % N.pt(P([0, -L - 1.55])))

    # ---- (b) the fold, in longitude and height about the trap's endpoint
    rt = endpoint(*t)
    ph0 = np.arctan2(rt[1], rt[0])
    chart = lambda r: np.array([(np.arctan2(r[..., 1], r[..., 0]) - ph0 + np.pi) % (2 * np.pi) - np.pi, r[..., 2]])
    xl, yl = (-0.9, 0.9), (-0.80, -0.38)
    inside = lambda ch: (ch[0] > xl[0]) & (ch[0] < xl[1]) & (ch[1] < yl[1])
    h = 0.7
    pb = N.Plot(xl, yl, 3.6, 4.0, origin=(8.2, 0.95))
    b += pb.axes(xlabel="{\\scriptsize longitude}", ylabel="{\\scriptsize $z$}", xticks=[(0, "$0$")],
                 yticks=[(-0.5, "$-0.5$")])
    for c in np.linspace(-h, h, 11):
        d = np.linspace(-h, h, 241)
        for (a1, a2), col in [((t[0] + c + 0 * d, t[1] + d), "ndBlue!75"), ((t[0] + d, t[1] + c + 0 * d), "ndGreen!80")]:
            ch = chart(endpoint(a1, a2))
            idx = np.where(inside(ch))[0]
            for sg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
                if len(sg) > 1:
                    b.append(pb.curve(ch[0][sg], ch[1][sg], "line width=0.4pt, %s" % col))
    # the singular curve through the trap, mapped by E: the fold
    near = [sg for sg in contour_segments(Dt, g, 0.0) if np.min(np.linalg.norm(sg - t, axis=1)) < 0.05][0]
    near = near[np.linalg.norm(near - t, axis=1) < 1.2]
    ch = chart(endpoint(near[:, 0], near[:, 1]))
    order = np.argsort(ch[0])
    keep = inside(ch[:, order])
    b.append(pb.curve(ch[0][order][keep], ch[1][order][keep], "line width=1.3pt, ndRed"))
    b.append("\\draw[line width=1.0pt, ndGold, dash pattern=on 3pt off 2pt] %s -- %s;" % (pb.pt(xl[0], rt[2]), pb.pt(xl[1], rt[2])))
    q = pb.P(0, rt[2])
    b.append("\\draw[ndRed, line width=1.4pt] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
             % (q[0] - 0.09, q[1] - 0.09, q[0] + 0.09, q[1] + 0.09, q[0] - 0.09, q[1] + 0.09, q[0] + 0.09, q[1] - 0.09))
    i0 = np.argmin(np.abs(ch[0][order][keep] - 0.55))
    sp = (ch[0][order][keep][i0], ch[1][order][keep][i0])
    b.append("\\node[note, anchor=west, text=ndRed] (cp) at %s {the image of the\\\\ singular controls:\\\\ a fold};" % pb.pt(0.98, -0.7))
    b.append("\\draw[pointer] (cp.west) to[bend left=15] %s;" % pb.pt(sp[0] + 0.03, sp[1] - 0.01))
    b.append("\\node[note, anchor=east, text=ndGold!55!ndInk] at %s {latitude\\\\ of the trap};" % pb.pt(xl[0] - 0.03, rt[2]))
    b.append("\\draw[->, >=stealth, line width=0.6pt, ndInk!70] %s -- %s node[midway, right, font=\\scriptsize\\itshape, text=ndInk!80] {$F$ grows};"
             % (pb.pt(1.02, -0.44), pb.pt(1.02, -0.58)))
    b.append("\\node[note, anchor=south] at %s {images of the controls near the trap};" % pb.pt(0.3, -0.343))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the fold at the trap};" % pb.pt(0.0, -0.89))

    N.write("ch09_landscape", b)


if __name__ == "__main__":
    main()
    print("written figures/ch09_landscape.tex")
