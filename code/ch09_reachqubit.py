"""ch09_reachqubit.py -- the reachable sets of the driven qubit, and first contact.

The driven qubit of Chapter 4, H = (1/2)(Delta sigma_z + u sigma_x) with
Delta = -omega = -1 and |u| <= u0 = 1.5, started at |0>.  Its reachable sets
R(t) on the Bloch sphere are nested, since the drift leaves |0> fixed and one
may wait there.  They are computed by sampling about 3.8 * 10^5 controls for
each t: bang-bang controls with up to six switches, controls taking the values
-u0, 0, u0, random piecewise-constant controls, and the extremals of
Proposition 9.x (t1, t_b repeated, t2, with t1, t2 <= t_b) on a 260 x 260 grid
for each sign.  The end points are binned on a grid in an azimuthal
equidistant chart, about |0> for the small sets and about |1> for the holes,
closed, and the boundaries taken as contours.
  * The largest polar angle reachable at t, theta_max(t), computed over the
    extremals (400 x 400 grid), is 0.742, 1.422, 2.001, 2.667 and 2.957 at
    t = 0.5, 1, 2, 3 and 3.3, and pi at t* = 2 pi/Omega = 3.485.  A bounded
    gradient search over 40-step piecewise-constant controls (scratch
    computation) gave the same values to 1e-3.  theta_max(t) <= Omega t;
    without the drift the bound u0 would give u0 t, reaching pi at 2.094.
  * First contact is not a touch from outside.  From about t = 2.5 the set
    R(t) wraps round the sphere and leaves only a hole about |1>, a curved
    quadrilateral whose size is about pi - theta_max(t) (0.53, 0.20, 0.03 at
    t = 3, 3.3, 3.45), and which closes at t*: |1> is an interior point of
    R(t*), though it is reached at no earlier time.
  * Along the time-optimal path of ch09_timeoptimal.py (u = -u0 for t1, then
    +u0), r(t) lies on the outer boundary of R(t) up to about t = 2.3, with
    the adjoint p = H x r of ch09_switchsphere.py pointing out of R(t): no
    state 0.05 or 0.1 beyond it in that direction is reachable at t = 1 and 2.
    Afterwards r(t) rides on a corner of the hole about |1> (within 0.02 of
    the hole's boundary, at its greatest distance from |1>, at t = 2.75, 3,
    3.3), and p points roughly towards |1> (cos 0.91, 0.83, 0.73).  The
    corner is sharp, so that a step along p soon leaves the hole again.
  * So the target is not on the boundary of R(t*): the hole closes at t*, and
    |1> is an interior point of R(t*) that no earlier time reaches.  The
    statement that the end point of a time-optimal path lies on the boundary
    of the reachable set, true for the particle of Figure 9.1, fails here;
    what the maximum principle uses is the set reached by controls near the
    optimal one.
(a) R(0.5), R(1) and R(2) in the azimuthal equidistant map about |0> (the
    point at polar angle theta and azimuth phi drawn at theta (cos phi, sin phi)),
    with the optimal path, solid while it lies on the boundaries, and its
    adjoint p at t = 1 and 2.
(b) The sphere seen from below |1>: the unreachable holes at t = 2.75, 3,
    3.3 and 3.45, and the last stretches, from t = 3, of the four optimal
    paths, which arrive at |1> as the hole closes.
(c) theta_max(t) against t, with the bounds Omega t and the drift-free u0 t.
Writes figures/ch09_reachqubit.tex.
"""
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import needham as N
import ch09_timeoptimal as TO
import ch09_switchsphere as SW

DELTA, U0, W, T1, TSTAR = TO.DELTA, TO.U0, TO.W, TO.T1, TO.TSTAR
TIMES = [0.5, 1.0, 2.0]
HOLES = [2.75, 3.0, 3.3, 3.45]
GRID, LIM = 520, np.pi + 0.06


def rotate(r, u, dt):
    """Rotate Bloch vectors r (n, 3) about h = (u, 0, Delta) by |h| dt."""
    w = np.sqrt(u * u + DELTA * DELTA)
    n = np.stack([u / w, 0 * u, DELTA / w + 0 * u], 1)
    th = (w * dt)[:, None]
    return r * np.cos(th) + np.cross(n, r) * np.sin(th) + n * np.sum(n * r, 1, keepdims=True) * (1 - np.cos(th))


def run(us, durs):
    r = np.tile([0, 0, 1.0], (us.shape[0], 1))
    for j in range(us.shape[1]):
        r = rotate(r, us[:, j], durs[:, j])
    return r


def extremals(t, n):
    """End points of the extremals t1, t_b (m times), t2 with t1, t2 <= t_b, both signs."""
    P = 2 * np.pi / W
    out = []
    for s0 in (1, -1):
        TB, F1 = np.meshgrid(np.linspace(P / n, P, n), np.linspace(0, 1, n))
        tb = TB.ravel()
        t1 = np.minimum(F1.ravel() * tb, t)
        rem = t - t1
        mm = np.floor(rem / tb).astype(int)
        t2 = rem - mm * tb
        keep = mm <= 60
        tb, t1, t2, mm = tb[keep], t1[keep], t2[keep], mm[keep]
        r = np.tile([0, 0, 1.0], (len(tb), 1))
        sgn = np.full(len(tb), float(s0))
        r = rotate(r, U0 * sgn, t1)
        for j in range(mm.max() if len(mm) else 0):
            act = mm > j
            sgn = np.where(act, -sgn, sgn)
            r[act] = rotate(r[act], U0 * sgn[act], tb[act])
        out.append(rotate(r, -U0 * sgn, t2))
    return np.vstack(out)


def samples(t, rng, n=200000):
    out = []
    for k in range(7):
        m = n // 14
        cuts = np.sort(rng.uniform(0, t, (m, k)), 1)
        durs = np.diff(np.concatenate([np.zeros((m, 1)), cuts, np.full((m, 1), t)], 1), axis=1)
        signs = rng.choice([-1, 1], (m, 1)) * (-1) ** np.arange(k + 1)[None, :]
        out.append(run(U0 * signs.astype(float), durs))
        out.append(run(rng.choice([-U0, 0.0, U0], (m, k + 1)), durs))
    m, k = n // 4, 12
    out.append(run(rng.uniform(-U0, U0, (m, k)), np.full((m, k), t / k)))
    out.append(extremals(t, 260))
    return np.vstack(out)


def chart(r):
    th = np.arccos(np.clip(r[..., 2], -1, 1))
    ph = np.arctan2(r[..., 1], r[..., 0])
    return np.stack([th * np.cos(ph), th * np.sin(ph)], -1)


def unchart(c):
    th = np.linalg.norm(c, axis=-1)
    ph = np.arctan2(c[..., 1], c[..., 0])
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)


def occupancy(r):
    c = chart(r)
    idx = np.clip(((c + LIM) / (2 * LIM) * GRID).astype(int), 0, GRID - 1)
    occ = np.zeros((GRID, GRID), bool)
    occ[idx[:, 1], idx[:, 0]] = True
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3)), iterations=2)
    return ndimage.binary_fill_holes(occ)


def inside(occ, c):
    i = np.clip(((np.asarray(c) + LIM) / (2 * LIM) * GRID).astype(int), 0, GRID - 1)
    return occ[i[..., 1], i[..., 0]]


def boundary(occ):
    g = np.linspace(-LIM, LIM, GRID)
    cs = plt.contour(g, g, occ.astype(float), levels=[0.5])
    segs = [s for s in cs.allsegs[0] if len(s) > 20]
    plt.close("all")
    return max(segs, key=len)


def theta_max(t):
    return float(np.arccos(np.clip(extremals(t, 400)[:, 2], -1, 1)).max()) if t > 0 else 0.0


def optimal(ts):
    r, H, u = SW.extremal(ts)
    return r, np.cross(H, r)


def hole(t, rng, G=480, L=1.7):
    """The unreachable region about |1> at time t: boundary in 3D, and its equivalent radius."""
    pts = np.vstack([samples(t, rng) for _ in range(2)])
    th = np.pi - np.arccos(np.clip(pts[:, 2], -1, 1))
    ph = np.arctan2(pts[:, 1], pts[:, 0])
    c = np.stack([th * np.cos(ph), th * np.sin(ph)], 1)[th < L]
    idx = np.clip(((c + L) / (2 * L) * G).astype(int), 0, G - 1)
    occ = np.zeros((G, G), bool)
    occ[idx[:, 1], idx[:, 0]] = True
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3)), iterations=2)
    lab, _ = ndimage.label(~occ)
    hl = lab == lab[G // 2, G // 2]
    assert not (hl[0].any() or hl[-1].any() or hl[:, 0].any() or hl[:, -1].any())   # inside the window
    g = np.linspace(-L, L, G)
    cs = plt.contour(g, g, hl.astype(float), levels=[0.5])
    seg = max(cs.allsegs[0], key=len)
    plt.close("all")
    rad = np.sqrt(hl.sum() / np.pi) * (2 * L / G)
    thb = np.pi - np.linalg.norm(seg, axis=1)
    phb = np.arctan2(seg[:, 1], seg[:, 0])
    return np.stack([np.sin(thb) * np.cos(phb), np.sin(thb) * np.sin(phb), np.cos(thb)], 1), rad


def optimal_paths(ts):
    """The four time-optimal paths (sign of the first bang, order of the bangs), at the times ts."""
    out = []
    for s0 in (-1, 1):
        for d in ((TO.T1, TO.T2), (TO.T2, TO.T1)):
            u = np.where(ts < d[0], s0 * U0, -s0 * U0)
            r = np.tile([0, 0, 1.0], (len(ts), 1))
            r = rotate(r, s0 * U0 + 0 * ts, np.minimum(ts, d[0]))
            r = rotate(r, -s0 * U0 + 0 * ts, np.maximum(ts - d[0], 0))
            out.append(r)
    return out


def beyond(t, rng, step):
    """Angular distance from the point a step beyond r(t) along p to the nearest sampled end point."""
    pts = np.vstack([samples(t, rng) for _ in range(2)])
    r, p = optimal(np.array([t]))
    q = np.cos(step) * r[0] + np.sin(step) * p[0] / np.linalg.norm(p[0])
    return float(np.min(np.arccos(np.clip(pts @ q, -1, 1))))


def checks(sets, holes, rng):
    vals = {0.5: 0.742, 1.0: 1.422, 2.0: 2.001, 3.0: 2.667, 3.3: 2.957}
    for t, v in vals.items():
        tm = theta_max(t)
        assert abs(tm - v) < 1.5e-3 and tm < min(W * t, np.pi) + 1e-9
    assert abs(theta_max(TSTAR) - np.pi) < 1e-3                       # the grid of extremals comes within 1e-4
    assert np.allclose(TO.bang_path(-1, (TO.T1, TO.T2))[1], [0, 0, -1], atol=1e-9)
    for r in optimal_paths(np.array([TSTAR])):                        # the four optimal paths end at |1>
        assert np.allclose(r[0], [0, 0, -1], atol=1e-9)
    # nested sets about |0>, and the optimal path on their boundaries with p outward
    for a, b in zip(TIMES[:-1], TIMES[1:]):
        oa, ob = sets[a][0], sets[b][0]
        assert (oa & ~ob).sum() < 0.002 * oa.sum()
    for t in TIMES:
        r, p = optimal(np.array([t]))
        assert np.min(np.linalg.norm(sets[t][1] - chart(r)[0], axis=1)) < 0.03
    for t in (1.0, 2.0):
        assert beyond(t, rng, 0.05) > 0.04 and beyond(t, rng, 0.1) > 0.08
    # the holes shrink, with size about pi - theta_max, and close at t*
    rads = [holes[t][1] for t in HOLES]
    assert all(a > b for a, b in zip(rads[:-1], rads[1:]))
    for t in (3.0, 3.3, 3.45):
        assert abs(holes[t][1] - (np.pi - theta_max(t))) < 0.07
    for t in (2.75, 3.0, 3.3):                                        # the optimal state on a corner of the hole
        bd = holes[t][0]
        r, p = optimal(np.array([t]))
        assert np.min(np.arccos(np.clip(bd @ r[0], -1, 1))) < 0.03
        rmax = (np.pi - np.arccos(np.clip(bd[:, 2], -1, 1))).max()
        assert abs((np.pi - np.arccos(r[0][2])) - rmax) < 0.03
        tp = np.array([0, 0, -1.0]) + r[0][2] * r[0]
        assert (p[0] / np.linalg.norm(p[0])) @ (tp / np.linalg.norm(tp)) > 0.7
    for r in optimal_paths(np.array([3.0, 3.3])):                     # the optimal states lie outside the holes
        for q, t in zip(r, (3.0, 3.3)):
            assert np.pi - np.arccos(np.clip(q[2], -1, 1)) > np.pi - theta_max(t)
    pts = np.vstack([samples(TSTAR, rng) for _ in range(2)])
    th, ph = np.arccos(np.clip(pts[:, 2], -1, 1)), np.arctan2(pts[:, 1], pts[:, 0])
    bins = np.digitize(ph, np.linspace(-np.pi, np.pi, 13)) - 1
    assert max(np.pi - th[bins == i].max() for i in range(12)) < 0.04   # at t*, |1> is closed in on all sides


def main():
    rng = np.random.default_rng(0)
    sets = {}
    for t in TIMES:
        occ = occupancy(samples(t, rng))
        sets[t] = (occ, boundary(occ))
    holes = {t: hole(t, rng) for t in HOLES}
    checks(sets, holes, rng)
    b = []
    washes = {0.5: "ndBlue!55", 1.0: "ndBlue!38", 2.0: "ndBlue!20"}
    labels = {0.5: "0.5", 1.0: "1", 2.0: "2"}
    tt = np.linspace(0, TSTAR, 700)
    ropt, popt = optimal(tt)

    # ---- (a) the map about |0>
    S = 1.05
    oa = np.array([2.45, 2.6])
    P = lambda c: oa + S * np.asarray(c)
    b.append("\\fill[ndPaper] %s circle (%.3f);" % (N.pt(oa), S * 2.25))
    b.append("\\draw[faint] %s circle (%.3f);" % (N.pt(oa), S * 2.25))
    b.append("\\draw[faint, densely dotted] %s circle (%.3f);" % (N.pt(oa), S * np.pi / 2))
    for t in (2.0, 1.0, 0.5):
        bd = sets[t][1]
        assert np.linalg.norm(bd, axis=1).max() < 2.25
        b.append("\\filldraw[fill=%s, draw=ndBlue!80!ndInk, line width=0.5pt] %s -- cycle;"
                 % (washes[t], " -- ".join(N.pt(P(q)) for q in bd[::2])))
    k23 = np.searchsorted(tt, 2.3)
    cp = chart(ropt)
    b.append(N.polyline([P(q) for q in cp[:k23 + 1]], "line width=1.5pt, ndRed"))
    b.append(N.polyline([P(q) for q in cp[k23:np.searchsorted(tt, 2.6)]], "line width=1.0pt, ndRed, dash pattern=on 2pt off 1.5pt"))
    for t in (1.0, 2.0):
        k = np.searchsorted(tt, t)
        r, ph = ropt[k], popt[k] / np.linalg.norm(popt[k])
        tip = chart(np.cos(0.35) * r + np.sin(0.35) * ph)
        b.append("\\draw[vec, ndGold!85!ndInk, line width=1.0pt] %s -- %s;" % (N.pt(P(cp[k])), N.pt(P(tip))))
        b.append("\\node[dot, minimum size=2.8pt] at %s {};" % N.pt(P(cp[k])))
    b.append("\\node[dot] at %s {};" % N.pt(oa))
    b.append("\\node[font=\\scriptsize, anchor=north west, inner sep=0.5pt] at %s {$\\ket0$};" % N.pt(oa + np.array([0.04, -0.04])))
    for t in (0.5, 1.0, 2.0):
        bd = sets[t][1]
        q = bd[np.argmin(bd[:, 0] - 0.3 * bd[:, 1])]
        b.append("\\node[font=\\scriptsize, fill=ndPaper, fill opacity=0.8, text opacity=1, inner sep=1pt, anchor=east] at %s {$\\Reach(%s)$};"
                 % (N.pt(P(q) + np.array([-0.05, 0.0])), labels[t]))
    b.append("\\node[note, anchor=north, text=ndInk!70] at %s {the equator};" % N.pt(oa + np.array([-0.55, -S * np.pi / 2 - 0.02])))
    k = np.searchsorted(tt, 2.0)
    b.append("\\node[note, anchor=north west, text=ndGold!55!ndInk] (pn) at %s {adjoint $\\vec p$:\\\\ outward normals};" % N.pt(oa + np.array([1.05, -1.25])))
    tip2 = chart(np.cos(0.25) * ropt[k] + np.sin(0.25) * popt[k] / np.linalg.norm(popt[k]))
    b.append("\\draw[pointer] (pn.north west) to[bend left=15] %s;" % N.pt(P(tip2) + np.array([0.04, -0.03])))
    b.append("\\node[note, anchor=south west, text=ndRed] (op) at %s {optimal path, on the front\\\\ of $\\Reach(t)$ until $t\\approx2.3$};" % N.pt(oa + np.array([0.6, 1.75])))
    b.append("\\draw[pointer] (op.south west) to[bend right=15] %s;" % N.pt(P(cp[np.searchsorted(tt, 1.5)]) + np.array([0.03, 0.04])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) growth: the map about $\\ket0$};" % N.pt(oa + np.array([0.0, -2.82])))

    # ---- (b) from below |1>
    V2 = N.View(az=20.0, el=-88.0, R=2.0, shift=(9.3, 2.6))
    o2 = V2.P((0, 0, 0))
    b.append("\\fill[ndBlue!22] %s circle (%.3f);" % (N.pt(o2), V2.R))
    b += V2.sphere(shade=False)
    fills = {2.75: "ndPaper", 3.0: "ndRose!40", 3.3: "ndRose!75", 3.45: "ndRed!70"}
    for t in HOLES:
        bd = holes[t][0]
        assert np.all(bd @ V2.d >= 0)
        b.append("\\filldraw[fill=%s, draw=ndInk!80, line width=0.5pt] %s -- cycle;" % (fills[t], " -- ".join(V2.pt(p) for p in bd[::2])))
    kk = np.searchsorted(tt, 3.0)
    for j, r in enumerate(optimal_paths(tt)):
        sty = "line width=1.3pt, ndBlue!80!ndInk" if j == 0 else "line width=0.8pt, ndBlue!60"
        b += V2.curve(r[kk:], sty, None)
        b.append("\\fill[%s] %s circle (1.3pt);" % ("ndBlue!80!ndInk" if j == 0 else "ndBlue!60", V2.pt(r[kk])))
    b.append("\\node[dot, fill=ndInk, minimum size=3.2pt] at %s {};" % V2.pt((0, 0, -1)))
    b.append("\\node[font=\\scriptsize, anchor=north west] at %s {$\\ket1$};" % N.pt(V2.P((0, 0, -1)) + np.array([0.05, -0.03])))
    tag = {2.75: "2.75", 3.0: "3", 3.3: "3.3", 3.45: "3.45"}
    for t, pos in [(2.75, (2.25, 1.35)), (3.0, (2.25, 0.55)), (3.3, (2.25, -0.25)), (3.45, (2.25, -1.05))]:
        bd = holes[t][0]
        tgt = bd[np.argmax([V2.P(q)[0] for q in bd])]
        b.append("\\node[font=\\scriptsize, anchor=west] (h%d) at %s {$t=%s$};" % (int(100 * t), N.pt(o2 + np.array(pos)), tag[t]))
        b.append("\\draw[pointer] (h%d.west) to[bend left=8] %s;" % (int(100 * t), N.pt(V2.P(tgt) + np.array([0.04, 0.0]))))
    b.append("\\node[note, anchor=north] at %s {the hole about $\\ket1$ shrinks, and closes at\\\\ $t^\\star=%.3f$; the optimal paths ride its corners};" % (N.pt(o2 + np.array([0.0, -2.02])), TSTAR))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) first contact, seen from below $\\ket1$};" % N.pt(o2 + np.array([0.0, -2.82])))

    # ---- (c) the farthest reach
    tg = np.linspace(0, TSTAR, 30)
    tm = np.array([theta_max(t) for t in tg])
    assert np.all(np.diff(tm) >= -1e-9) and np.all(tm <= np.minimum(W * tg, np.pi) + 1e-9)
    pc = N.Plot((0, 3.7), (0, 3.4), 3.6, 3.6, origin=(14.0, 0.75))
    b += pc.axes(xlabel="{\\scriptsize $t$}", ylabel="{\\scriptsize $\\theta_{\\max}(t)$}",
                 xticks=[(np.pi / U0, "$\\frac{\\pi}{u_0}$"), (TSTAR, "$t^\\star$")], yticks=[(np.pi / 2, "$\\frac\\pi2$"), (np.pi, "$\\pi$")])
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (pc.pt(0, np.pi), pc.pt(3.7, np.pi)))
    b.append(pc.curve([0, np.pi / W], [0, np.pi], "line width=0.9pt, ndGold, dash pattern=on 3pt off 2pt"))
    b.append(pc.curve([0, np.pi / U0], [0, np.pi], "line width=0.9pt, ndGreen, dash pattern=on 1.5pt off 1.5pt"))
    b.append(pc.curve(tg, tm, "line width=1.5pt, ndBlue"))
    b.append("\\node[dot, fill=ndRed] at %s {};" % pc.pt(TSTAR, np.pi))
    b.append("\\node[note, anchor=east, text=ndGold!55!ndInk] at %s {$\\Omega t$};" % pc.pt(1.05, 2.75))
    b.append("\\node[note, anchor=west, text=ndGreen!80!ndInk] (nd) at %s {no drift: $u_0t$};" % pc.pt(1.75, 2.45))
    b.append("\\draw[pointer] (nd.west) to[bend right=10] %s;" % pc.pt(1.55, U0 * 1.55 - 0.05))
    b.append("\\node[note, anchor=north west, text=ndBlue] at %s {with drift};" % pc.pt(1.75, 1.75))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the farthest reach};" % pc.pt(1.85, -0.98))
    N.write("ch09_reachqubit", b)
    return {t: theta_max(t) for t in (0.5, 1.0, 2.0, 3.0, 3.3, TSTAR)}, {t: holes[t][1] for t in HOLES}


if __name__ == "__main__":
    vals, rads = main()
    print("theta_max: " + ", ".join("%.3f -> %.4f" % kv for kv in vals.items()))
    print("hole radius: " + ", ".join("%.2f -> %.3f" % kv for kv in rads.items()))
    print("written figures/ch09_reachqubit.tex")
