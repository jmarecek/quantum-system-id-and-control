"""ch10_penalised.py -- penalised geodesics of SU(2) converge to a sub-Riemannian one.

The example of ch10_su2.py: a qubit without drift, the field in the xy-plane
cheap and sigma_z penalised by q, g_q(A, A) = w_x^2 + w_y^2 + q w_z^2, target
V = exp(-i (pi/2) sigma_z/2), time [0, 1].  The minimising g_q-geodesics are
known in closed form: for q <= 4 the direct route exp(-i t phi sigma_z/2); for
q > 4 a circle of twisted routes
    U(t) = exp(-t (q-1) A_k) exp(t (A_p + q A_k)),  w_z = (2 pi - phi)/(q - 1),
of which we draw the one whose field starts along x.

(a) The geodesics drawn by their rotation vectors theta n, U = exp(-i theta n.sigma/2):
    the direct route up the z-axis (gold, all q <= 4), the twisted routes for
    q = 4.5, 6, 8, 16, 64 (blue, darker for larger q), and their limit, the
    sub-Riemannian geodesic of Figure ch09:subriemann (red), which they
    approach uniformly.
(b) The control: the field in the plane, (w_x, w_y)(t), turns clockwise at
    the same rate 2 pi - phi = 3 pi/2 for every q > 4, along arcs of radius
    w_p = 1.66, 3.24, 4.07 for q = 4.5, 8, 64, tending to d_inf = 4.156, while
    the penalised part w_z = 1.35, 0.67, 0.075 tends to 0.
(c) How fast: the uniform distance sup_t d(gamma_q(t), gamma(t)) (the rotation
    angle of gamma_q gamma^{-1}) and the L^2 distance of the controls both
    decay like 1/q: q times them tends to 2.27 and 7.13.
Writes figures/ch10_penalised.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N
import ch10_su2 as S

QS = [4.5, 6.0, 8.0, 16.0, 64.0]
TS = np.linspace(0, 1, 241)


def limit_curve(ts=TS):
    """The sub-Riemannian geodesic: U(t) = exp(-t C) exp(t (A_p + C)), C = gen(0,0,2pi-phi), A_p = gen(d_inf,0,0)."""
    C = S.gen([0, 0, 2 * np.pi - S.PHI])
    Ap = S.gen([S.d_inf(), 0, 0])
    return np.array([expm(-t * C) @ expm(t * (Ap + C)) for t in ts])


def limit_field(ts=TS):
    a = -(2 * np.pi - S.PHI) * ts
    return np.stack([S.d_inf() * np.cos(a), S.d_inf() * np.sin(a), 0 * ts], 1)


def angle(U, W):
    """The bi-invariant distance between U and W: the rotation angle of U W^{-1}."""
    w = np.clip(np.real(np.trace(U @ W.conj().T)) / 2, -1, 1)
    return 2 * np.arccos(w)


def distances(q, ts=TS):
    wp, wz = S.minimiser(q)
    U = S.geodesic(q, wp, wz, ts)
    sup = max(angle(a, b) for a, b in zip(U, limit_curve(ts)))
    l2 = np.sqrt(np.trapz(np.sum((S.field(q, wp, wz, ts) - limit_field(ts)) ** 2, 1), ts))
    return sup, l2


def checks():
    S.checks()
    # the limit is the sub-Riemannian geodesic of Chapter 9, reparametrised to unit time
    import ch09_subriemann as C9
    t9, U9, T9, c9 = C9.geodesic(S.PHI, 0.0, n=241)
    assert max(np.linalg.norm(a - b) for a, b in zip(limit_curve(), U9)) < 1e-12
    assert abs(T9 - S.d_inf()) < 1e-12 and abs(c9 * T9 - (2 * np.pi - S.PHI)) < 1e-12
    # the twisted routes start along x and their field turns at the rate 3 pi/2
    for q in QS:
        wp, wz = S.minimiser(q)
        f = S.field(q, wp, wz, TS)
        assert abs(f[0, 1]) < 1e-12 and abs(np.arctan2(f[-1, 1], f[-1, 0]) - (-(2 * np.pi - S.PHI) + 2 * np.pi)) < 1e-9
    assert np.allclose([S.minimiser(q)[0] for q in (4.5, 8, 64)], [1.66, 3.24, 4.07], atol=0.01)
    assert np.allclose([S.minimiser(q)[1] for q in (4.5, 8, 64)], [1.35, 0.67, 0.075], atol=0.005)
    # both distances decay like 1/q
    qd = [(q, *distances(q, np.linspace(0, 1, 401))) for q in (256.0, 1024.0, 4096.0)]
    for q, sup, l2 in qd:
        assert abs(q * sup - 2.27) < 0.02 and abs(q * l2 - 7.13) < 0.05
    sups = [distances(q)[0] for q in QS]
    assert np.all(np.diff(sups) < 0)


def main():
    checks()
    b = []
    # ---- (a) the geodesics in rotation-vector coordinates
    V = N.View(az=25.0, el=22.0, R=1.25, shift=(2.6, 2.2))
    o = V.P((0, 0, 0))
    for e, lab, off in [((1.75, 0, 0), "x", (-0.1, -0.12)), ((0, 1.75, 0), "y", (0.12, -0.03)), ((0, 0, 2.2), "z", (0.0, 0.12))]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (V.pt((0, 0, 0)), V.pt(e)))
        b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%s$};" % (N.pt(V.P(e) + np.array(off)), lab))
    sq = [(-0.55, -0.55, 0), (0.55, -0.55, 0), (0.55, 0.55, 0), (-0.55, 0.55, 0)]
    b.append("\\fill[ndSky, fill opacity=0.7] %s -- cycle;" % " -- ".join(V.pt(p) for p in sq))
    b.append("\\draw[line width=1.4pt, ndGold] %s -- %s;" % (V.pt((0, 0, 0)), V.pt((0, 0, S.PHI))))
    shades = [30, 45, 60, 75, 95]
    for q, sh in zip(QS, shades):
        wp, wz = S.minimiser(q)
        cv = np.array([S.rotvec(U) for U in S.geodesic(q, wp, wz, TS)])
        b.append(N.polyline([V.P(p) for p in cv], "line width=0.8pt, ndBlue!%d" % sh))
    lim = np.array([S.rotvec(U) for U in limit_curve()])
    b.append(N.polyline([V.P(p) for p in lim], "line width=1.5pt, ndRed"))
    b.append("\\node[dot] at %s {};" % V.pt((0, 0, 0)))
    b.append("\\node[dot, fill=ndRed] at %s {};" % V.pt((0, 0, S.PHI)))
    b.append("\\node[font=\\scriptsize, anchor=north east] at %s {$\\Id$};" % N.pt(o + np.array([-0.05, -0.04])))
    b.append("\\node[font=\\scriptsize, anchor=south west] at %s {$V$};" % N.pt(V.P((0, 0, S.PHI)) + np.array([0.06, 0.02])))
    b.append("\\node[note, anchor=west, text=ndGold!55!ndInk] (dr) at %s {$q\\le4$: the\\\\ direct route};" % N.pt(o + np.array([1.55, 1.35])))
    b.append("\\draw[pointer] (dr.west) to[bend right=10] %s;" % N.pt(V.P((0, 0, 0.75 * S.PHI)) + np.array([0.06, 0.0])))
    k = 130
    wp, wz = S.minimiser(4.5)
    p45 = S.rotvec(S.geodesic(4.5, wp, wz, [TS[k]])[0])
    b.append("\\node[note, anchor=east, text=ndBlue] (tw) at %s {$q=4.5,6,8,16,64$:\\\\ twisted routes};" % N.pt(o + np.array([-1.45, 1.75])))
    b.append("\\draw[pointer] (tw.east) to[bend left=10] %s;" % N.pt(V.P(p45) + np.array([-0.04, 0.02])))
    pl = lim[95]
    b.append("\\node[note, anchor=east, text=ndRed] (sr) at %s {$q=\\infty$: the\\\\ sub-Riemannian\\\\ geodesic};" % N.pt(o + np.array([-1.5, -0.35])))
    b.append("\\draw[pointer] (sr.east) to[bend right=10] %s;" % N.pt(V.P(pl) + np.array([-0.05, 0.0])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the minimising $g_q$-geodesics};" % N.pt(o + np.array([0.1, -1.25])))

    # ---- (b) the field in the plane
    cc = np.array([8.2, 2.35])
    R = 0.38                                                       # cm per unit of w
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$w_x$};" % (N.pt(cc + [-1.85, 0]), N.pt(cc + [1.9, 0])))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$w_y$};" % (N.pt(cc + [0, -1.85]), N.pt(cc + [0, 1.9])))
    for q, col, lab in [(4.5, "ndBlue!30", "4.5"), (8.0, "ndBlue!60", "8"), (64.0, "ndBlue!95", "64")]:
        wp, wz = S.minimiser(q)
        f = S.field(q, wp, wz, TS)
        b.append(N.polyline([cc + R * v[:2] for v in f], "line width=1.0pt, %s" % col))
        b.append("\\node[dot, minimum size=2.4pt, fill=%s] at %s {};" % (col, N.pt(cc + R * f[-1, :2])))
    fl = limit_field()
    b.append(N.polyline([cc + R * v[:2] for v in fl], "line width=1.4pt, ndRed, dash pattern=on 4pt off 1.5pt"))
    end = cc + R * fl[-1, :2]
    tang = R * (fl[-1, :2] - fl[-3, :2])
    b.append("\\draw[->, >=stealth, line width=1.4pt, ndRed] %s -- %s;" % (N.pt(end - 0.02 * tang / np.linalg.norm(tang)), N.pt(end)))
    b.append("\\node[dot] at %s {};" % N.pt(cc + R * fl[0, :2]))
    b.append("\\node[font=\\scriptsize, anchor=south west] at %s {$t=0$};" % N.pt(cc + R * fl[0, :2] + [0.03, 0.03]))
    b.append("\\node[note, anchor=north west] at %s {%s};" % (N.pt(cc + [-1.95, -1.95]),
             "each turns through $\\tfrac{3\\pi}2$; radius $w_p$,\\\\ penalised part $w_z$:\\\\"
             "\\textcolor{ndBlue!45}{$q=4.5$: $w_p=1.66$, $w_z=1.35$}\\\\"
             "\\textcolor{ndBlue!70}{$q=8$: $w_p=3.24$, $w_z=0.67$}\\\\"
             "\\textcolor{ndBlue}{$q=64$: $w_p=4.07$, $w_z=0.075$}\\\\"
             "\\textcolor{ndRed}{$q=\\infty$: $w_p=4.156$, $w_z=0$}"))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the control};" % N.pt(cc + [0, -4.4]))

    # ---- (c) the rate of convergence
    qs = np.geomspace(4.5, 4096, 40)
    ds = np.array([distances(q) for q in qs])
    pc = N.Plot((np.log10(4), np.log10(5000)), (-3.5, 1.3), 4.0, 3.5, origin=(11.95, 0.75))
    b += pc.axes(xlabel="{\\scriptsize $q$}", xticks=[(1, "$10$"), (2, "$10^2$"), (3, "$10^3$")],
                 yticks=[(-3, "$10^{-3}$"), (-2, "$10^{-2}$"), (-1, "$10^{-1}$"), (0, "$1$"), (1, "$10$")])
    b.append(pc.curve(np.log10(qs), np.log10(ds[:, 1]), "line width=1.2pt, ndBlue"))
    b.append(pc.curve(np.log10(qs), np.log10(ds[:, 0]), "line width=1.2pt, ndRed"))
    b.append(pc.curve(np.log10([30, 3000]), np.log10([7.13 / 30, 7.13 / 3000]) + 0.35, "dashedink"))
    b.append("\\node[note, anchor=south west] at %s {slope $-1$};" % pc.pt(np.log10(300), np.log10(7.13 / 300) + 0.42))
    b.append("\\node[note, anchor=south west, text=ndBlue] at %s {controls, $L^2$};" % pc.pt(np.log10(9), np.log10(distances(9)[1]) + 0.05))
    b.append("\\node[note, anchor=north east, text=ndRed] at %s {curves,\\\\ uniform};" % pc.pt(np.log10(60), np.log10(distances(60)[0]) - 0.1))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) distance from the limit};" % pc.pt(np.log10(140), -4.5))
    N.write("ch10_penalised", b)


if __name__ == "__main__":
    main()
    for q in (4.5, 8, 64, 1024):
        print(q, "sup, L2 =", ["%.4g" % v for v in distances(q)])
    print("written figures/ch10_penalised.tex")
