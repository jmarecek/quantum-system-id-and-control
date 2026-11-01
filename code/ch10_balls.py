"""ch10_balls.py -- penalised metrics flatten, and their distances increase to the sub-Riemannian one.

The example of ch10_su2.py: su(2) with the coordinates (w_x, w_y, w_z) of
A = -(i/2) w.sigma, and the penalised metric g_q(A, A) = w_x^2 + w_y^2 + q w_z^2.

(a) The unit balls of g_q in su(2): the ellipsoids w_x^2 + w_y^2 + q w_z^2 <= 1,
    with the semi-axis 1/sqrt(q) along z, for q = 1 (the round ball of the
    bi-invariant metric), 4 and 16, flattening to the disc of the available
    directions (q = infinity), on which the sub-Riemannian metric lives.
(b) The distance d_q(I, V) from I to V = exp(-i phi sigma_z/2) against q, on a
    logarithmic scale, for phi = pi/4, pi/2 and pi.  It is sqrt(q) phi, along
    the direct route, up to q = 2 pi/phi (= 8, 4, 2), where a circle of twisted
    geodesics branches off; thereafter it bends over and increases to the
    sub-Riemannian distance sqrt(phi (4 pi - phi)) (= 3.042, 4.156, 5.441).
    d_q^2 and its derivative are continuous at the branch point.  For
    phi = pi/2 the continuation of the direct route (dashed) and the limit
    (dotted) would cross at q = 7.0, but the minimiser has left the direct
    route at q = 4.
Writes figures/ch10_balls.tex.
"""
import numpy as np
import needham as N
import ch10_su2 as S

PHIS = [(np.pi / 4, "\\frac\\pi4", "ndGreen"), (np.pi / 2, "\\frac\\pi2", "ndBlue"), (np.pi, "\\pi", "ndRed")]


def checks():
    S.checks()
    for phi, _, _ in PHIS:
        qb = S.q_branch(phi)
        assert abs(S.length(qb, phi) - np.sqrt(qb) * phi) < 1e-12
        assert abs(S.length(1e9, phi) - S.d_inf(phi)) < 1e-6
        for q in np.geomspace(1.01, 1000, 200):
            Ls = [L for _, _, L in S.all_geodesic_lengths(q, phi)]
            assert abs(min(Ls) - S.length(q, phi)) < 1e-9
        h = 1e-6
        l2 = lambda q: S.length(q, phi) ** 2
        assert abs((l2(qb) - l2(qb - h)) - (l2(qb + h) - l2(qb))) < 1e-9
    assert np.allclose([S.q_branch(p) for p, _, _ in PHIS], [8, 4, 2])
    assert np.allclose([S.d_inf(p) for p, _, _ in PHIS], [3.042, 4.156, 5.441], atol=1e-3)
    # the unit ball of g_q has the semi-axis 1/sqrt(q) along z: g_q(e_z/sqrt(q), e_z/sqrt(q)) = 1
    for q in (1, 4, 16):
        assert abs(q * (1 / np.sqrt(q)) ** 2 - 1) < 1e-15


def main():
    checks()
    b = []
    # ---- (a) the unit balls
    V = N.View(az=25.0, el=18.0, R=1.75, shift=(2.4, 2.45))
    o = V.P((0, 0, 0))
    for q, fill, draw in [(1, "ndSky!45", "ink"), (4, "ndSky!80", "line width=0.6pt, ndBlue!60"), (16, "ndBlue!30", "line width=0.6pt, ndBlue")]:
        poly = N.ellipsoid_outline(V, (0, 0, 0), (1, 1, 1 / np.sqrt(q)), n=80)
        b.append("\\filldraw[fill=%s, fill opacity=0.6, draw=none] %s -- cycle;" % (fill, " -- ".join(N.pt(p) for p in poly)))
        b.append("\\draw[%s] %s -- cycle;" % (draw, " -- ".join(N.pt(p) for p in poly)))
    eq = np.array([[np.cos(a), np.sin(a), 0] for a in np.linspace(0, 2 * np.pi, 241)])
    b.append("\\filldraw[fill=ndRose, fill opacity=0.7, draw=ndRed, line width=1.1pt] %s -- cycle;" % " -- ".join(V.pt(p) for p in eq))
    for e, lab, off in [((2.6, 0, 0), "w_x", (-0.14, -0.1)), ((0, 1.45, 0), "w_y", (0.16, -0.02)), ((0, 0, 1.4), "w_z", (0.0, 0.14))]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (V.pt((0, 0, 0)), V.pt(e)))
        b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%s$};" % (N.pt(V.P(e) + np.array(off)), lab))
    for q in (1, 4, 16):
        b.append("\\node[dot, minimum size=2.6pt] at %s {};" % V.pt((0, 0, 1 / np.sqrt(q))))
    b.append("\\node[font=\\scriptsize, anchor=west] at %s {$1$};" % N.pt(V.P((0, 0, 1)) + np.array([0.06, 0.08])))
    b.append("\\node[font=\\scriptsize, anchor=west] at %s {$\\tfrac12$};" % N.pt(V.P((0, 0, 0.5)) + np.array([0.06, 0.05])))
    b.append("\\node[font=\\scriptsize, anchor=west] at %s {$\\tfrac14$};" % N.pt(V.P((0, 0, 0.25)) + np.array([0.06, 0.05])))
    b.append("\\node[note, anchor=west] (q1) at %s {$q=1$: the round ball};" % N.pt(o + np.array([1.95, 1.25])))
    b.append("\\draw[pointer] (q1.west) to[bend right=10] %s;" % N.pt(V.P((0.45, 0.45, 0.77)) + np.array([0.03, 0.03])))
    b.append("\\node[note, anchor=west, text=ndBlue] (q4) at %s {$q=4$, $16$: semi-axis\\\\ $1/\\sqrt q$ along $w_z$};" % N.pt(o + np.array([1.95, 0.35])))
    b.append("\\draw[pointer] (q4.west) to[bend right=10] %s;" % N.pt(V.P((0.6, 0.7, 0.2)) + np.array([0.03, 0.0])))
    b.append("\\node[note, anchor=west, text=ndRed] (qi) at %s {$q=\\infty$: the disc of\\\\ the available directions};" % N.pt(o + np.array([1.95, -0.85])))
    b.append("\\draw[pointer] (qi.west) to[bend left=10] %s;" % N.pt(V.P((0.55, 0.835, 0)) + np.array([0.03, -0.03])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the unit balls of $g_q$};" % N.pt(o + np.array([0.6, -2.05])))

    # ---- (b) the distance against q
    pl = N.Plot((0, 3), (0, 6.2), 5.4, 3.9, origin=(9.4, 0.55))
    b += pl.axes(xlabel="{\\scriptsize $q$}", ylabel="{\\scriptsize $d_q(\\Id,V)$}",
                 xticks=[(0, "$1$"), (np.log10(4), "$4$"), (1, "$10$"), (2, "$10^2$"), (3, "$10^3$")],
                 yticks=[(np.pi / 2, "$\\frac\\pi2$"), (np.pi, "$\\pi$"), (3 * np.pi / 2, "$\\frac{3\\pi}2$")])
    qs = np.geomspace(1, 1000, 400)
    for phi, lab, col in PHIS:
        di = S.d_inf(phi)
        b.append(pl.curve([0, 3], [di, di], "line width=0.5pt, %s, densely dotted" % col))
        qb = S.q_branch(phi)
        qd = qs[(qs >= qb) & (np.sqrt(qs) * phi <= 6.1)]
        b.append(pl.curve(np.log10(qd), np.sqrt(qd) * phi, "line width=0.6pt, %s, dash pattern=on 3pt off 2pt" % col))
        b.append(pl.curve(np.log10(qs), [S.length(q, phi) for q in qs], "line width=1.3pt, %s" % col))
        b.append("\\node[dot, minimum size=3pt, fill=%s] at %s {};" % (col, pl.pt(np.log10(qb), S.length(qb, phi))))
        b.append("\\node[font=\\scriptsize, anchor=west, text=%s] at %s {$\\varphi=%s$};" % (col, pl.pt(3.02, di), lab))
    qc = (S.d_inf() / S.PHI) ** 2
    b.append("\\draw[ndBlue, line width=0.6pt] %s circle (2pt);" % pl.pt(np.log10(qc), S.d_inf()))
    b.append("\\node[note, anchor=north west, text=ndBlue] (br) at %s {at $q=2\\pi/\\varphi$ a circle of\\\\ twisted geodesics branches off\\\\ (dots); for $\\varphi=\\frac\\pi2$, at $q=4$};" % pl.pt(1.12, 2.55))
    b.append("\\draw[pointer] (br.north west) to[bend left=15] %s;" % pl.pt(np.log10(4) + 0.03, S.length(4) - 0.08))
    b.append("\\node[note, anchor=south, text=ndBlue] (cr) at %s {not at the crossing, $q=7$};" % pl.pt(2.05, 5.62))
    b.append("\\draw[pointer] (cr.south west) to[bend right=20] %s;" % pl.pt(np.log10(qc) + 0.04, S.d_inf() + 0.1))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the penalised distance};" % pl.pt(1.5, -1.1))
    N.write("ch10_balls", b)


if __name__ == "__main__":
    main()
    for phi, lab, _ in PHIS:
        print("phi=%s: branch at q=%.2f, d_inf=%.4f" % (lab, S.q_branch(phi), S.d_inf(phi)))
    print("written figures/ch10_balls.tex")
