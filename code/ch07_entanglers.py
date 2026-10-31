"""ch07_entanglers.py -- the perfect entanglers in the Weyl chamber.

A two-qubit gate is a perfect entangler if it turns some product state into a
maximally entangled one.  By the criterion of Makhlin, proved by Zhang, Vala,
Sastry and Whaley (PRA 67, 042313, 2003), this happens if and only if the convex
hull of the eigenvalues of m(U) contains 0.  In the Weyl chamber O A1 A2 A3 the
perfect entanglers fill the polyhedron L M N P Q A2, which is what remains when
the three corner tetrahedra
    L Q P O   (c1 + c2 <= pi/2),   N P A2 A3   (c2 + c3 >= pi/2),   L M N A1   (c1 - c2 >= pi/2)
are cut away; its volume is pi^3/48, half that of the chamber.  Under the Haar
measure on the gates they are about 85% (Watts, O'Connor and Vala, Entropy 15,
1963 (2013)), since the measure is largest in the middle of the chamber.
The script checks the criterion against a direct maximisation of the
concurrence over product inputs for three gates, the volumes, and the Haar
fraction.  Writes figures/ch07_entanglers.tex.
"""
import numpy as np
from scipy.optimize import minimize
import needham as N
import ch07_weyl as W

PI = np.pi
EXAMPLES = [("1", (PI / 4, 0, 0)), ("2", (PI / 2, 0, 0)), ("3", (PI / 2, PI / 4, 0))]


def product(x):
    th1, ph1, th2, ph2 = x
    a = np.array([np.cos(th1 / 2), np.exp(1j * ph1) * np.sin(th1 / 2)])
    b = np.array([np.cos(th2 / 2), np.exp(1j * ph2) * np.sin(th2 / 2)])
    return np.kron(a, b)


def concurrence(psi):
    return 2 * abs(psi[0] * psi[3] - psi[1] * psi[2])


def max_concurrence(U, rng, starts=40):
    best = 0.0
    for _ in range(starts):
        r = minimize(lambda x: -concurrence(U @ product(x)), rng.uniform(0, 2 * PI, 4), method="Nelder-Mead",
                     options=dict(xatol=1e-10, fatol=1e-12, maxiter=4000))
        best = max(best, -r.fun)
    return best


def check():
    W.checks()
    rng = np.random.default_rng(11)
    out = {}
    for name, c in EXAMPLES:
        cmax = max_concurrence(W.nonlocal_gate(c), rng)
        pe = W.hull_contains_zero(W.m_phases(c))
        assert pe == W.perfect_entangler_c(c)
        assert (cmax > 1 - 1e-6) == pe
        out[name] = cmax
    assert abs(out["1"] - np.sin(PI / 4)) < 1e-6                 # [gamma,0,0] reaches at most sin(gamma)
    # the polyhedron L M N P Q A2: the vertices satisfy the three inequalities, with equality on the cuts
    for k in ("L", "M", "N", "P", "Q", "A2"):
        assert W.perfect_entangler_c(W.VERTICES[k]) and W.in_chamber(W.VERTICES[k])
    # the controlled-U edge O L meets the perfect entanglers only at L
    for g in np.linspace(0, PI / 2, 101)[:-1]:
        assert not W.perfect_entangler_c((g, 0, 0))
    # the Haar measure: about 85% of all gates
    haar = np.mean([W.perfect_entangler(W.haar(rng, 4)) for _ in range(20000)])
    assert abs(haar - 0.849) < 0.02
    print("max concurrence from product states: [pi/4,0,0] %.4f, CNOT %.6f, [pi/2,pi/4,0] %.6f; Haar fraction %.3f"
          % (out["1"], out["2"], out["3"], haar))
    return out, haar


def main():
    cmax, haar = check()
    V0 = {k: W.unit(v) for k, v in W.VERTICES.items()}
    ch = [V0["O"], V0["A1"], V0["A2"], V0["A3"]]
    pe = [V0[k] for k in ("L", "M", "N", "P", "Q", "A2")]
    corners = [[V0[k] for k in t] for t in (("L", "Q", "P", "O"), ("N", "P", "A2", "A3"), ("L", "M", "N", "A1"))]
    b = []
    # ---- (a) the polyhedron of perfect entanglers in the chamber
    Va = N.View(az=-75.0, el=22.0, R=5.2, shift=(0.0, 0.0))
    for t in corners:
        b += W.draw_polytope(Va, t, front=None, back=None, fill="ndRose, fill opacity=0.75")
    b += W.draw_polytope(Va, pe, front="line width=1.0pt, ndBlue", back="line width=0.5pt, ndBlue!70, densely dashed",
                         fill="ndSky, fill opacity=0.85")
    b += W.draw_polytope(Va, ch, front="ink", back="dashedink")
    b.append("\\draw[line width=1.6pt, ndBlue!45] %s -- %s;" % (Va.pt(V0["O"]), Va.pt(V0["L"])))
    P = {k: Va.P(v) for k, v in V0.items()}
    for k in V0:
        b.append("\\node[dot, fill=%s, minimum size=3.2pt] at %s {};" % ("ndBlue" if k in ("L", "M", "N", "P", "Q", "A2") else "ndInk", N.pt(P[k])))
    labels = {"O": ("O", (-0.1, 0.0, "east")), "A1": ("A_1", (0.1, 0.0, "west")), "A3": ("A_3", (0.0, 0.1, "south")),
              "L": ("L:\\ \\mathrm{CNOT}", (0.0, -0.12, "north")), "P": ("P:\\ \\sqrt{\\mathrm{SWAP}}", (-0.12, 0.05, "south east")),
              "N": ("N:\\ \\sqrt{\\mathrm{SWAP}}^{\\,\\dagger}", (0.12, 0.05, "west"))}
    for k, (lab, (dx, dy, anc)) in labels.items():
        b.append("\\node[font=\\scriptsize, anchor=%s] at %s {$%s$};" % (anc, N.pt(P[k] + np.array([dx, dy])), lab))
    for k, lab, pos, anc in [("Q", "Q:\\ \\sqrt{\\mathrm{iSWAP}}", (-0.6, 1.05), "east"), ("A2", "A_2:\\ \\mathrm{iSWAP}", (6.35, 0.8), "west"),
                             ("M", "M", (6.35, 0.15), "west")]:
        b.append("\\node[font=\\scriptsize, anchor=%s] (l%s) at (%.2f,%.2f) {$%s$};" % (anc, k, pos[0], pos[1], lab))
        side = "east" if anc == "east" else "west"
        b.append("\\draw[pointer] (l%s.%s) to[bend %s=10] %s;" % (k, side, "left" if anc == "east" else "right",
                                                                    N.pt(P[k] + np.array([0.06 if anc == "east" else -0.06, 0.0]))))
    # the three example gates of panel (b), marked in the chamber
    for name, c in EXAMPLES:
        q = Va.P(W.unit(c))
        if name == "2":
            continue                                                 # CNOT is the point L
        b.append("\\node[circle, draw=ndRed, fill=white, line width=0.8pt, inner sep=0pt, minimum size=4pt] at %s {};" % N.pt(q))
        b.append("\\node[font=\\scriptsize\\bfseries, text=ndRed, anchor=south] at %s {%s};" % (N.pt(q + np.array([0.0, 0.06])), name))
    b.append("\\node[font=\\scriptsize\\bfseries, text=ndRed, anchor=south east] at %s {2};" % N.pt(P["L"] + np.array([-0.1, 0.02])))
    # notes
    b.append("\\node[note, anchor=south west, text=ndBlue] (pe) at %s {perfect entanglers:\\\\ half the chamber,\\\\ $\\pi^3/48$ of $\\pi^3/24$};"
             % N.pt(P["A3"] + np.array([0.5, -0.35])))
    b.append("\\draw[pointer] (pe.south west) to[bend right=15] %s;" % N.pt(Va.P(np.mean(pe, axis=0)) + np.array([0.25, 0.25])))
    b.append("\\node[note, anchor=north, text=ndRed!80!ndInk] (nc) at %s {not perfect entanglers:\\\\ the three corners};"
             % N.pt(P["A1"] + np.array([0.4, -0.45])))
    b.append("\\draw[pointer] (nc.north) to[bend right=10] %s;" % N.pt(Va.P(np.mean(corners[2], axis=0)) + np.array([0.05, -0.02])))
    b.append("\\node[note, anchor=north, text=ndBlue] (cu) at %s {of the controlled-$U$ gates\\\\ only CNOT};"
             % N.pt(Va.P(0.2 * V0["L"]) + np.array([-0.4, -0.55])))
    b.append("\\draw[pointer] (cu.north) -- %s;" % N.pt(Va.P(0.25 * V0["L"]) + np.array([0.0, -0.06])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the perfect entanglers in the Weyl chamber};"
             % N.pt(np.array([P["L"][0], P["A1"][1] - 1.45])))

    # ---- (b) Makhlin's criterion: the eigenvalues of m(U) and their convex hull
    r, x0, y0, dx = 0.72, 9.5, 2.15, 2.25
    verdicts = {"1": "$0$ outside:\\\\ not a perfect\\\\ entangler", "2": "$0$ on the hull:\\\\ CNOT, a perfect\\\\ entangler",
                "3": "$0$ inside:\\\\ a perfect\\\\ entangler"}
    coords = {"1": "[\\tfrac\\pi4,0,0]", "2": "[\\tfrac\\pi2,0,0]", "3": "[\\tfrac\\pi2,\\tfrac\\pi4,0]"}
    for k, (name, c) in enumerate(EXAMPLES):
        cx = x0 + k * dx
        z = W.m_phases(c)
        inside = W.hull_contains_zero(z)
        b.append("\\draw[faint] (%.3f,%.3f) circle (%.3f);" % (cx, y0, r))
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (cx - r - 0.12, y0, cx + r + 0.12, y0))
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (cx, y0 - r - 0.12, cx, y0 + r + 0.12))
        pts = sorted({(round(w.real, 9), round(w.imag, 9)) for w in z}, key=lambda p: np.arctan2(p[1], p[0]))
        page = [(cx + r * px, y0 + r * py) for px, py in pts]
        col = "ndSky" if inside else "ndRose"
        if len(page) > 2:
            b.append("\\filldraw[fill=%s, draw=ndBlue, line width=0.8pt] %s -- cycle;" % (col, " -- ".join(N.pt(p) for p in page)))
        else:
            b.append("\\draw[line width=1.2pt, %s] %s -- %s;" % ("ndBlue" if inside else "ndRed", N.pt(page[0]), N.pt(page[1])))
        for (px, py), p in zip(pts, page):
            mult = sum(1 for w in z if abs(w - complex(px, py)) < 1e-6)
            b.append("\\node[dot, fill=ndBlue, minimum size=3.2pt] at %s {};" % N.pt(p))
            if mult > 1:
                u = np.array([px, py])
                b.append("\\node[font=\\tiny, text=ndInk!80, anchor=west] at %s {$\\times%d$};" % (N.pt(np.array(p) + np.array([0.09, 0.0])), mult))
        b.append("\\draw[ndRed, line width=0.7pt] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
                 % (cx - 0.06, y0 - 0.06, cx + 0.06, y0 + 0.06, cx - 0.06, y0 + 0.06, cx + 0.06, y0 - 0.06))
        b.append("\\node[font=\\scriptsize\\bfseries, text=ndRed] at (%.3f,%.3f) {%s};" % (cx - r - 0.05, y0 + r + 0.05, name))
        b.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {$%s$};" % (cx, y0 + r + 0.32, coords[name]))
        b.append("\\node[note, anchor=north] at (%.3f,%.3f) {%s};" % (cx, y0 - r - 0.18, verdicts[name]))
    b.append("\\node[font=\\small, anchor=north, align=center] at (%.3f,%.3f) {(b) the eigenvalues of $m(U)$ and their convex hull};"
             % (x0 + dx, y0 - r - 1.25))

    # ---- (c) by volume and by Haar measure
    bx, by, bw, bh = x0 - 0.6, -1.2, 5.7, 0.3
    for k, (lab, f) in enumerate([("by volume in the chamber", 0.5), ("by the Haar measure", haar)]):
        yk = by - k * 0.8
        b.append("\\fill[ndSky] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (bx, yk, bx + f * bw, yk + bh))
        b.append("\\fill[ndRose] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (bx + f * bw, yk, bx + bw, yk + bh))
        b.append("\\draw[ndInk!60, line width=0.4pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (bx, yk, bx + bw, yk + bh))
        b.append("\\node[font=\\scriptsize, anchor=south west] at (%.3f,%.3f) {%s};" % (bx - 0.05, yk + bh, lab))
        b.append("\\node[font=\\scriptsize\\bfseries, text=ndBlue, anchor=east] at (%.3f,%.3f) {$%d\\%%$};" % (bx + f * bw - 0.06, yk + bh / 2, round(100 * f)))
    b.append("\\node[font=\\small, anchor=north, align=center] at (%.3f,%.3f) {(c) the share of perfect entanglers};"
             % (bx + bw / 2, by - 0.95))
    N.write("ch07_entanglers", b)
    print("written figures/ch07_entanglers.tex")


if __name__ == "__main__":
    main()
