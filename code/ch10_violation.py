"""ch10_violation.py -- the energy bookkeeping of the penalised geodesics, and the a priori bound.

The example of ch10_su2.py (target exp(-i (pi/2) sigma_z/2), time [0, 1]).
Along the minimising g_q-geodesic gamma_q the field in the plane has the size
w_p and the penalised part w_z, both constant in time, so that
    E_q(gamma_q) = (w_p^2 + q w_z^2)/2 = d_q^2/2,   E_1(gamma_q) = (w_p^2 + w_z^2)/2,
    violation = int g(P_perp gamma_q', P_perp gamma_q') dt = w_z^2,
and E_q - E_1 = (q - 1) w_z^2 / 2.

(a) E_q(gamma_q) increases to min E_inf = d_inf^2/2 = 8.636; so does
    E_1(gamma_q), which is constant, phi^2/2 = 1.234, along the direct route
    (q <= 4) and then rises; the gap between them, (q - 1)/2 times the
    violation, closes.
(b) The violation against q on logarithmic scales, with the a priori bound
    (2 min E_inf - d_g^2)/(q - 1) = 14.80/(q - 1) of the Proposition.  It is
    phi^2 = 2.467 up to q = 4 and then (3 pi/2)^2/(q - 1)^2, so that it decays
    like q^{-2}, faster than the bound, which it never reaches: the ratio is
    largest, 1/2, at q = 4.  q times the violation (green) tends to 0 like 22.2/q.
Writes figures/ch10_violation.tex.
"""
import numpy as np
import needham as N
import ch10_su2 as S


def checks():
    S.checks()
    Einf = S.d_inf() ** 2 / 2
    assert abs(Einf - 8.636) < 1e-3 and abs(S.PHI ** 2 / 2 - 1.234) < 1e-3
    for q in (1.0, 2.0, 4.0):
        Eq, E1, v = S.energies(q)
        assert abs(E1 - S.PHI ** 2 / 2) < 1e-12 and abs(v - S.PHI ** 2) < 1e-12
    for q in (4.5, 8.0, 64.0, 1e3, 1e5):
        Eq, E1, v = S.energies(q)
        assert abs((Eq - E1) - (q - 1) * v / 2) < 1e-9
        assert abs(v - (1.5 * np.pi) ** 2 / (q - 1) ** 2) < 1e-12
    assert abs((1.5 * np.pi) ** 2 - 22.2) < 0.01
    assert abs(S.energies(1e6)[0] - Einf) < 1e-4 and abs(S.energies(1e6)[1] - Einf) < 1e-4


def main():
    checks()
    b = []
    Einf = S.d_inf() ** 2 / 2
    # ---- (a) the energies
    qs = np.geomspace(1, 1e4, 500)
    E = np.array([S.energies(q) for q in qs])
    pa = N.Plot((0, 4), (0, 9.5), 5.6, 3.8, origin=(1.0, 0.75))
    b += pa.axes(xlabel="{\\scriptsize $q$}", xticks=[(0, "$1$"), (np.log10(4), "$4$"), (2, "$10^2$"), (4, "$10^4$")],
                 yticks=[(S.PHI ** 2 / 2, "$1.23$"), (Einf, "$8.64$")])
    b.append("\\fill[ndSky!60] %s -- %s -- cycle;" % (" -- ".join(pa.pt(np.log10(q), e) for q, e in zip(qs, E[:, 0])),
                                                       " -- ".join(pa.pt(np.log10(q), e) for q, e in zip(qs[::-1], E[::-1, 1]))))
    b.append(pa.curve([0, 4], [Einf, Einf], "line width=0.5pt, ndInk!60, densely dotted"))
    b.append(pa.curve(np.log10(qs), E[:, 0], "line width=1.3pt, ndBlue"))
    b.append(pa.curve(np.log10(qs), E[:, 1], "line width=1.3pt, ndGreen"))
    b.append("\\draw[dashedink] %s -- %s;" % (pa.pt(np.log10(4), 0), pa.pt(np.log10(4), 9.2)))
    q7 = 7.0
    b.append("\\node[note, anchor=west, text=ndBlue] (eq) at %s {$E_q(\\gamma_q)=\\min E_q$};" % pa.pt(2.0, 6.7))
    b.append("\\draw[pointer] (eq.west) to[bend right=10] %s;" % pa.pt(np.log10(q7) + 0.04, S.energies(q7)[0] + 0.05))
    b.append("\\node[note, anchor=west] (gap) at %s {gap: $\\tfrac{q-1}2\\times$ violation};" % pa.pt(2.0, 5.0))
    b.append("\\draw[pointer] (gap.west) to[bend left=10] %s;" % pa.pt(np.log10(q7), 0.5 * (S.energies(q7)[0] + S.energies(q7)[1])))
    b.append("\\node[note, anchor=west, text=ndGreen] (e1) at %s {$E_1(\\gamma_q)$};" % pa.pt(2.0, 3.3))
    b.append("\\draw[pointer] (e1.west) to[bend left=10] %s;" % pa.pt(np.log10(10) + 0.03, S.energies(10)[1] - 0.15))
    b.append("\\node[note, anchor=south] at %s {$\\min E_\\infty=\\tfrac12d_\\infty^2$};" % pa.pt(3.0, Einf + 0.15))
    b.append("\\node[note, anchor=north west] at %s {direct route for $q\\le4$};" % pa.pt(0.03, 1.0))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the energies};" % pa.pt(2.0, -1.7))

    # ---- (b) the violation and the bound
    qs = np.geomspace(2, 1e5, 500)
    v = np.array([S.energies(q)[2] for q in qs])
    pb = N.Plot((np.log10(2), 5), (-9, 1.6), 5.6, 3.8, origin=(9.3, 0.75))
    b += pb.axes(xlabel="{\\scriptsize $q$}", xticks=[(1, "$10$"), (2, "$10^2$"), (3, "$10^3$"), (4, "$10^4$"), (5, "$10^5$")],
                 yticks=[(-8, "$10^{-8}$"), (-6, "$10^{-6}$"), (-4, "$10^{-4}$"), (-2, "$10^{-2}$"), (0, "$1$")])
    b.append(pb.curve(np.log10(qs), np.log10(S.bound(qs)), "line width=1.0pt, ndInk, dash pattern=on 4pt off 2pt"))
    b.append(pb.curve(np.log10(qs), np.log10(v), "line width=1.4pt, ndBlue"))
    b.append(pb.curve(np.log10(qs), np.log10(qs * v), "line width=1.2pt, ndGreen"))
    b.append("\\node[note, anchor=north east] at %s {a priori bound $\\frac{14.80}{q-1}$};" % pb.pt(4.95, np.log10(S.bound(1e5)) - 0.35))
    b.append("\\node[note, anchor=north east, text=ndBlue] at %s {violation\\\\ $\\int g(P^\\perp\\dot\\gamma_q,P^\\perp\\dot\\gamma_q)$,\\\\ slope $-2$};" % pb.pt(3.05, np.log10(S.energies(1000)[2]) - 0.35))
    b.append("\\node[note, anchor=south west, text=ndGreen] at %s {$q\\,\\times$ violation, slope $-1$};" % pb.pt(2.2, np.log10(150 * S.energies(150)[2]) + 0.3))
    b.append("\\node[note, anchor=north west] (hf) at %s {half the\\\\ bound at $q=4$};" % pb.pt(0.75, -1.2))
    b.append("\\draw[pointer] (hf.north west) to[bend left=20] %s;" % pb.pt(np.log10(4), np.log10(S.energies(4.0)[2]) - 0.1))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the constraint violation};" % pb.pt(3.15, -10.9))
    N.write("ch10_violation", b)


if __name__ == "__main__":
    main()
    for q in (2, 4, 16, 256, 1e4):
        Eq, E1, v = S.energies(q)
        print("q=%g: E_q=%.4f E_1=%.4f violation=%.3g bound=%.3g" % (q, Eq, E1, v, S.bound(q)))
    print("written figures/ch10_violation.tex")
