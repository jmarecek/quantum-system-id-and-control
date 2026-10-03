"""ch03_secular.py -- why the secular approximation works.

(a) A non-secular term oscillates as exp(i (omega' - omega) t).  Over the
system timescale its running average (1/t) int_0^t is of order
1 / (|omega'-omega| t) and vanishes, whereas a secular term (omega = omega')
keeps its full weight.  (b) The Bohr frequencies omega = E_j - E_k of a
three-level system: the approximation needs the gaps between distinct Bohr
frequencies to be large compared to 1/tau_S.  Writes figures/ch03_secular.tex.
"""
import numpy as np
import needham as N

DW = 4.0                                    # omega' - omega for the non-secular term
E = np.array([0.0, 1.0, 2.6])               # energies of a three-level system


def main():
    t = np.linspace(1e-3, 8, 800)
    osc = np.cos(DW * t)
    avg = np.sin(DW * t) / (DW * t)                       # (1/t) int_0^t cos(DW s) ds
    num = np.cumsum(osc) * (t[1] - t[0]) / t
    assert np.allclose(avg[50:], num[50:], atol=0.02)
    assert abs(avg[-1]) < 1 / (DW * t[-1]) + 1e-12
    b = []
    Pl = N.Plot((0, 8), (-1.05, 1.2), 8.0, 4.0)
    b += Pl.axes("$t$", "", yticks=[(-1, "$-1$"), (0, "$0$"), (1, "$1$")], yzero=0.0)
    b.append(Pl.curve(t, osc, "line width=0.8pt, ndRed!60"))
    b.append(Pl.curve(t, avg, "line width=1.6pt, ndRed"))
    b.append("\\draw[line width=1.6pt, ndBlue] %s -- %s;" % (Pl.pt(0, 1), Pl.pt(8, 1)))
    b.append("\\node[note, anchor=south, text=ndBlue] at %s {secular term, $\\omega=\\omega'$: kept};" % Pl.pt(4.3, 1.02))
    b.append("\\node[note, anchor=north, text=ndRed] (n) at %s {non-secular, $\\omega\\ne\\omega'$ (thin): its running average (thick)\\\\ dies like $1/(|\\omega-\\omega'|\\,t)$};" % Pl.pt(4.0, -1.12))
    b.append("\\draw[pointer] (n.north) to[bend right=25] %s;" % Pl.pt(2.6, avg[np.searchsorted(t, 2.6)] - 0.03))
    b.append("\\node[font=\\small] at %s {(a) fast terms average out};" % N.pt(Pl.P(4, -1.05) + np.array([0, -1.35])))
    # (b) Bohr frequencies
    bohr = sorted({round(a - c, 6) for a in E for c in E})
    o = np.array([13.4, 0.0])
    S = 1.25
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk] {$\\omega$};" % (N.pt(o + (-0.2 - 2.8 * S, 0)), N.pt(o + (2.9 * S, 0))))
    for w in bohr:
        col = "ndBlue" if w == 0 else "ndInk"
        b.append("\\draw[line width=1.4pt, %s] %s -- %s;" % (col, N.pt(o + (S * w, 0)), N.pt(o + (S * w, 1.2 if w == 0 else 0.8))))
        b.append("\\node[font=\\scriptsize, below] at %s {$%g$};" % (N.pt(o + (S * w, 0)), w))
    gaps = np.diff(bohr)
    g = gaps.min()
    k = int(np.argmin(gaps))
    y = 1.5
    b.append("\\draw[<->, >=stealth, line width=0.6pt, ndRed] %s -- %s node[midway, above, font=\\scriptsize, text=ndRed] {smallest gap $%g$};"
             % (N.pt(o + (S * bohr[k], y)), N.pt(o + (S * bohr[k + 1], y)), g))
    b.append("\\node[note, anchor=north] at %s {Bohr frequencies $E_j-E_k$ for $E=(0,1,2.6)$;\\\\ secular approximation valid if\\\\ (smallest gap)$^{-1}\\ll\\tau_S$};" % N.pt(o + (0, -0.6)))
    b.append("\\node[font=\\small] at %s {(b) the spectrum it relies on};" % N.pt(o + (0, -3.45)))
    N.write("ch03_secular", b)
    print("written figures/ch03_secular.tex; Bohr", bohr, "min gap", g)


if __name__ == "__main__":
    main()
