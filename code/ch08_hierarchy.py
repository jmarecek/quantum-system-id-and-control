"""ch08_hierarchy.py -- nested classes of dynamical maps as a TikZ diagram.

Output: figures/ch08_hierarchy.tex
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

LEVELS = [
    ("Hermiticity-preserving, trace-preserving linear maps", "gray!10"),
    ("positive trace-preserving maps (P)", "qocA!12"),
    ("completely positive trace-preserving maps (CPTP): Kraus, Choi $\\succeq 0$", "qocA!25"),
    ("CP-divisible families: $\\mathcal{E}_{t,s}$ CPTP, time-local GKSL with $\\gamma_k(t)\\ge 0$", "qocC!25"),
    ("dynamical semigroups $\\mathcal{E}_t=\\ee^{t\\mathcal{L}}$: GKSL with constant $H,\\gamma$", "qocB!30"),
    ("unitary conjugations $\\rho\\mapsto U\\rho U^\\dagger$", "qocD!35"),
]

if __name__ == "__main__":
    W, H, step = 13.0, 0.95, 0.95
    out = []
    n = len(LEVELS)
    for i, (label, fill) in enumerate(LEVELS):
        w = W - 1.6 * i
        h = H * (n - i) + 0.35 * (n - i - 1)
        y0 = -h
        out.append("\\draw[rounded corners=6pt, fill=%s, draw=black!60] (%.2f,%.2f) rectangle (%.2f,0);"
                   % (fill, -w / 2, y0, w / 2))
        out.append("\\node[anchor=south, font=\\footnotesize, text width=%.2fcm, align=center] at (0,%.2f) {%s};"
                   % (w - 0.4, y0 + 0.05, label))
    out.append("\\node[anchor=west, font=\\footnotesize, gray] at (%.2f,-0.5) {non-Markovian $\\uparrow$};" % (W / 2 + 0.2))
    out.append("\\node[anchor=west, font=\\footnotesize, gray] at (%.2f,%.2f) {Markovian $\\downarrow$};" % (W / 2 + 0.2, -H * n))
    T.write_tikz("ch08_hierarchy", "\n".join(out))
    print("wrote figures/ch08_hierarchy.tex")
