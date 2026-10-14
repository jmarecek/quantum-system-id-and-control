"""ch04_ladder.py -- the ladder from data to the master equation.

The conditions of Chapter 4 as a ladder: each rung is a question, answered
yes on the way down; next to it, an example from the damped qubit (or the two
coupled qubits) in which the answer is no, with what goes wrong.  The
results reached on the way are drawn as plain boxes.
Writes figures/ch04_ladder.tex.
"""
import needham as N

RUNG = "draw=ndInk, line width=0.9pt, rounded corners=3pt, fill=ndSky, text width=6.1cm, minimum height=0.9cm, align=center, font=\\small"
RESULT = "draw=ndInk, line width=0.6pt, fill=ndPaper, text width=6.1cm, minimum height=0.7cm, align=center, font=\\small"
FAIL = "note, anchor=west, align=left, text=ndRed!80!black"


def main():
    rungs = [
        ("c", 0.0, "rung", "controllable?\\\\ {\\footnotesize $\\rank\\CM=n$; free if any state can be prepared}",
         "only $\\ket1$ prepared: $x,y$ are never excited,\\\\ so $\\omega$ and $\\gamma_\\phi$ never act on the data"),
        ("o", -1.5, "rung", "observable?\\\\ {\\footnotesize $\\rank\\OM=n$; the accessible set is everything}",
         "only $\\sigma_z$ read: $x,y$ are never seen,\\\\ so $\\omega$ and $\\gamma_\\phi$ are invisible"),
        ("d", -3.0, "rung", "rich enough data?\\\\ {\\footnotesize several sampling rates; pulses of varying width}",
         "one sampling rate $\\tau$: $\\omega$ and $\\omega+2\\pi/\\tau$\\\\ give the same samples (aliasing)"),
        ("r1", -4.3, "result", "$(\\mathbf A,\\vec\\beta)$ up to a similarity $\\mathbf T$", None),
        ("f", -5.6, "rung", "frame fixed?\\\\ {\\footnotesize $\\mathbf B=\\Id$, or $\\mathbf B$ and $\\mathbf C$ known}",
         "preparation and readout both unknown:\\\\ a $16$-dimensional gauge, as in gate-set tomography"),
        ("r2", -6.9, "result", "$(\\mathbf A,\\vec\\beta)$ unique", None),
        ("m", -8.2, "rung", "$\\mathbf M$ invertible?\\\\ {\\footnotesize or $\\mathbf R_1,\\mathbf R_3$ of full rank; on the support of $\\gamma$}",
         "two qubits, any $\\gamma\\in\\Herm[15]$: $\\rank\\mathbf R_2=210<225$;\\\\ restricted to the physical support: full rank"),
        ("r3", -9.5, "result", "$(\\vec\\theta,\\gamma)$ unique: the master equation", None),
    ]
    b = []
    for name, y, kind, text, fail in rungs:
        b.append("\\node[%s] (%s) at (0,%.2f) {%s};" % (RUNG if kind == "rung" else RESULT, name, y, text))
        if fail:
            b.append("\\node[%s] (%s-no) at (4.0,%.2f) {%s};" % (FAIL, name, y, fail))
            b.append("\\draw[->, >=stealth, line width=0.7pt, ndRed!70, dash pattern=on 2.5pt off 1.5pt] (%s.east) -- node[above, font=\\scriptsize, text=ndRed!80!black] {no} (%s-no.west);" % (name, name))
    for a, c in zip(rungs[:-1], rungs[1:]):
        lab = "node[right, font=\\scriptsize] {yes}" if a[2] == "rung" else ""
        b.append("\\draw[->, >=stealth, line width=0.9pt, ndInk] (%s.south) -- %s (%s.north);" % (a[0], lab, c[0]))
    # brackets on the left
    for (top, bottom, text) in [(0.45, -1.95, "minimality"), (-2.55, -7.25, "identification\\\\ of $\\mathbf A$"), (-7.75, -9.85, "reconstruction")]:
        b.append("\\draw[faint, line width=0.8pt] (-3.35,%.2f) -- (-3.5,%.2f) -- (-3.5,%.2f) -- (-3.35,%.2f);" % (top, top, bottom, bottom))
        b.append("\\node[note, anchor=east, align=right] at (-3.6,%.2f) {%s};" % ((top + bottom) / 2, text))
    b.append("\\node[note, anchor=south] at (0,0.6) {measured data};")
    N.write("ch04_ladder", b)
    print("written figures/ch04_ladder.tex")


if __name__ == "__main__":
    main()
