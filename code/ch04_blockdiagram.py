"""ch04_blockdiagram.py -- prepare, evolve, measure: the state-space picture.

A block diagram of the bilinear state-space model of an open quantum system,
    x' = A x + beta + sum_j u_j N_j x,   x(0) = B x0,   y = C x,
with the blocks chosen by the experimenter (preparation B, controls u_j,
readout C) washed in rose and the blocks given by nature, to be identified
(the generator A, the coupling matrices N_j), washed in sky blue.  Without the
lower loop the model is linear.  Writes figures/ch04_blockdiagram.tex.
"""
import needham as N

BOX = "draw=ndInk, line width=0.9pt, rounded corners=3pt, minimum width=0.95cm, minimum height=0.75cm, font=\\large"
NATURE = BOX + ", fill=ndSky"
CHOSEN = BOX + ", fill=ndRose"
WIRE = "->, >=stealth, line width=0.9pt, ndInk"


def main():
    b = []
    # equation on top
    b.append("\\node at (5.6,3.25) {$\\dot{\\vec x}=\\mathbf A\\vec x+\\vec\\beta+\\sum_j u_j\\,\\mathbf N_j\\vec x,"
             "\\qquad \\vec x(0)=\\mathbf B\\vec x_0,\\qquad \\vec y=\\mathbf C\\vec x$};")
    # blocks
    b.append("\\node[circle, draw=ndInk, line width=0.9pt, inner sep=1.5pt, fill=white] (S) at (3.0,0) {$+$};")
    b.append("\\node[%s] (I) at (4.8,0) {$\\int$};" % NATURE)
    b.append("\\node[dot] (P) at (6.6,0) {};")
    b.append("\\node[%s] (C) at (8.4,0) {$\\mathbf C$};" % CHOSEN)
    b.append("\\node[%s] (B) at (4.8,1.75) {$\\mathbf B$};" % CHOSEN)
    b.append("\\node[%s] (A) at (4.8,-1.35) {$\\mathbf A$};" % NATURE)
    b.append("\\node[%s] (Nj) at (5.7,-2.75) {$\\mathbf N_j$};" % NATURE)
    b.append("\\node[circle, draw=ndInk, line width=0.9pt, inner sep=1pt, fill=white] (M) at (4.0,-2.75) {$\\times$};")
    b.append("\\node[%s, minimum width=1.1cm, font=\\normalsize] (U) at (4.0,-4.0) {$u_j(t)$};" % CHOSEN)
    # wires
    b.append("\\draw[%s] (S) -- (I);" % WIRE)
    b.append("\\draw[line width=0.9pt, ndInk] (I) -- node[above] {$\\vec x(t)$} (P);")
    b.append("\\draw[%s] (P) -- (C);" % WIRE)
    b.append("\\draw[%s] (C) -- (10.3,0) node[right] {$\\vec y(t)$};" % WIRE)
    b.append("\\draw[%s] (3.75,1.75) node[left] {$\\vec x_0$} -- (B);" % WIRE)
    b.append("\\draw[%s] (B) -- node[right, font=\\small] {$\\vec x(0)$} (I);" % WIRE)
    b.append("\\draw[%s] (3.0,1.1) node[above] {$\\vec\\beta$} -- (S);" % WIRE)
    b.append("\\draw[%s] (P) |- (A);" % WIRE)
    b.append("\\draw[%s] (A) -| (S);" % WIRE)
    b.append("\\draw[%s] (P) |- (Nj);" % WIRE)
    b.append("\\draw[%s] (Nj) -- (M);" % WIRE)
    b.append("\\draw[%s] (U) -- (M);" % WIRE)
    b.append("\\draw[%s] (M) -| (2.2,0) -- (S);" % WIRE)
    # notes
    b.append("\\node[note, anchor=west] (nB) at (6.6,1.85) {state preparation;\\\\ any state: $\\mathbf B=\\Id$};")
    b.append("\\draw[pointer] (nB.west) -- (B.east);")
    b.append("\\node[note, anchor=south] at (8.1,0.45) {observables we\\\\ choose to measure};")
    b.append("\\node[note, anchor=south] at (11.1,0.3) {expectation values,\\\\ estimated from shots};")
    b.append("\\node[note, anchor=west] (nA) at (7.0,-1.35) {GKSL generator on the\\\\ coherence vector:\\\\ Hamiltonian and dissipation};")
    b.append("\\draw[pointer] (nA.west) to[bend right=10] (A.east);")
    b.append("\\node[note, anchor=west] (nN) at (7.0,-2.95) {controls act \\emph{multiplicatively}:\\\\ $u_j\\mathbf N_j\\vec x$, a bilinear system;\\\\ without this loop, a linear one};")
    b.append("\\draw[pointer] (nN.west) to[bend left=10] (Nj.east);")
    b.append("\\node[note, anchor=east] at (2.8,1.3) {affine term};")
    # legend
    b.append("\\node[draw=ndInk, line width=0.6pt, rounded corners=2pt, fill=ndRose, minimum width=0.5cm, minimum height=0.3cm] at (7.2,-4.25) {};")
    b.append("\\node[note, anchor=west] at (7.55,-4.25) {chosen by the experimenter};")
    b.append("\\node[draw=ndInk, line width=0.6pt, rounded corners=2pt, fill=ndSky, minimum width=0.5cm, minimum height=0.3cm] at (7.2,-3.8) {};")
    b.append("\\node[note, anchor=west] at (7.55,-3.8) {given by nature, to be identified};")
    N.write("ch04_blockdiagram", b)
    print("written figures/ch04_blockdiagram.tex")


if __name__ == "__main__":
    main()
