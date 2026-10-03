"""ch03_dendrogram.py -- the dendrogram of open-quantum-system descriptions.

Redrawn after Figure 2 of the survey (Parvaiz et al.), as a TikZ diagram:
boxes are equations (red outline: non-Markovian, blue: Markovian), edges carry
the assumption that turns one description into the next.  Every box is a
hyperlink to the corresponding equation of Chapter 3, and every assumption a
hyperlink to its entry in the list of assumptions.  On the slides the diagram
is revealed route by route with beamer overlays (key `step`); in the handout,
where overlay specifications are ignored, it appears in full.
Writes figures/ch03_dendrogram.tex.
"""
import os
import needham as N

# ---------------------------------------------------------------- boxes
# id: (x, y, tag, Markovian?, equation label, step, LaTeX of the equation)
BOXES = {
    "a1": (8.6, 16.0, "(a)", False, "eq:ch03:nzme", 1,
           r"\Pproj\dot\rhofull=g\Pproj\Lind(t)\Pproj\rhofull(t)+g\Pproj\Lind(t)\mathcal G(t,t_0)\Qproj\rhofull(t_0)"
           r"+\int_{t_0}^{t}\Kker(t,\tau)\Pproj\rhofull(\tau)\,\mathrm d\tau"),
    "a2": (7.6, 13.4, "", False, "eq:ch03:nzme-hom", 1,
           r"\Pproj\dot\rhofull=\int_{t_0}^{t}\Kker(t,\tau)\Pproj\rhofull(\tau)\,\mathrm d\tau"),
    "c1": (14.4, 12.4, "(c)", False, "eq:ch03:born", 2,
           r"\dot\rhoS(t)=-\int_0^t\mathrm d\tau\,\TrB\big\{\comm{V_tH_I}{\comm{V_\tau H_I}{\rhofull(\tau)}}\big\}"),
    "c2": (14.4, 8.2, "", False, "eq:ch03:redfield", 2,
           r"\dot\rhoS(t)=-\int_0^t\mathrm d\tau\,\TrB\big\{\comm{V_tH_I}{\comm{V_\tau H_I}{\rhoS(t)\otimes\rhoB}}\big\}"),
    "c3": (14.4, 5.0, "", True, "eq:ch03:redfield-markov", 2,
           r"\dot\rhoS(t)=-\int_0^\infty\mathrm d\tau\,\TrB\big\{\comm{V_tH_I}{\comm{V_{t-\tau}H_I}{\rhoS(t)\otimes\rhoB}}\big\}"),
    "b1": (2.8, 11.4, "(b)", False, "eq:ch03:reduced", 3,
           r"\rhoS(t)=\TrB\big[U(t)\rhofull(0)U^\dagger(t)\big]"),
    "b2": (2.8, 8.9, "", False, "eq:ch03:kraus", 3,
           r"\rhoS(t)=\sum_{k}E_k(t)\rhoS(0)E_k^\dagger(t)"),
    "b3": (4.4, 6.6, "", False, "eq:ch03:lindblad", 3,
           r"\dot\rhoS=-\imath\comm{H(t)}{\rhoS}+\sum_k\gamma_k(t)\Big(L_k(t)\rhoS L_k^\dagger(t)-\tfrac12\acomm{L_k^\dagger(t)L_k(t)}{\rhoS}\Big)"),
    "d":  (6.8, 2.0, "(d)", True, "eq:ch03:lindblad-markov", 4,
           r"\dot\rhoS=-\imath\comm{H}{\rhoS}+\sum_k\gamma_k\Big(L_k\rhoS L_k^\dagger-\tfrac12\acomm{L_k^\dagger L_k}{\rhoS}\Big)"),
    "e1": (3.2, -2.0, "(e)", True, "eq:ch03:BLDS_lindbladian", 5,
           r"\dot\xvec=\Amat\xvec+\bvec"),
    "e2": (10.6, -2.0, "", True, "eq:ch03:BLDS_lindbladian", 5,
           r"\dot\xvec=\Amat\xvec+\bvec+\sum_j u_j\Nmat_j\xvec"),
}


def L(target, text):
    """An assumption that links to its entry in the list, in ink colour."""
    return r"\hyperref[%s]{\textcolor{ndInk!85}{%s}}" % (target, text)


SEP = L("ass:ch03:sep", r"$\rhofull(0)=\rhoS(0)\otimes\rhoB(0)$")

# ---------------------------------------------------------------- edges
# (path as a list of points or anchors, label, position on the first segment, anchor, step)
J1 = (14.4, 10.4)                          # Nakajima-Zwanzig joins the microscopic route
J2 = (4.4, 5.0)                            # the secular route joins the dynamical-map route
EDGES = [
    (["a1.south", "a1.south |- a2.north"], SEP, 0.5, "right", 1),
    (["a2.south", (7.6, 10.4), J1], r"$\mathcal G(t,t_0)=\Id+O(g)$", 0.35, "right", 2),
    (["c1.south", J1], "", 0.5, "right", 2),
    ([J1, "c2.north"],
     L("ass:ch03:born", r"Born: $\rhofull(\tau)\to\rhoS(\tau)\otimes\rhoB$") + r";\ "
     + L("ass:ch03:markov", r"Markov I: $\rhoS(\tau)\to\rhoS(t)$"), 0.5, "right", 2),
    (["c2.south", "c3.north"],
     L("ass:ch03:markov", r"Markov II: $\tau\to t-\tau$, $\int_0^t\to\int_0^\infty$"), 0.5, "right", 2),
    (["b1.south", "b2.north"], SEP, 0.5, "right", 3),
    (["b2.south", "b2.south |- b3.north"], r"differentiate; map invertible", 0.5, "right", 3),
    (["b3.south", J2], r"$L_k(t)\to L_k$, $\gamma_k(t)\to\gamma_k$", 0.5, "left", 4),
    (["c3.west", J2], r"secular: $\omega=\omega'$", 0.25, "above", 4),
    ([J2, "b3.south |- d.north"], r"$\gamma_k\ge0$", 0.55, "left", 4),
    (["d.south", (6.8, 0.0)], r"coherence vector $x_j=\Tr[F_j\rhoS]$", 0.45, "right", 5),
    ([(6.8, 0.0), (3.2, 0.0), "e1.north"], r"no control", 0.5, "above", 5),
    ([(6.8, 0.0), (10.6, 0.0), "e2.north"], r"control: $H=\Hdrift+\sum_ju_jF_j$", 0.42, "below", 5),
]

OVERLAY_STYLES = ", ".join([
    "invisible/.style={opacity=0, text opacity=0}",
    "alt/.code args={<#1>#2#3}{\\alt<#1>{\\pgfkeysalso{#2}}{\\pgfkeysalso{#3}}}",
    "step/.style={alt={<#1->{}{invisible}}}",
    "eqbox/.style={draw, line width=1.1pt, rounded corners=2pt, fill=white, inner sep=3pt, font=\\footnotesize}",
    "nonmarkov/.style={eqbox, draw=ndRed}",
    "markov/.style={eqbox, draw=ndBlue}",
    "tag/.style={font=\\small\\bfseries, text=ndInk}",
    "assume/.style={font=\\scriptsize\\itshape, text=ndInk!85, fill=white, fill opacity=0.85, text opacity=1, inner sep=1.5pt}",
    "link/.style={line width=0.8pt, ndInk!80}",
])


K = 0.66                                   # vertical compression, for a 16:9 slide


def point(p):
    return "(%s)" % p if isinstance(p, str) else "(%.2f,%.2f)" % (p[0], K * p[1])


def main():
    b = []
    for key, (x, y, tag, markov, lab, step, tex) in BOXES.items():
        style = "markov" if markov else "nonmarkov"
        b.append("\\node[%s, step=%d] (%s) at (%.2f,%.2f) {\\hyperref[%s]{\\textcolor{ndInk}{$\\displaystyle %s$}}};"
                 % (style, step, key, x, K * y, lab, tex))
        if tag:
            b.append("\\node[tag, anchor=east, step=%d] at (%s.west) {%s};" % (step, key, tag))
    for path, label, pos, anchor, step in EDGES:
        pts = " -- ".join(point(p) for p in path)
        lab = (" node[assume, step=%d, pos=%.2f, %s] {%s}" % (step, pos, anchor, label)) if label else ""
        # the label node is attached to the first segment for simplicity
        first = " -- ".join(point(p) for p in path[:2])
        rest = " -- ".join(point(p) for p in path[1:])
        b.append("\\draw[link, step=%d] %s%s;" % (step, first, lab))
        if len(path) > 2:
            b.append("\\draw[link, step=%d] %s;" % (step, rest))
    # legend
    b.append("\\node[nonmarkov, font=\\scriptsize, anchor=west] at (15.6,%.2f) {non-Markovian};" % (K * 16.0))
    b.append("\\node[markov, font=\\scriptsize, anchor=west] at (15.6,%.2f) {Markovian};" % (K * 16.0 - 0.6))
    b.append("\\node[assume, anchor=west] at (15.6,%.2f) {edges: assumptions};" % (K * 16.0 - 1.2))
    N.write("ch03_dendrogram", b)
    # TikZ keys for the overlays are added to the picture options
    path = os.path.join(N.T.FIGDIR, "ch03_dendrogram.tex")
    s = open(path).read()
    s = s.replace("\\begin{tikzpicture}[", "\\begin{tikzpicture}[" + OVERLAY_STYLES + ", ", 1)
    open(path, "w").write(s)
    print("written figures/ch03_dendrogram.tex")


if __name__ == "__main__":
    main()
