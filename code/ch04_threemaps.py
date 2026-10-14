"""ch04_threemaps.py -- three questions, three maps.

Controllability asks whether the map from inputs to states is onto,
observability whether the map from initial states to outputs is one-to-one,
and identifiability whether the map from parameters to input-output behaviour
is one-to-one.  The first two are questions about the states of a given model;
the third is a question about the model itself.  Each map is drawn between two
blobs, with a failure: an unreachable state, and two states, or two parameter
values, with the same image.  Writes figures/ch04_threemaps.tex.
"""
import numpy as np
import needham as N


def blob(cx, cy, rx, ry, ph, style):
    t = np.linspace(0, 2 * np.pi, 160)
    r = 1 + 0.07 * np.sin(3 * t + ph) + 0.04 * np.cos(5 * t + 2 * ph)
    pts = [(cx + rx * r[i] * np.cos(t[i]), cy + ry * r[i] * np.sin(t[i])) for i in range(len(t))]
    return "\\filldraw[%s] %s -- cycle;" % (style, " -- ".join(N.pt(p) for p in pts))


def wave(cx, cy, col, w=0.42, h=0.11, ph=0.0):
    s = np.linspace(-1, 1, 30)
    return N.polyline([(cx + w / 2 * x, cy + h * np.sin(5 * x + ph)) for x in s], "line width=0.9pt, %s" % col)


def arrow(p, q, bend=12, col="ndInk!70"):
    return "\\draw[->, >=stealth, line width=0.7pt, %s, shorten <=3pt, shorten >=3pt] %s to[bend left=%d] %s;" % (col, N.pt(p), bend, N.pt(q))


def main():
    b = []
    LEFT = "fill=ndRose, fill opacity=0.6, draw=ndInk, line width=0.8pt"
    RIGHT = "fill=ndSky, fill opacity=0.6, draw=ndInk, line width=0.8pt"
    # (a) controllability: inputs -> states, onto?
    ox = 0.0
    b.append(blob(ox + 0.9, 0.0, 0.75, 1.15, 0.3, LEFT))
    b.append(blob(ox + 3.7, 0.0, 1.05, 1.3, 1.1, RIGHT))
    band = [(ox + 2.9, -0.85), (ox + 4.3, 0.35), (ox + 4.55, 0.0), (ox + 3.15, -1.15)]
    b.append("\\filldraw[fill=ndBlue!35, draw=ndBlue, line width=0.6pt] %s -- cycle;" % " -- ".join(N.pt(p) for p in band))
    ins = [(ox + 0.9, 0.6), (ox + 0.9, 0.0), (ox + 0.9, -0.6)]
    outs = [(ox + 4.1, 0.05), (ox + 3.7, -0.4), (ox + 3.3, -0.75)]
    for k, (p, q) in enumerate(zip(ins, outs)):
        b.append(wave(*p, "ndRed", ph=k))
        b.append("\\fill[ndInk] %s circle (1.5pt);" % N.pt(q))
        b.append(arrow((p[0] + 0.25, p[1]), q, bend=-10 + 8 * k))
    b.append("\\node[font=\\large, text=ndRed] at %s {$\\times$};" % N.pt((ox + 3.55, 0.85)))
    b.append("\\node[note, anchor=south] at %s {not reachable};" % N.pt((ox + 3.55, 0.95)))
    b.append("\\node[note, text=ndBlue, anchor=west] at %s {reachable};" % N.pt((ox + 4.62, -0.35)))
    b.append("\\node[font=\\small] at %s {inputs $u(\\cdot)$};" % N.pt((ox + 0.9, -1.55)))
    b.append("\\node[font=\\small] at %s {states $\\vec x(T)$};" % N.pt((ox + 3.7, -1.75)))
    b.append("\\node[font=\\bfseries] at %s {controllability};" % N.pt((ox + 2.3, 2.0)))
    b.append("\\node[note] at %s {is every state reached?\\\\ is the map \\emph{onto}?};" % N.pt((ox + 2.3, -2.45)))
    # (b) observability: initial states -> outputs, one-to-one?
    ox = 5.9
    b.append(blob(ox + 0.95, 0.0, 0.85, 1.2, 2.0, RIGHT))
    b.append(blob(ox + 3.75, 0.0, 0.95, 1.15, 0.7, LEFT))
    x0, x1 = (ox + 0.65, 0.45), (ox + 1.25, -0.45)
    b.append("\\draw[dashedink, line width=0.8pt] %s -- %s;" % (N.pt(x0), N.pt(x1)))
    for p, lab, pos in ((x0, "$\\vec x$", "left"), (x1, "$\\vec x'$", "right")):
        b.append("\\node[dot, label={[font=\\footnotesize]%s:%s}] at %s {};" % (pos, lab, N.pt(p)))
    y = (ox + 3.75, 0.0)
    b.append(wave(*y, "ndBlue", w=0.55))
    b.append(arrow(x0, (y[0] - 0.3, y[1] + 0.05), bend=-15))
    b.append(arrow(x1, (y[0] - 0.3, y[1] - 0.05), bend=15))
    b.append("\\node[note, anchor=north] at %s {same output};" % N.pt((ox + 3.75, -0.25)))
    b.append("\\node[note, anchor=north] at %s {differ along $\\ker\\OM$};" % N.pt((ox + 0.95, -0.95)))
    b.append("\\node[font=\\small] at %s {initial states $\\vec x(0)$};" % N.pt((ox + 0.95, -1.65)))
    b.append("\\node[font=\\small] at %s {outputs $\\vec y(\\cdot)$};" % N.pt((ox + 3.75, -1.6)))
    b.append("\\node[font=\\bfseries] at %s {observability};" % N.pt((ox + 2.35, 2.0)))
    b.append("\\node[note] at %s {do different states give\\\\ different outputs? \\emph{one-to-one}?};" % N.pt((ox + 2.35, -2.45)))
    # (c) identifiability: parameters -> behaviours, one-to-one?
    ox = 11.8
    b.append(blob(ox + 0.95, 0.0, 0.85, 1.2, 1.4, "fill=ndGold!25, draw=ndInk, line width=0.8pt"))
    b.append(blob(ox + 3.75, 0.0, 0.95, 1.15, 2.6, LEFT))
    s = np.linspace(-0.8, 0.8, 40)
    fib = [(ox + 0.95 + 0.45 * x, 0.55 * x + 0.12 * np.sin(3 * x)) for x in s]
    b.append(N.polyline(fib, "line width=1.4pt, ndRed"))
    t0, t1 = fib[8], fib[31]
    for p, lab, pos in ((t0, "$\\vec\\theta$", "left"), (t1, "$\\vec\\theta'$", "right")):
        b.append("\\node[dot, label={[font=\\footnotesize]%s:%s}] at %s {};" % (pos, lab, N.pt(p)))
    beh = (ox + 3.75, 0.0)
    b.append("\\node[draw=ndInk, line width=0.6pt, rounded corners=1.5pt, fill=white, font=\\scriptsize, inner sep=1.5pt] at %s {$u\\mapsto y$};" % N.pt(beh))
    b.append(arrow(t0, (beh[0] - 0.42, beh[1] - 0.05), bend=10))
    b.append(arrow(t1, (beh[0] - 0.42, beh[1] + 0.05), bend=-10))
    b.append("\\node[note, anchor=north] at %s {same behaviour};" % N.pt((ox + 3.75, -0.28)))
    b.append("\\node[note, text=ndRed, anchor=south] at %s {a valley of equal fits};" % N.pt((ox + 0.95, 1.32)))
    b.append("\\node[font=\\small] at %s {parameters $(\\vec\\theta,\\gamma)$};" % N.pt((ox + 0.95, -1.65)))
    b.append("\\node[font=\\small] at %s {behaviours};" % N.pt((ox + 3.75, -1.6)))
    b.append("\\node[font=\\bfseries] at %s {identifiability};" % N.pt((ox + 2.35, 2.0)))
    b.append("\\node[note] at %s {do different parameters give\\\\ different behaviour? \\emph{one-to-one}?};" % N.pt((ox + 2.35, -2.45)))
    # what the questions are about
    b.append("\\draw[faint, line width=0.7pt] (0.1,-2.9) -- (0.1,-3.05) -- (10.6,-3.05) -- (10.6,-2.9);")
    b.append("\\node[note] at (5.35,-3.45) {questions about the \\emph{states} of a given model};")
    b.append("\\draw[faint, line width=0.7pt] (11.9,-2.9) -- (11.9,-3.05) -- (16.5,-3.05) -- (16.5,-2.9);")
    b.append("\\node[note] at (14.2,-3.45) {a question about the \\emph{model} itself};")
    N.write("ch04_threemaps", b)
    print("written figures/ch04_threemaps.tex")


if __name__ == "__main__":
    main()
