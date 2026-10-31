"""ch10_complexity.py -- schematic of the penalty-metric picture.

Left: unit balls {v : g_q(v,v) = 1} of the penalised metrics for q = 1, 4, 16
in a plane spanned by a cheap direction (horizontal) and an expensive
direction (vertical).  As q grows the ball flattens onto the cheap line: the
sub-Riemannian limit, where only horizontal velocities are allowed.
Right: a path from I to V that is short in the round metric but long in g_q,
versus a horizontal (zig-zag) path whose g_q-length does not depend on q.
Exports figures/ch10_complexity.tex.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

out = []
# --- left panel: unit balls ---------------------------------------------------
out.append("\\begin{scope}")
out.append("\\draw[->] (-2.6,0) -- (2.6,0) node[right,font=\\small] {cheap ($D$)};")
out.append("\\draw[->] (0,-2.3) -- (0,2.3) node[above,font=\\small] {expensive ($D^\\perp$)};")
for k, q in enumerate([1, 4, 16]):
    colour = T.PALETTE[k][0]
    out.append("\\draw[thick,%s] (0,0) ellipse (2 and %.3f);" % (colour, 2.0 / np.sqrt(q)))
    out.append("\\node[%s,font=\\scriptsize,anchor=south west] at (%.2f,%.2f) {$q=%d$};"
               % (colour, 1.45, 2.0 / np.sqrt(q) * 0.7, q))
out.append("\\draw[very thick,black] (-2,0) -- (2,0);")
out.append("\\node[font=\\scriptsize,anchor=north] at (0,-0.1) {$q\\to\\infty$: horizontal segment};")
out.append("\\end{scope}")
# --- right panel: two paths --------------------------------------------------
out.append("\\begin{scope}[xshift=7.5cm]")
out.append("\\fill (-2.2,-1.5) circle (2pt) node[below,font=\\small] {$\\Id$};")
out.append("\\fill (2.2,1.5) circle (2pt) node[above,font=\\small] {$V$};")
out.append("\\draw[thick,%s] (-2.2,-1.5) -- (2.2,1.5);" % T.PALETTE[1][0])
out.append("\\node[%s,font=\\scriptsize,rotate=34] at (0.1,-0.25) {round geodesic: uses $D^\\perp$};" % T.PALETTE[1][0])
# horizontal zig-zag using only cheap directions (horizontal and 'a second cheap direction' drawn slanted)
pts = [(-2.2, -1.5)]
x, y = -2.2, -1.5
for i in range(6):
    x += 4.4 / 6.0
    pts.append((x, y))
    y += 3.0 / 6.0
    pts.append((x, y))
path = " -- ".join("(%.3f,%.3f)" % p for p in pts)
out.append("\\draw[thick,%s] %s;" % (T.PALETTE[0][0], path))
out.append("\\node[%s,font=\\scriptsize,anchor=north west] at (-2.2,-1.7) {horizontal path: length independent of $q$};" % T.PALETTE[0][0])
out.append("\\node[font=\\scriptsize,anchor=south east,text width=3.6cm,align=right] at (2.4,-1.4) {vertical legs stand for a second cheap direction whose bracket with the first generates $D^\\perp$};")
out.append("\\end{scope}")
T.write_tikz("ch10_complexity", "\n".join(out))
print("written figures/ch10_complexity.tex")
