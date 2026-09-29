"""ch01_roadmap.py -- the map of the course: four parts, thirteen lectures, and
the dependencies between them, as a TikZ diagram.  Writes figures/ch01_roadmap.tex.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

parts = [  # (title, colour, [(lecture number, short title)])
    ("Models", "qocA", [(1, "motivating examples"), (2, "closed systems"), (3, "open systems")]),
    ("Identification", "qocB", [(4, "state space"), (5, "frequency domain"), (6, "sample complexity")]),
    ("Geometry", "qocC", [(7, "states"), (8, "channels"), (9, "optimal control"), (10, "circuit complexity")]),
    ("Algorithms", "qocD", [(11, "tomography"), (12, "optimal control"), (13, "examples revisited")]),
]
out = []
x = 0.0
colw = 3.6
for title, col, lects in parts:
    out.append(r"\node[font=\bfseries, text=%s] at (%.2f,0.9) {%s};" % (col, x + colw/2 - 0.3, title))
    for i, (n, name) in enumerate(lects):
        y = -1.1 * i
        out.append(r"\node[draw=%s, thick, rounded corners, fill=%s!8, minimum width=%.1fcm, minimum height=0.85cm, align=center, font=\small] (L%d) at (%.2f,%.2f) {\textbf{%d}\ %s};"
                   % (col, col, colw - 0.5, n, x + colw/2 - 0.3, y, n, name))
    x += colw
# dependencies (arrows) between lectures
deps = [(2, 3), (3, 4), (4, 5), (5, 6), (2, 7), (3, 8), (7, 9), (9, 10), (6, 11), (4, 11), (9, 12), (10, 12), (11, 13), (12, 13), (1, 13)]
for a, b in deps:
    style = "->, >=stealth, gray!70, thick"
    if a == 1 and b == 13:
        out.append(r"\draw[%s, dashed] (L1.west) -- ++(-0.35,0) |- ($(L13.south)+(0,-0.35)$) -- (L13.south);" % style)
    else:
        out.append(r"\draw[%s] (L%d) -- (L%d);" % (style, a, b))
T.write_tikz("ch01_roadmap", "\n".join(out), options="x=1cm,y=1cm")
print("written figures/ch01_roadmap.tex")
