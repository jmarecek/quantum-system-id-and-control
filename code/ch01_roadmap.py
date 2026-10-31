"""ch01_roadmap.py -- the map of the course: four parts, thirteen lectures, and
the dependencies between them.  Each part is a column of lecture cards on a
pale wash of its colour; an arrow a -> b means that lecture b uses lecture a.
Writes figures/ch01_roadmap.tex.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

PARTS = [  # (title, colour, [(lecture number, short title)])
    ("Models", "ndBlue", [(1, "motivating examples"), (2, "closed systems"), (3, "open systems")]),
    ("Identification", "ndGold", [(4, "state space"), (5, "frequency domain"), (6, "sample complexity")]),
    ("Geometry", "ndGreen", [(7, "states"), (8, "channels"), (9, "optimal control"), (10, "circuit complexity")]),
    ("Algorithms", "ndRed", [(11, "tomography"), (12, "optimal control"), (13, "examples revisited")]),
]
DEPS = [(2, 3), (3, 4), (4, 5), (5, 6), (2, 7), (3, 8), (7, 9), (9, 10), (6, 11), (4, 11), (9, 12), (10, 12),
        (11, 13), (12, 13)]
COLW, CARDW, CARDH, ROWH = 3.35, 2.75, 0.9, 1.25      # column pitch, card size, row pitch (cm)


def main():
    b = []
    where = {}
    for c, (title, col, lects) in enumerate(PARTS):
        x = c * COLW
        top, bottom = 1.08, -ROWH * (len(lects) - 1) - CARDH / 2 - 0.25
        b.append("\\fill[%s!9, rounded corners=6pt] (%.2f,%.2f) rectangle (%.2f,%.2f);"
                 % (col, x - CARDW / 2 - 0.18, bottom, x + CARDW / 2 + 0.18, top))
        b.append("\\node[font=\\bfseries, text=%s!85!ndInk] at (%.2f,0.78) {%s};" % (col, x, title))
        for i, (n, name) in enumerate(lects):
            y = -ROWH * i
            where[n] = (x, y)
            b.append("\\node[draw=%s, line width=0.7pt, rounded corners=3pt, fill=white, minimum width=%.2fcm, "
                     "minimum height=%.2fcm, inner sep=1pt] (L%d) at (%.2f,%.2f) {};" % (col, CARDW, CARDH, n, x, y))
            b.append("\\node[circle, fill=%s, text=white, font=\\footnotesize\\bfseries, inner sep=0pt, minimum size=0.42cm] "
                     "at (%.2f,%.2f) {%d};" % (col, x - CARDW / 2 + 0.3, y, n))
            label = name.replace(" ", "\\\\", 1) if len(name) > 11 else name      # long titles on two lines
            b.append("\\node[font=\\small, anchor=west, align=left, inner sep=0pt] at (%.2f,%.2f) {%s};" % (x - CARDW / 2 + 0.58, y, label))
    arrow = "->, >=stealth, line width=0.6pt, ndInk!55, shorten >=1pt, shorten <=1pt"
    for a, c in DEPS:
        (xa, ya), (xc, yc) = where[a], where[c]
        if xa == xc and abs(ya - yc - ROWH) < 1e-9:            # down a column, to the next card
            b.append("\\draw[%s] (L%d.south) -- (L%d.north);" % (arrow, a, c))
        elif xa == xc:                                         # down a column, past a card: round its right side
            b.append("\\draw[%s] (L%d.345) .. controls +(0.75,0) and +(0.75,0) .. (L%d.15);" % (arrow, a, c))
        elif abs(xc - xa - COLW) < 1e-9 and ya == yc:          # straight across to the next column
            b.append("\\draw[%s] (L%d.east) -- (L%d.west);" % (arrow, a, c))
        elif abs(xc - xa - COLW) < 1e-9:                       # to the next column, another row
            b.append("\\draw[%s] (L%d.east) to[out=0, in=180] (L%d.west);" % (arrow, a, c))
        else:                                                  # over a column: through the gap between two rows
            ygap = max(ya, yc) - ROWH / 2 if ya != yc else ya - ROWH / 2
            b.append("\\draw[%s] (L%d.east) to[out=0, in=180] (%.2f,%.2f) -- (%.2f,%.2f) to[out=0, in=180] (L%d.west);"
                     % (arrow, a, xa + COLW - CARDW / 2 - 0.05, ygap, xc - COLW + CARDW / 2 + 0.05, ygap, c))
    # lecture 1 is taken up again in lecture 13: a dashed arrow under the map
    yb = -ROWH * 3 - 0.85
    b.append("\\draw[%s, dash pattern=on 3pt off 2pt] (L1.west) -- ++(-0.3,0) |- (%.2f,%.2f) -| (L13.south);"
             % (arrow, where[13][0] - 0.6, yb))
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {the examples of lecture 1 are solved in lecture 13};"
             % ((where[1][0] + where[13][0]) / 2, yb - 0.05))
    N.write("ch01_roadmap", b)
    print("written figures/ch01_roadmap.tex")


if __name__ == "__main__":
    main()
