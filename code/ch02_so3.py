"""ch02_so3.py -- rotations of space do not commute.

A marked frame (a tripod of three coloured unit vectors, with a small flag on
the first one) is rotated by 90 degrees about the x-axis and then about the
y-axis (top row), and in the opposite order (bottom row).  The end results
differ: R_y R_x != R_x R_y.  Writes figures/ch02_so3.tex.
"""
import numpy as np
import needham as N

Q = np.pi / 2


def Rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def Ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


COLS = ["ndRed", "ndGreen", "ndBlue"]
NAMES = ["e_1", "e_2", "e_3"]


def tripod(R, shift):
    V = N.View(az=35.0, el=22.0, R=1.05, shift=shift)
    b = V.sphere(shade=True)
    b += V.great_circle([0, 0, 1])
    for v, lab in [((1.35, 0, 0), "x"), ((0, 1.35, 0), "y"), ((0, 0, 1.35), "z")]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[font=\\tiny, text=ndInk!60] at %s {$%s$};"
                 % (V.pt((0, 0, 0)), V.pt(v), V.pt(1.12 * np.array(v)), lab))
    # the frame: three arrows, drawn back to front
    cols = R.T                                        # images of e_1, e_2, e_3
    order = np.argsort([c @ V.d for c in cols])
    for k in order:
        b.append("\\draw[vec, %s] %s -- %s;" % (COLS[k], V.pt((0, 0, 0)), V.pt(cols[k])))
    # a flag on the first vector, spanned by e_1 and e_2, to make orientation visible
    e1, e2 = cols[0], cols[1]
    flag = [0.55 * e1, 0.95 * e1, 0.95 * e1 + 0.35 * e2, 0.55 * e1 + 0.35 * e2]
    b.append("\\fill[ndGold, fill opacity=0.8] %s -- cycle;" % " -- ".join(V.pt(p) for p in flag))
    return b


def main():
    b = []
    dx, y1, y2 = 3.7, 0.0, -3.6
    rows = [(y1, [np.eye(3), Rx(Q), Ry(Q) @ Rx(Q)], ["start", "$R_x$ (about $x$)", "then $R_y$"]),
            (y2, [np.eye(3), Ry(Q), Rx(Q) @ Ry(Q)], ["start", "$R_y$ (about $y$)", "then $R_x$"])]
    for y, mats, labs in rows:
        for k, (R, lab) in enumerate(zip(mats, labs)):
            b += tripod(R, (k * dx, y))
            b.append("\\node[font=\\small] at (%.2f,%.2f) {%s};" % (k * dx, y - 1.55, lab))
            if k < 2:
                b.append("\\draw[->, >=stealth, line width=0.8pt, ndInk!70] (%.2f,%.2f) -- (%.2f,%.2f)"
                         " node[midway, above, font=\\scriptsize] {$90^\\circ$};"
                         % (k * dx + 1.35, y, (k + 1) * dx - 1.35, y))
    A, B = Ry(Q) @ Rx(Q), Rx(Q) @ Ry(Q)
    assert not np.allclose(A, B)
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {$R_yR_x$};" % (2 * dx + 1.4, y1))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {$R_xR_y$};" % (2 * dx + 1.4, y2))
    b.append("\\node[note, anchor=west] (d) at (%.2f,%.2f) {different end\\\\ positions:\\\\ $R_yR_x\\ne R_xR_y$};"
             % (2 * dx + 1.4, (y1 + y2) / 2))
    b.append("\\node[note] at (%.2f,%.2f) {legend: \\textcolor{ndRed}{$e_1$} (with the gold flag), \\textcolor{ndGreen}{$e_2$}, \\textcolor{ndBlue}{$e_3$}};"
             % (dx, y2 - 2.15))
    N.write("ch02_so3", b)
    print("written figures/ch02_so3.tex")


if __name__ == "__main__":
    main()
