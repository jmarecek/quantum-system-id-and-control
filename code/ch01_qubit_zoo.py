"""ch01_qubit_zoo.py -- what |0> and |1> are on six physical carriers.

A gallery of tiles after the table of physical realisations in the Wikipedia
article "Qubit" (https://en.wikipedia.org/wiki/Qubit); the information is the
table's, the drawings are ours.  Each tile shows the two states side by side:
  * a superconducting circuit (transmon, phase qubit): the ground and first
    excited levels of the well 1 - cos(phi), computed from
    H = 4 E_C n^2 - E_J cos(phi) with E_J/E_C = 10 (a transmon has about 50,
    and a smaller anharmonicity; the cartoon exaggerates it);
  * a superconducting loop (flux qubit): a current clockwise or anticlockwise
    round a loop interrupted by three Josephson junctions;
  * a double quantum dot: one electron in the left or in the right well of
    the potential (x^2 - 1)^2;
  * an electron in a quantum dot: its spin up or down in the field B;
  * a photon: horizontal or vertical polarisation of the beam;
  * an atom held in an optical tweezer: its spin up or down.
Writes figures/ch01_qubit_zoo.tex.
"""
import numpy as np

import needham as N

W, H, GAP = 4.1, 2.75, 0.2                                   # a tile and the space between tiles
EJ_EC = 10.0


def transmon_levels(ej_ec=EJ_EC, n_cut=20, k=3):
    """The lowest k levels of H = 4 E_C n^2 - E_J cos(phi), in units of E_J, measured from the bottom of the well."""
    n = np.arange(-n_cut, n_cut + 1)
    Hm = np.diag(4.0 / ej_ec * n ** 2.0) - 0.5 * (np.eye(len(n), k=1) + np.eye(len(n), k=-1))
    return np.linalg.eigvalsh(Hm)[:k] + 1.0


def tile(b, x0, y0, title):
    b.append("\\filldraw[fill=ndPaper, draw=ndInk!35, line width=0.5pt, rounded corners=3pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0, y0, x0 + W, y0 + H))
    b.append("\\node[font=\\small, anchor=north west] at (%.3f,%.3f) {%s};" % (x0 + 0.08, y0 + H - 0.04, title))


def caption(b, x, y, ket, words):
    b.append("\\node[anchor=north, align=center, inner sep=1pt] at (%.3f,%.3f) {$\\ket{%d}$\\\\[-1pt] {\\footnotesize\\itshape %s}};" % (x, y, ket, words))


def well(b, cx, cy, k, levels):
    """The well 1 - cos(phi) on [-pi, pi], 1.5 wide and 1.0 high, with the levels and a ball on level k."""
    sx, sy = 0.75 / np.pi, 0.5
    phi = np.linspace(-np.pi, np.pi, 160)
    b.append(N.polyline(np.c_[cx + sx * phi, cy - 0.5 + sy * (1 - np.cos(phi))], "ink"))
    for j, e in enumerate(levels):
        half = np.arccos(1 - e) * sx
        style = "ndBlue, line width=0.8pt" if j < 2 else "ndInk!45, line width=0.6pt, dash pattern=on 2pt off 1.5pt"
        b.append("\\draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);" % (style, cx - half, cy - 0.5 + sy * e, cx + half, cy - 0.5 + sy * e))
    b.append("\\shade[ball color=ndBlue!70] (%.3f,%.3f) circle (0.075);" % (cx, cy - 0.5 + sy * levels[k] + 0.08))


def loop(b, cx, cy, clockwise):
    r = 0.45
    b.append("\\draw[ink, line width=1.6pt, ndInk!80] (%.3f,%.3f) circle (%.3f);" % (cx, cy, r))
    for a in (90, 210, 330):
        x, y = cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a))
        b.append("\\fill[ndPaper] (%.3f,%.3f) circle (0.075);" % (x, y))
        b.append("\\draw[ndRed, line width=0.8pt] (%.3f,%.3f) -- ++(0.12,0.12) (%.3f,%.3f) -- ++(0.12,-0.12);" % (x - 0.06, y - 0.06, x - 0.06, y + 0.06))
    a0, a1 = (150, -150) if clockwise else (-150, 150)
    b.append("\\draw[vec, ndBlue, line width=1pt] (%.3f,%.3f) arc (%d:%d:0.26);" % (cx + 0.26 * np.cos(np.radians(a0)), cy + 0.26 * np.sin(np.radians(a0)), a0, a1))


def double_well(b, cx, cy, side):
    x = np.linspace(-1.55, 1.55, 160)
    v = (x ** 2 - 1) ** 2
    b.append(N.polyline(np.c_[cx + 0.5 * x, cy - 0.45 + 0.6 * np.minimum(v, 1.6)], "ink"))
    b.append("\\shade[ball color=ndGreen!80] (%.3f,%.3f) circle (0.09);" % (cx + 0.5 * side, cy - 0.45 + 0.09))


def spin(b, cx, cy, up, ball, field=True):
    """A particle (shaded ball) with its spin arrow; the field B to its left."""
    b.append("\\shade[ball color=%s] (%.3f,%.3f) circle (0.15);" % (ball, cx, cy))
    s = 1 if up else -1
    b.append("\\draw[vec, ndRed, line width=1.1pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (cx, cy - s * 0.4, cx, cy + s * 0.42))
    if field:
        b.append("\\draw[faint, ->, >=stealth, line width=0.6pt, ndInk!55] (%.3f,%.3f) -- (%.3f,%.3f) node[left, font=\\scriptsize, text=ndInk!70] {$B$};" % (cx - 0.55, cy - 0.35, cx - 0.55, cy + 0.35))


def quantum_dot(b, cx, cy, up):
    b.append("\\fill[ndGreen!25] (%.3f,%.3f) ellipse (0.42 and 0.3);" % (cx, cy - 0.05))
    b.append("\\draw[ndGreen!80!ndInk, line width=0.5pt] (%.3f,%.3f) ellipse (0.42 and 0.3);" % (cx, cy - 0.05))
    spin(b, cx, cy - 0.05, up, "ndGreen!80")


def tweezer(b, cx, cy, up):
    """A focused laser beam (an hourglass) holding an atom at its waist."""
    y = np.linspace(-0.55, 0.55, 60)
    w = 0.09 * np.sqrt(1 + (y / 0.18) ** 2)
    pts = np.r_[np.c_[cx - w, cy + y], np.c_[cx + w, cy + y][::-1]]
    b.append("\\fill[ndRose] %s -- cycle;" % " -- ".join(N.pt(p) for p in pts))
    spin(b, cx, cy, up, "ndGold!80", field=False)


def photon(b, cx, cy, angle):
    b.append("\\filldraw[fill=ndGold!25, draw=ndGold!80!ndInk, line width=0.6pt] (%.3f,%.3f) circle (0.42);" % (cx, cy))
    b.append("\\fill[ndGold!80!ndInk] (%.3f,%.3f) circle (0.035);" % (cx, cy))
    dx, dy = 0.36 * np.cos(np.radians(angle)), 0.36 * np.sin(np.radians(angle))
    b.append("\\draw[<->, >=stealth, line width=1.1pt, ndRed] (%.3f,%.3f) -- (%.3f,%.3f);" % (cx - dx, cy - dy, cx + dx, cy + dy))


def main():
    levels = transmon_levels()
    spacing = np.diff(levels)
    print("transmon levels (E_J/E_C = %g, units of E_J): %s; spacings %s" % (EJ_EC, np.round(levels, 3), np.round(spacing, 3)))
    assert spacing[1] < spacing[0] and levels[2] < 2.0           # anharmonic, and the third level still in the well
    tiles = [
        ("transmon: energy", lambda b, x, y, k: well(b, x, y + 0.05, k, levels), ("ground state", "first excited")),
        ("flux qubit: current", lambda b, x, y, k: loop(b, x, y, k == 0), ("clockwise", "anticlockwise")),
        ("double dot: charge", lambda b, x, y, k: double_well(b, x, y, -1 if k == 0 else 1), ("left dot", "right dot")),
        ("quantum dot: spin", lambda b, x, y, k: quantum_dot(b, x + 0.12, y, k == 0), ("up", "down")),
        ("photon: polarisation", lambda b, x, y, k: photon(b, x, y, 0 if k == 0 else 90), ("horizontal", "vertical")),
        ("atom in a tweezer: spin", lambda b, x, y, k: tweezer(b, x, y, k == 0), ("up", "down")),
    ]
    b = []
    for i, (title, draw, words) in enumerate(tiles):
        row, col = divmod(i, 2)
        x0, y0 = col * (W + GAP), (2 - row) * (H + GAP)
        tile(b, x0, y0, title)
        for k in (0, 1):
            cx = x0 + W * (0.27 if k == 0 else 0.73)
            draw(b, cx, y0 + 1.35, k)
            caption(b, cx, y0 + 0.66, k, words[k])
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (x0 + W / 2, y0 + 0.2, x0 + W / 2, y0 + H - 0.5))
    N.write("ch01_qubit_zoo", b)
    print("written figures/ch01_qubit_zoo.tex")


if __name__ == "__main__":
    main()
