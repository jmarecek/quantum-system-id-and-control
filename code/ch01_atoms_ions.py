"""ch01_atoms_ions.py -- neutral atoms in tweezers and ions in a trap, drawn as cartoons.

For the slide on neutral atoms and ions in Lecture 1.
  (a) Neutral atoms, after the processor of Bluvstein et al. (Nature 604, 451,
      2022): a plane of optical tweezers seen obliquely, with a storage zone
      at the back (a few tweezers empty) and an entangling zone at the front,
      lit by the Rydberg laser.  Moving tweezers carry atoms from one zone to
      the other, and the atoms of a pair sit within the blockade radius of
      each other.  The levels: the hyperfine qubit |0>, |1>, which stores the
      information, and the Rydberg level |r>, which entangles; u(t) is the
      laser pulse.
  (b) Ions in a linear Paul trap: four rods carrying the radio-frequency
      voltage and two end caps, with a chain of five ions on the axis.  The
      positions of the ions are the equilibrium of the harmonic trap and the
      Coulomb repulsion, i.e. the minimum of
          sum_k z_k^2 / 2 + sum_{j<k} 1 / |z_j - z_k|
      in units of (e^2 / (4 pi eps_0 m nu^2))^{1/3}, and are closer together
      in the middle of the chain.  Two ions are addressed by focused beams;
      the springs stand for the shared vibration that couples them.
Writes figures/ch01_atoms_ions.tex.
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

SHEAR, LIFT = 0.55, 0.5                         # the oblique view of the plane of tweezers
PLANE = (3.9, 3.4)                              # its extent (X, Y) in cm
BASE_A = 4.1                                    # page height of the front edge of the plane
IONS = 5


def plane(X, Y):
    return np.array([X + SHEAR * Y, BASE_A + LIFT * Y])


def atom(p, r=0.085, colour="ndGreen!75"):
    return "\\shade[ball color=%s] (%.3f,%.3f) circle (%.3f);" % (colour, p[0], p[1], r)


def empty(p, r=0.075):
    return "\\draw[ndInk!45, line width=0.4pt] (%.3f,%.3f) circle (%.3f);" % (p[0], p[1], r)


def chain(n):
    """Equilibrium positions of n ions in a harmonic trap, in the natural unit of length."""
    def energy(z):
        d = np.abs(z[:, None] - z[None, :])[np.triu_indices(n, 1)]
        return 0.5 * np.sum(z ** 2) + np.sum(1.0 / d)
    res = minimize(energy, np.linspace(-1.5, 1.5, n), method="BFGS", options={"gtol": 1e-10})
    return np.sort(res.x)


def levels(b, x0, y0, rows, width=0.6):
    """Horizontal level lines with kets on the left: rows = [(height, ket)]."""
    for h, k in rows:
        b.append("\\draw[ink] (%.3f,%.3f) -- ++(%.3f,0);" % (x0, y0 + h, width))
        b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {$\\ket{%s}$};" % (x0 - 0.02, y0 + h, k))


def wiggle(x, y0, y1, colour, cycles=4.5, amp=0.06):
    pts = [(x + amp * np.sin(2 * np.pi * cycles * s), y0 + (y1 - y0 - 0.12) * s) for s in np.linspace(0, 1, 160)]
    return [N.polyline(pts, "line width=0.8pt, %s" % colour),
            "\\draw[->, >=stealth, line width=0.8pt, %s] (%.3f,%.3f) -- (%.3f,%.3f);" % (colour, pts[-1][0], pts[-1][1], x, y1)]


def main():
    z = chain(IONS)
    assert np.allclose(z, -z[::-1], atol=1e-6)                       # symmetric about the centre of the trap
    gaps = np.diff(z)
    assert gaps[len(gaps) // 2] < gaps[0]                             # closer together in the middle
    assert abs(z[-1] - 1.7429) < 2e-3 and abs(z[-2] - 0.8221) < 2e-3  # the known equilibrium of five ions
    b = []
    # ---- (a) the plane of tweezers
    X, Y = PLANE
    corners = [plane(0, 0), plane(X, 0), plane(X, Y), plane(0, Y)]
    b.append("\\fill[ndSky, fill opacity=0.5] %s -- cycle;" % " -- ".join(N.pt(c) for c in corners))
    band = [plane(0, 0.25), plane(X, 0.25), plane(X, 1.35), plane(0, 1.35)]
    b.append("\\fill[ndRose, fill opacity=0.9] %s -- cycle;" % " -- ".join(N.pt(c) for c in band))
    b.append("\\draw[ink, line width=0.5pt] %s -- cycle;" % " -- ".join(N.pt(c) for c in corners))
    # the storage zone: two rows of tweezers, a few of them empty
    gone = {(2, 3.0), (6, 3.0), (5, 2.4), (3, 2.4)}
    for row in (3.0, 2.4):
        for k in range(8):
            p = plane(0.3 + 0.48 * k, row)
            b.append(empty(p) if (k, row) in gone else atom(p))
    # the entangling zone: pairs within the blockade radius; one pair still being filled
    pairs = [0.45, 1.22, 1.99, 2.76, 3.5]
    arriving = 1
    for k, xc in enumerate(pairs):
        b.append(atom(plane(xc - 0.12, 0.8)))
        b.append(empty(plane(xc + 0.12, 0.8)) if k == arriving else atom(plane(xc + 0.12, 0.8)))
    # the blockade radius about one pair
    t = np.linspace(0, 2 * np.pi, 120)
    ring = [plane(pairs[3] + 0.3 * np.cos(s), 0.8 + 0.3 * np.sin(s)) for s in t]
    b.append(N.polyline(ring, "dashedink, ndInk!80"))
    # a moving tweezer carries an atom from the storage zone (site (3, 2.4), now empty) to the pair
    src, dst = np.array([0.3 + 0.48 * 3, 2.4]), np.array([pairs[arriving] + 0.12, 0.8])
    path = [plane(*(src + (dst - src) * s + np.array([0.35 * np.sin(np.pi * s), 0.0]))) for s in np.linspace(0, 1, 60)]
    b.append(N.polyline(path[3:-3], "->, >=stealth, line width=0.7pt, ndRed"))
    q = plane(*(src + (dst - src) * 0.5 + np.array([0.35, 0.0])))
    b.append("\\fill[ndRed, fill opacity=0.25] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- cycle;" % (q[0] - 0.16, q[1] + 1.05, q[0] + 0.16, q[1] + 1.05, q[0], q[1]))
    b.append(atom(q))
    b.append("\\node[note, anchor=south, align=center] (mv) at %s {a moving tweezer\\\\ carries an atom};" % N.pt(q + np.array([-0.85, 1.0])))
    b.append("\\draw[pointer] (mv.south east) to[bend left=10] (%.3f,%.3f);" % (q[0] - 0.15, q[1] + 0.85))
    # labels of the zones
    b.append("\\node[note, anchor=south] at %s {storage zone};" % N.pt(plane(X * 0.62, Y) + np.array([0.0, 0.03])))
    b.append("\\node[note, anchor=north, align=center] at %s {entangling zone,\\\\ lit by the Rydberg laser};" % N.pt(plane(X / 2, 0) + np.array([0.0, -0.04])))
    bl = plane(pairs[3] + 0.3, 0.8 - 0.3)
    b.append("\\node[note, anchor=north west, align=left] (br) at %s {blockade\\\\ radius};" % N.pt(np.array([plane(X, 0)[0] + 0.12, BASE_A + 0.05])))
    b.append("\\draw[pointer] (br.west) to[bend left=20] %s;" % N.pt(bl))
    # the levels of an atom
    lx, ly = 6.15, BASE_A + 0.25
    levels(b, lx, ly, [(0.0, "0"), (0.3, "1"), (1.15, "r")])
    b += wiggle(lx + 0.3, ly + 0.33, ly + 1.12, "ndRed")
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at (%.3f,%.3f) {$u(t)$};" % (lx + 0.4, ly + 0.74))
    b.append("\\node[note, anchor=west] at (%.3f,%.3f) {Rydberg};" % (lx + 0.65, ly + 1.15))
    b.append("\\node[note, anchor=west] at (%.3f,%.3f) {qubit};" % (lx + 0.65, ly + 0.15))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) neutral atoms in optical tweezers};" % (3.4, BASE_A - 1.05))
    # ---- (b) the linear Paul trap
    x0, x1, yc, ry = 0.3, 5.0, 1.1, 0.5
    xm = (x0 + x1) / 2
    for dy, shade in ((ry, "ndGold!30!white"), (-ry, "ndGold!30!white")):                    # the back rods
        b.append("\\filldraw[%s, draw=ndInk!60, line width=0.4pt, rounded corners=0.9mm] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                 % (shade, x0 + 0.3, yc + dy + 0.12 - 0.09, x1 + 0.3, yc + dy + 0.12 + 0.09))
    for xe in (x0 + 0.35, x1 - 0.05):                                                         # the end caps
        b.append("\\draw[ndGold!80!ndInk, line width=1.6pt] (%.3f,%.3f) ellipse (0.13 and 0.42);" % (xe, yc))
    scale = 0.62
    ions = [np.array([xm + scale * zk, yc]) for zk in z]
    for p, q in zip(ions[:-1], ions[1:]):                                                     # the springs
        s = np.linspace(0, 1, 13)
        pts = [(p[0] + 0.13 + (q[0] - p[0] - 0.26) * u, yc + (0.07 * (-1) ** i if 0 < i < len(s) - 1 else 0.0)) for i, u in enumerate(s)]
        b.append(N.polyline(pts, "line width=0.5pt, ndInk!55"))
    for p in ions:
        b.append(atom(p, r=0.11, colour="ndBlue!70"))
    for dy in (ry, -ry):                                                                      # the front rods
        b.append("\\filldraw[ndGold!60!white, draw=ndInk, line width=0.5pt, rounded corners=0.9mm] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                 % (x0, yc + dy - 0.1, x1, yc + dy + 0.1))
    for k, p in zip((1, 3), (ions[1], ions[3])):                                              # the addressing beams
        b.append("\\fill[ndRed, fill opacity=0.3] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- cycle;" % (p[0] - 0.14, yc + 1.25, p[0] + 0.14, yc + 1.25, p[0], p[1] + 0.12))
        b.append("\\node[font=\\scriptsize, text=ndRed, anchor=south] at (%.3f,%.3f) {$u_{%d}(t)$};" % (p[0], yc + 1.25, k + 1))
    b.append("\\node[note, anchor=north, align=center] (sp) at (%.3f,%.3f) {the ions repel one another;\\\\ a shared vibration couples them};" % (xm, yc - ry - 0.18))
    b.append("\\draw[pointer] (sp.north) -- (%.3f,%.3f);" % ((ions[2][0] + ions[3][0]) / 2, yc - 0.1))
    b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {rf};" % (x0 - 0.03, yc + ry))
    b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {rf};" % (x0 - 0.03, yc - ry))
    # the levels of an ion
    lx, ly = 6.15, yc - 0.45
    levels(b, lx, ly, [(0.0, "0"), (0.9, "1")])
    b += wiggle(lx + 0.3, ly + 0.03, ly + 0.87, "ndRed")
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at (%.3f,%.3f) {$u_k(t)$};" % (lx + 0.4, ly + 0.45))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) ions in a linear Paul trap};" % (3.4, -0.55))
    N.write("ch01_atoms_ions", b)
    print("equilibrium of %d ions:" % IONS, np.round(z, 4))


if __name__ == "__main__":
    main()
