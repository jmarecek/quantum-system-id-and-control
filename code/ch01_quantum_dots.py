"""ch01_quantum_dots.py -- a gate-defined double quantum dot, drawn as a cartoon.

For the slide on spin qubits in Lecture 1.  The heterostructure is of the kind
used for hole spin qubits: a strained Ge quantum well between SiGe layers on a
Si substrate (after the stack in the AIP paper cited on the slide), with five
gate electrodes on top: two plungers P_1, P_2, each of which draws one hole
beneath it, and three barriers B_L, B, B_R, which wall the holes in.

Below the stack is the potential energy of a hole along the channel, computed
as the sum of the contributions of the five gates.  A gate is a strip of
width w at height d above the well, and contributes (Davies)
    G(x) = [arctan((x - x_g + w/2)/d) - arctan((x - x_g - w/2)/d)] / pi,
with a negative weight for a plunger and a positive one for a barrier.  The
middle barrier gate sets the height of the wall between the two dots, hence
the tunnel coupling t_c: the solid curve is drawn with a higher wall, the
dashed one with a lower.  To the right, the two spin levels split by
E_Z = g mu_B B in the magnetic field, and the microwave burst u(t) on a
plunger, which turns the spin through the spin-orbit coupling (electric-dipole
spin resonance).
Writes figures/ch01_quantum_dots.tex.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

DEPTH = np.array([0.5, 0.3])                    # the back edge of the stack, seen obliquely
WIDTH = 5.2                                     # the length of the channel drawn, in cm
D = 0.4                                         # height of the gates above the well
GATES = [("B_L", 0.78, 0.46, 0.9), ("P_1", 1.78, 0.7, -1.0), ("B", 2.6, 0.42, 0.75),
         ("P_2", 3.42, 0.7, -1.0), ("B_R", 4.42, 0.46, 0.9)]          # (name, centre, width, weight)
LOW_BARRIER = 0.3                               # the weight of B for the dashed curve
LAYERS = [("Si", 0.4, "ndBlue!22!white"), ("SiGe", 0.7, "ndGreen!20!white"), ("Ge well", 0.3, "ndRed!40!white"),
          ("SiGe", 0.62, "ndGreen!20!white"), ("oxide", 0.1, "ndInk!15!white")]  # bottom to top: (label, thickness, fill)


def strip(x, xg, w, d=D):
    """The potential at the well of a gate strip of width w centred at xg, per unit gate voltage."""
    return (np.arctan((x - xg + w / 2) / d) - np.arctan((x - xg - w / 2) / d)) / np.pi


def potential(x, barrier=None):
    """The potential energy of a hole along the channel; `barrier` replaces the weight of the middle gate."""
    u = np.zeros_like(x)
    for name, xg, w, c in GATES:
        u += (barrier if (name == "B" and barrier is not None) else c) * strip(x, xg, w)
    return u


def minima(x, u):
    i = np.where((u[1:-1] < u[:-2]) & (u[1:-1] < u[2:]))[0] + 1
    return x[i], u[i]


def prism(x0, y0, w, h, fill, top=True):
    """Front face, right face and (optionally) top face of a block seen obliquely."""
    dx, dy = DEPTH
    out = ["\\filldraw[%s, draw=ndInk, line width=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (fill, x0, y0, x0 + w, y0 + h),
           "\\filldraw[%s!80!ndInk, draw=ndInk, line width=0.5pt] (%.3f,%.3f) -- ++(%.3f,%.3f) -- ++(0,%.3f) -- ++(%.3f,%.3f) -- cycle;"
           % (fill, x0 + w, y0, dx, dy, h, -dx, -dy)]
    if top:
        out.append("\\filldraw[%s!55!white, draw=ndInk, line width=0.5pt] (%.3f,%.3f) -- ++(%.3f,%.3f) -- ++(%.3f,0) -- ++(%.3f,%.3f) -- cycle;"
                   % (fill, x0, y0 + h, dx, dy, w, -dx, -dy))
    return out


def burst(x0, x1, y, amp, cycles=8, n=400):
    """A microwave burst: a sine under a Gaussian envelope."""
    t = np.linspace(0.0, 1.0, n)
    return [(x0 + (x1 - x0) * s, y + amp * np.exp(-((s - 0.5) / 0.2) ** 2) * np.sin(2 * np.pi * cycles * s)) for s in t]


def main():
    x = np.linspace(0.0, WIDTH, 1201)
    u_hi, u_lo = potential(x), potential(x, LOW_BARRIER)
    xm, um = minima(x, u_hi)
    assert len(xm) == 2 and abs(xm[0] - GATES[1][1]) < 0.1 and abs(xm[1] - GATES[3][1]) < 0.1      # one dot under each plunger
    mid = (x > xm[0]) & (x < xm[1])
    assert u_lo[mid].max() < u_hi[mid].max() - 0.1                                                # the lower barrier gate, the lower wall
    assert len(minima(x, u_lo)[0]) == 2                                                             # still two dots
    for side in (x < xm[0], x > xm[1]):
        assert u_hi[side].max() > um.max() + 0.25                                                  # the outer barriers confine
    b = []
    # ---- the potential energy along the channel
    Pl = N.Plot((0.0, WIDTH), (-0.5, 0.35), WIDTH, 1.9, origin=(0.0, 0.0))
    b += Pl.axes("$x$", "")
    b.append("\\node[font=\\scriptsize, anchor=south west] at %s {energy of a hole};" % Pl.pt(0.05, 0.4))
    b.append(Pl.curve(x, u_lo, "dashedink"))
    b.append(Pl.curve(x, u_hi, "ink"))
    level = um + 0.07
    for xc, lv in zip(xm, level):
        b.append("\\draw[ndInk, line width=0.6pt] %s -- %s;" % (Pl.pt(xc - 0.3, lv), Pl.pt(xc + 0.3, lv)))
    spins = [(xm[0], level[0], 1), (xm[1], level[1], -1)]
    for xc, lv, s in spins:
        p = Pl.P(xc, lv)
        b.append("\\shade[ball color=ndRed!75] (%.3f,%.3f) circle (0.085);" % (p[0], p[1] + 0.085))
        b.append("\\draw[vec, line width=0.9pt, ndRed] (%.3f,%.3f) -- (%.3f,%.3f);" % (p[0] + 0.17, p[1] + 0.085 - 0.2 * s, p[0] + 0.17, p[1] + 0.085 + 0.2 * s))
    xb = x[mid][np.argmax(u_hi[mid])]
    lv = level.mean()
    b.append("\\draw[<->, >=stealth, line width=0.5pt, ndBlue] %s -- %s node[midway, below=-1pt, font=\\scriptsize, text=ndBlue] {$t_c$};"
             % (Pl.pt(xb - 0.38, lv + 0.11), Pl.pt(xb + 0.38, lv + 0.11)))
    b.append("\\node[note, anchor=south west, align=left] (bn) at %s {the gate $B$ sets the wall\\\\ (dashed: lower), and so\\\\ the tunnel coupling $t_c$};" % Pl.pt(2.95, 0.27))
    b.append("\\draw[pointer] (bn.west) to[bend right=20] %s;" % Pl.pt(xb + 0.06, potential(np.array([xb + 0.06]))[0] + 0.02))
    # ---- the two spin levels, to the right
    xz, y0, y1 = 5.95, 0.3, 1.2
    b.append("\\draw[ink] (%.3f,%.3f) -- ++(0.8,0) node[right, font=\\scriptsize] {$\\ket{\\downarrow}$};" % (xz, y0))
    b.append("\\draw[ink] (%.3f,%.3f) -- ++(0.8,0) node[right, font=\\scriptsize] {$\\ket{\\uparrow}$};" % (xz, y1))
    b.append("\\draw[<->, >=stealth, faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (xz + 0.62, y0 + 0.04, xz + 0.62, y1 - 0.04))
    wig = [(xz + 0.25 + 0.06 * np.sin(2 * np.pi * 5 * s), y0 + 0.05 + (y1 - y0 - 0.2) * s) for s in np.linspace(0, 1, 200)]
    b.append(N.polyline(wig, "line width=0.8pt, ndRed"))
    b.append("\\draw[->, >=stealth, line width=0.8pt, ndRed] (%.3f,%.3f) -- (%.3f,%.3f);" % (wig[-1][0], wig[-1][1], xz + 0.25, y1 - 0.03))
    b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,%.3f) {$E_Z=g\\mu_B B$};" % (xz + 0.45, y1 + 0.12))
    # ---- the heterostructure
    y = 3.0
    tops = {}
    for k, (lab, h, fill) in enumerate(LAYERS):
        b += prism(0.0, y, WIDTH, h, fill, top=(k == len(LAYERS) - 1))
        tops[lab] = (y, h)
        if lab != "oxide":
            b.append("\\node[font=\\scriptsize, anchor=east, inner sep=0pt] at (%.3f,%.3f) {%s};" % (WIDTH - 0.1, y + h / 2, lab))
        y += h
    yw, hw = tops["Ge well"]
    for xc, _, s in spins:
        b.append("\\shade[ball color=ndRed!75] (%.3f,%.3f) circle (0.085);" % (xc, yw + hw / 2))
        b.append("\\draw[vec, line width=0.9pt, ndRed] (%.3f,%.3f) -- (%.3f,%.3f);" % (xc + 0.17, yw + hw / 2 - 0.2 * s, xc + 0.17, yw + hw / 2 + 0.2 * s))
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {one hole in each dot};" % ((xm[0] + xm[1]) / 2, yw - 0.1))
    gy, gh = y, 0.3
    for name, xg, w, c in GATES:
        b += prism(xg - w / 2, gy, w, gh, "ndGold!60!white")
        b.append("\\node[font=\\scriptsize, inner sep=0pt] at (%.3f,%.3f) {$%s$};" % (xg, gy + gh / 2, name))
    # the drive on the first plunger, and the field
    xp = GATES[1][1]
    by = gy + gh + DEPTH[1] + 0.55
    b.append(N.polyline(burst(xp - 1.0, xp + 0.6, by, 0.24), "line width=0.7pt, ndRed"))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=east] at (%.3f,%.3f) {$u(t)$};" % (xp - 1.05, by))
    b.append("\\draw[pointer] (%.3f,%.3f) -- (%.3f,%.3f);" % (xp + 0.15, by - 0.28, xp + DEPTH[0] / 2, gy + gh + DEPTH[1] / 2))
    b.append("\\node[note, anchor=west, align=left] at (%.3f,%.3f) {a microwave burst on $P_1$\\\\ turns the spin};" % (xp + 0.7, by + 0.05))
    b.append("\\draw[vec, ndBlue] (%.3f,%.3f) -- ++(0.9,0) node[right, text=ndBlue, font=\\scriptsize] {$\\mathbf B$};" % (WIDTH + 0.25, by + 0.05))
    N.write("ch01_quantum_dots", b)
    print("dots at x = %.2f, %.2f; wall %.3f (higher) and %.3f (lower) above the dots"
          % (xm[0], xm[1], u_hi[mid].max() - um.max(), u_lo[mid].max() - minima(x, u_lo)[1].max()))


if __name__ == "__main__":
    main()
