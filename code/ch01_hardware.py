"""ch01_hardware.py -- a superconducting quantum computer, from the outside in.

A cartoon in place of the photographs of earlier years:
  * the dilution refrigerator, drawn as its stack of plates at about 300 K,
    50 K, 4 K, 1 K (the still), 100 mK (the cold plate) and 10 mK (the mixing
    chamber), with the coaxial lines that carry the pulses down through
    attenuators (squares) and the readout signal up through amplifiers
    (triangles), and the sample holder below the mixing chamber;
  * a zoom on the circuit board in the sample holder, with the qubit chip at
    its centre, and a zoom on the chip: four transmons (two pads and a
    junction each), the couplers between neighbours, and a readout resonator
    for each qubit on a common feed line;
  * the room-temperature electronics beside the refrigerator: signal sources,
    FPGA-based pulse shaping, which makes the pulse u(t) (a microwave burst
    under a Gaussian envelope, drawn in red), and digitisers for the readout.
Writes figures/ch01_hardware.tex.
"""
import numpy as np

import needham as N

CX = 3.95                                                   # axis of the refrigerator
PLATES = [("300\\,K", 7.75, 1.75), ("50\\,K", 7.05, 1.62), ("4\\,K", 6.35, 1.50),
          ("1\\,K", 5.65, 1.36), ("100\\,mK", 5.0, 1.24), ("10\\,mK", 4.35, 1.14)]
T_PLATE, RY = 0.11, 0.10                                    # thickness and half-height of a plate
OUT_X = CX - 0.55                                           # the readout line
IN_X = (CX - 0.15, CX + 0.1, CX + 0.35)                     # the drive and flux lines
BOX = (CX - 0.62, CX + 0.62, 3.3, 3.68)                     # the sample holder
RACK = (6.72, 9.45, 5.25, 8.0)


def plate(x, y, r, fill="ndGold!40", side="ndGold!75"):
    """A plate seen slightly from above: a flat cylinder of radius r with its top at height y."""
    return ["\\fill[%s] (%.3f,%.3f) -- (%.3f,%.3f) arc (180:360:%.3f and %.3f) -- (%.3f,%.3f) -- cycle;"
            % (side, x - r, y, x - r, y - T_PLATE, r, RY, x + r, y),
            "\\draw[ndInk, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f) arc (180:360:%.3f and %.3f) -- (%.3f,%.3f);"
            % (x - r, y, x - r, y - T_PLATE, r, RY, x + r, y),
            "\\filldraw[fill=%s, draw=ndInk, line width=0.5pt] (%.3f,%.3f) ellipse (%.3f and %.3f);" % (fill, x, y, r, RY)]


def coax(x, y0, y1, colour, wiggle=0.045):
    """A coaxial line from y0 down to y1, with a gentle wiggle between the plates."""
    y = np.linspace(y0, y1, 120)
    s = (y0 - y) / (y0 - y1)
    return N.polyline(np.c_[x + wiggle * np.sin(9 * np.pi * s) * np.sin(np.pi * s), y], "%s, line width=0.7pt" % colour)


def attenuator(x, y):
    return "\\filldraw[fill=white, draw=ndInk, line width=0.45pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x - 0.075, y - 0.065, x + 0.075, y + 0.065)


def amplifier(x, y):
    """A triangle pointing up: the signal travels upwards."""
    return "\\filldraw[fill=ndGreen!35, draw=ndInk, line width=0.45pt] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- cycle;" % (x - 0.1, y - 0.08, x + 0.1, y - 0.08, x, y + 0.1)


def burst(x0, x1, y, h, cycles=9):
    """A microwave burst under a Gaussian envelope, from x0 to x1, centred at height y."""
    s = np.linspace(0, 1, 260)
    env = np.exp(-((s - 0.5) / 0.17) ** 2)
    return np.c_[x0 + (x1 - x0) * s, y + h * env * np.sin(2 * np.pi * cycles * s)]


def transmon(b, x, y, w=0.36, h=0.22, colour="ndBlue"):
    """Two pads and the junction between them, centred at (x, y)."""
    g = 0.07
    b.append("\\filldraw[fill=%s!35, draw=ndInk, line width=0.45pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (colour, x - w / 2, y - h / 2, x - g / 2, y + h / 2))
    b.append("\\filldraw[fill=%s!35, draw=ndInk, line width=0.45pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (colour, x + g / 2, y - h / 2, x + w / 2, y + h / 2))
    b.append("\\draw[ndRed, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
             % (x - 0.035, y - 0.035, x + 0.035, y + 0.035, x - 0.035, y + 0.035, x + 0.035, y - 0.035))


def meander(x, y0, y1, width=0.16, turns=4):
    """A meandering resonator from (x, y0) to (x, y1)."""
    pts = [(x, y0)]
    ys = np.linspace(y0, y1, 2 * turns + 1)
    for k in range(1, len(ys)):
        side = width / 2 if k % 2 else -width / 2
        pts += [(x + side, ys[k - 1]), (x + side, ys[k])]
    pts.append((x, y1))
    return N.polyline(pts, "ndGreen, line width=0.55pt")


def main():
    b = []
    # ---- the refrigerator: support rods, lines, then the plates over them
    for (_, y0, r0), (_, y1, r1) in zip(PLATES[:-1], PLATES[1:]):
        for sgn in (-1, 1):
            b.append("\\draw[ndInk!45, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (CX + sgn * (r1 - 0.12), y0 - T_PLATE, CX + sgn * (r1 - 0.12), y1))
    for sgn in (-1, 1):                                                  # the sample holder hangs from the mixing chamber
        b.append("\\draw[ndGold!70!ndInk, line width=1.2pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (CX + sgn * 0.55, PLATES[-1][1] - T_PLATE, CX + sgn * 0.55, BOX[3]))
    top, bottom = PLATES[0][1] + 0.6, BOX[3]
    b.append(coax(OUT_X, top, bottom, "ndGreen"))
    for x in IN_X:
        b.append(coax(x, top, bottom, "ndBlue"))
    for name, y, r in PLATES:
        b += plate(CX, y, r)
        b.append("\\node[font=\\footnotesize, anchor=west] at (%.3f,%.3f) {%s};" % (CX + r + 0.06, y - T_PLATE / 2, name))
    for _, y, _ in (PLATES[2], PLATES[3], PLATES[5]):                   # attenuators on the way down
        b += [attenuator(x, y - T_PLATE - RY - 0.12) for x in IN_X]
    for _, y, _ in (PLATES[2], PLATES[5]):                              # amplifiers on the way up
        b.append(amplifier(OUT_X, y - T_PLATE - RY - 0.15))
    b.append("\\filldraw[fill=ndInk!12, draw=ndInk, line width=0.6pt, rounded corners=1pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (BOX[0], BOX[2], BOX[1], BOX[3]))
    b.append("\\fill[ndGreen!45] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (BOX[0] + 0.12, BOX[2] + 0.08, BOX[1] - 0.12, BOX[3] - 0.08))
    b.append("\\fill[ndSky!80!ndInk] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (CX - 0.07, BOX[2] + 0.13, CX + 0.07, BOX[3] - 0.13))
    b.append("\\node[font=\\small\\bfseries, anchor=north west, align=left] at (0.0,9.45) {dilution cryostat,\\\\ about 10\\,mK};")
    b.append("\\node[note, font=\\small\\itshape, anchor=east] (att) at (2.2,6.6) {attenuators\\\\ on the way\\\\ down};")
    b.append("\\draw[pointer] (att.east) to[bend left=10] (%.3f,%.3f);" % (IN_X[0] - 0.09, PLATES[2][1] - T_PLATE - RY - 0.12))
    b.append("\\node[note, font=\\small\\itshape, anchor=east] (wir) at (2.4,5.15) {wiring and\\\\ filtering stages\\\\ (fragile\\\\ below 1\\,K)};")
    b.append("\\draw[pointer] (wir.east) to[bend left=10] (%.3f,%.3f);" % (OUT_X - 0.07, 5.3))
    b.append("\\node[note, font=\\small\\itshape, anchor=east] (amp) at (2.55,3.95) {amplifiers on\\\\ the way up};")
    b.append("\\draw[pointer] (amp.east) to[bend right=10] (%.3f,%.3f);" % (OUT_X - 0.11, PLATES[5][1] - T_PLATE - RY - 0.15))

    # ---- the electronics, and the cables to the top of the refrigerator
    x0, x1, y0, y1 = RACK
    b.append("\\filldraw[fill=ndInk!8, draw=ndInk, line width=0.8pt, rounded corners=2pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0, y0, x1, y1))
    units = [("signal\\\\ sources", "ndBlue"), ("FPGA pulse\\\\ shaping", "ndRed"), ("digitisers", "ndGreen")]
    h = (y1 - y0 - 0.12 * 4) / 3
    centres = []
    for k, (name, col) in enumerate(units):
        ua, ub = y1 - 0.12 - k * (h + 0.12) - h, y1 - 0.12 - k * (h + 0.12)
        centres.append((ua + ub) / 2)
        b.append("\\filldraw[fill=ndPaper, draw=ndInk, line width=0.5pt, rounded corners=1pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0 + 0.1, ua, x1 - 0.1, ub))
        b.append("\\node[font=\\footnotesize, anchor=west, align=left, text=%s] at (%.3f,%.3f) {%s};" % (col if col != "ndBlue" else "ndInk", x0 + 0.16, (ua + ub) / 2, name))
        sx0, sx1 = x1 - 0.7, x1 - 0.16                                      # a small screen on each unit
        b.append("\\filldraw[fill=white, draw=ndInk, line width=0.4pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (sx0, ua + 0.14, sx1, ub - 0.14))
        mid, amp = (ua + ub) / 2, (ub - ua - 0.34) / 2
        if k == 0:
            s = np.linspace(0, 1, 120)
            b.append(N.polyline(np.c_[sx0 + 0.05 + (sx1 - sx0 - 0.1) * s, mid + amp * np.sin(2 * np.pi * 4 * s)], "ndBlue, line width=0.6pt"))
        elif k == 1:
            b.append(N.polyline(burst(sx0 + 0.05, sx1 - 0.05, mid, amp, cycles=7), "ndRed, line width=0.7pt"))
        else:
            s = np.linspace(0, 1, 120)
            b.append(N.polyline(np.c_[sx0 + 0.05 + (sx1 - sx0 - 0.1) * s, mid + amp * np.exp(-3 * s) * np.cos(2 * np.pi * 3 * s)], "ndGreen, line width=0.6pt"))
    flange = PLATES[0][1] + RY
    exits = [(x0 + 0.55, IN_X[2], 8.3, "ndBlue"), (x0 + 0.85, IN_X[1], 8.45, "ndBlue"), (x0 + 1.15, IN_X[0], 8.6, "ndBlue"), (x0 + 1.45, OUT_X, 8.75, "ndGreen")]
    for xr, xl, yc, col in exits:
        b.append("\\draw[%s, line width=0.7pt, rounded corners=4pt] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f);" % (col, xr, y1, xr, yc, xl, yc, xl, flange))
    b.append(N.polyline(burst(4.55, 6.3, 9.15, 0.28), "ndRed, line width=0.9pt"))
    b.append("\\node[font=\\small, text=ndRed, anchor=west] at (6.35,9.15) {$u(t)$};")
    b.append("\\node[font=\\small, anchor=north, align=center, text width=%.2fcm] at (%.3f,%.3f) {room-temperature electronics --\\\\ \\textcolor{ndRed}{this is where\\\\ $u(t)$ is made}};"
             % (x1 - x0 + 0.1, (x0 + x1) / 2, y0 - 0.08))

    # ---- zoom 1: the circuit board in the sample holder
    bx0, bx1, by0, by1 = 0.0, 3.3, -0.3, 2.55
    bc = ((bx0 + bx1) / 2, (by0 + by1) / 2)
    for (px, py), (qx, qy) in [((BOX[0], BOX[2]), (bx0, by1)), ((BOX[1], BOX[2]), (bx1, by1))]:
        b.append("\\draw[ndGold!80!ndInk, dashed, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (px, py, qx, qy))
    b.append("\\filldraw[fill=ndPaper, draw=ndGold!80!ndInk, line width=1pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (bx0, by0, bx1, by1))
    pr = 1.05                                                            # the board: a green square with connectors round its edge
    b.append("\\filldraw[fill=ndGreen!40, draw=ndInk, line width=0.6pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (bc[0] - pr, bc[1] - pr, bc[0] + pr, bc[1] + pr))
    cs = 0.28
    for k in range(16):
        side, t = divmod(k, 4)
        u = -pr + 0.3 + t * (2 * pr - 0.6) / 3
        p = [(u, pr - 0.17), (pr - 0.17, -u), (-u, -pr + 0.17), (-pr + 0.17, u)][side]
        q = [(u * 0.25, cs), (cs, -u * 0.25), (-u * 0.25, -cs), (-cs, u * 0.25)][side]
        b.append("\\draw[ndGold!85!ndInk, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (bc[0] + p[0], bc[1] + p[1], bc[0] + q[0], bc[1] + q[1]))
        b.append("\\filldraw[fill=ndGold!60, draw=ndInk, line width=0.35pt] (%.3f,%.3f) circle (0.075);" % (bc[0] + p[0], bc[1] + p[1]))
    b.append("\\filldraw[fill=ndSky!80!ndInk, draw=ndInk, line width=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (bc[0] - cs + 0.04, bc[1] - cs + 0.04, bc[0] + cs - 0.04, bc[1] + cs - 0.04))
    b.append("\\node[font=\\small, text=ndGold!80!ndInk, anchor=north] at (%.3f,%.3f) {circuit board (zoom)};" % (bc[0], by0 - 0.04))

    # ---- zoom 2: the qubit chip
    qx0, qx1, qy0, qy1 = 4.15, 9.45, -0.3, 2.55
    for (px, py), (qx, qy) in [((bc[0] + cs - 0.04, bc[1] + cs - 0.04), (qx0, qy1)), ((bc[0] + cs - 0.04, bc[1] - cs + 0.04), (qx0, qy0))]:
        b.append("\\draw[ndBlue, dashed, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (px, py, qx, qy))
    b.append("\\filldraw[fill=ndSky!45, draw=ndBlue, line width=1pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (qx0, qy0, qx1, qy1))
    feed = qy0 + 0.55                                                    # the common feed line of the readout
    b.append("\\draw[ndGreen, line width=0.9pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (qx0 + 0.15, feed, qx1 - 0.15, feed))
    qubits = [(4.95 + 1.25 * k, 1.75) for k in range(4)]
    for (xa, ya), (xb, yb) in zip(qubits[:-1], qubits[1:]):
        b.append("\\draw[ndInk, line width=1.1pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xa + 0.27, ya, xb - 0.27, yb))
        xm = (xa + xb) / 2
        b.append("\\filldraw[fill=ndGold!55, draw=ndInk, line width=0.4pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (xm - 0.09, ya - 0.09, xm + 0.09, ya + 0.09))
    for x, y in qubits:
        b.append(meander(x, y - 0.16, feed, width=0.2, turns=3))
        transmon(b, x, y, w=0.54, h=0.32)
    b.append("\\node[note, anchor=south, font=\\small\\itshape] (q) at (%.3f,2.08) {qubits};" % ((qubits[0][0] + qubits[1][0]) / 2))
    b.append("\\draw[pointer] (q.west) to[bend right=20] (%.3f,%.3f);" % (qubits[0][0], qubits[0][1] + 0.17))
    b.append("\\draw[pointer] (q.east) to[bend left=20] (%.3f,%.3f);" % (qubits[1][0], qubits[1][1] + 0.17))
    xc = [(a[0] + c[0]) / 2 for a, c in zip(qubits[:-1], qubits[1:])]
    b.append("\\node[note, anchor=south, font=\\small\\itshape] (c) at (%.3f,2.08) {couplers};" % ((xc[1] + xc[2]) / 2))
    b.append("\\draw[pointer] (c.west) to[bend right=20] (%.3f,%.3f);" % (xc[1], qubits[0][1] + 0.1))
    b.append("\\draw[pointer] (c.east) to[bend left=20] (%.3f,%.3f);" % (xc[2], qubits[0][1] + 0.1))
    b.append("\\node[note, anchor=north, font=\\small\\itshape] (r) at (%.3f,%.3f) {readout resonators on a feed line};" % ((qx0 + qx1) / 2, feed - 0.04))
    b.append("\\draw[pointer] (%.3f,%.3f) to[bend right=25] (%.3f,%.3f);" % (xc[2] + 0.75, feed - 0.12, qubits[3][0] + 0.1, feed + 0.45))
    b.append("\\node[font=\\small, text=ndBlue, anchor=north] at (%.3f,%.3f) {qubit chip (zoom)};" % ((qx0 + qx1) / 2, qy0 - 0.04))
    N.write("ch01_hardware", b)
    print("written figures/ch01_hardware.tex")


if __name__ == "__main__":
    main()
