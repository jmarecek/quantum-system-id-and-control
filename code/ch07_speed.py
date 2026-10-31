"""ch07_speed.py -- the length of a trajectory and the distance it covers.

The driven qubit of Chapter 4, H = -(omega/2) sigma_z + (u/2) sigma_x with
omega = 1, started from |0> with the constant control u = 1.5.  Its Bloch vector
turns about h = (u, 0, -omega) on a small circle.  The Fubini--Study distance
from |0> is d(t) = arccos |<0|psi(t)>|, half the polar angle of the Bloch
vector; the Fubini--Study length of the path is L(t) = int_0^t Delta H, and
since Delta H^2 = (|h|^2 - (h.r)^2)/4 and h.r is constant on the circle,
Delta H = u/2 throughout.  The farthest point, reached at t = pi/|h|, is at
d = 0.983 while L = 1.307 there.  The resonant pulse without drift (omega = 0,
the same u) runs along a meridian, a geodesic, and L = d = u t/2 until it
reaches |1> at t = pi/u.  L >= d always: the length of a path is at least the
distance between its ends (Mandelstam--Tamm, Anandan--Aharonov).
Writes figures/ch07_speed.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N
import ch04_examples as E                      # OMEGA = 1

I2, X, Y, Z = T.pauli()
U = 1.5


def path(omega, u, ts):
    """States psi(t) from |0> under H = -(omega/2) Z + (u/2) X."""
    H = -omega / 2 * Z + u / 2 * X
    return np.array([expm(-1j * H * t) @ np.array([1, 0], complex) for t in ts]), H


def main():
    hd = np.array([U, 0.0, -E.OMEGA])
    th = np.pi / np.linalg.norm(hd)
    ts = np.linspace(0, 2 * th, 701)
    out = {}
    for name, om in (("driven", E.OMEGA), ("resonant", 0.0)):
        psi, H = path(om, U, ts)
        r = np.array([T.bloch_vector(np.outer(p, p.conj())) for p in psi])
        dH = np.sqrt(np.real([p.conj() @ H @ H @ p - (p.conj() @ H @ p) ** 2 for p in psi]))
        assert np.allclose(dH, U / 2)                                       # constant speed u/2
        d = np.arccos(np.clip(np.abs(psi[:, 0]), 0, 1))
        assert np.allclose(d, np.arccos(np.clip(r[:, 2], -1, 1)) / 2)       # half the polar angle
        L = np.concatenate([[0], np.cumsum(0.5 * (dH[1:] + dH[:-1]) * np.diff(ts))])
        assert np.all(L >= d - 1e-9)                                        # length >= distance
        out[name] = (r, d, L)
    r, d, L = out["driven"]
    h = hd / np.linalg.norm(hd)
    assert np.allclose(r @ h, h[2])                                          # a circle about h
    i = int(np.argmax(d))
    assert abs(ts[i] - th) < 0.01 and abs(d[i] - 0.983) < 1e-3 and abs(L[i] - 1.307) < 2e-3
    assert np.all(L[ts > 0.1] - d[ts > 0.1] > 1e-6)                         # strictly longer than the geodesic
    rr, dr, Lr = out["resonant"]
    k = ts <= np.pi / U
    assert np.allclose(Lr[k], dr[k], atol=1e-9) and abs(dr[k][-1] - np.pi / 2) < 0.01
    print("driven: Delta H = %.3f, farthest d = %.3f at t = %.3f (pi/|h| = %.3f), L there = %.3f, period %.3f"
          % (U / 2, d[i], ts[i], th, L[i], 2 * th))
    print("resonant: L = d up to t = pi/u = %.3f, where d = pi/2" % (np.pi / U))

    b = []
    # ---- (a) the Bloch sphere
    V = N.View(az=-125.0, el=17.0, R=2.15)
    b += V.sphere()
    b += V.great_circle([0, 0, 1])
    for v, lab, pos in [((0, 0, 1.24), "\\ket0", "above"), ((1.45, 0, 0), "x", "left"), ((0, -1.45, 0), "-y", "right")]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s, text=ndInk] {$%s$};" % (V.pt((0, 0, 1) if lab == "\\ket0" else (0, 0, 0)), V.pt(v), pos, lab))
    b.append("\\node[below, text=ndInk] at %s {$\\ket1$};" % V.pt((0, 0, -1.04)))
    hx = np.array([[0, -hd[2], hd[1]], [hd[2], 0, -hd[0]], [-hd[1], hd[0], 0]])      # r' = h x r
    full = np.array([expm(hx * t) @ [0, 0, 1.0] for t in np.linspace(0, 2 * th, 241)])
    assert np.allclose(full[:: 40], np.array([T.bloch_vector(np.outer(p, p.conj())) for p in path(E.OMEGA, U, np.linspace(0, 2 * th, 241)[:: 40])[0]]))
    b += V.curve(full, "line width=1.5pt, ndGreen", "line width=0.7pt, ndGreen, densely dotted")
    # the geodesic from |0> to the farthest point, and the pi-pulse meridian
    P = r[i]
    s = np.linspace(0, 1, 120)
    ang = np.arccos(P[2])
    geo = np.array([np.sin((1 - x) * ang) / np.sin(ang) * np.array([0, 0, 1.0]) + np.sin(x * ang) / np.sin(ang) * P for x in s])
    b += V.curve(geo, "line width=1.0pt, ndInk, dash pattern=on 3pt off 2pt", "line width=0.6pt, ndInk, densely dotted")
    b += V.curve(rr[k], "line width=1.5pt, ndBlue", "line width=0.7pt, ndBlue, densely dotted")
    b.append("\\draw[vec, ndGreen!70!ndInk, line width=1.0pt] %s -- %s node[anchor=north, font=\\small] {$\\vec h$};" % (V.pt((0, 0, 0)), V.pt(0.9 * h)))
    b.append("\\node[dot, fill=ndInk, minimum size=4.2pt] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[dot, fill=ndGreen] at %s {};" % V.pt(P))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % V.pt((0, 0, -1)))
    b.append("\\node[note, anchor=west, text=ndGreen!80!ndInk] (g) at (2.3,1.4) {$u=%.1f$, $\\omega=%d$:\\\\ a small circle};" % (U, E.OMEGA))
    b.append("\\draw[pointer] (g.west) to[bend right=15] %s;" % V.pt(full[30]))
    b.append("\\node[note, anchor=east] (q) at (-2.35,0.55) {the geodesic\\\\ to the farthest\\\\ point};")
    b.append("\\draw[pointer] (q.east) to[bend left=12] %s;" % V.pt(geo[60]))
    b.append("\\node[note, anchor=west, text=ndBlue] (p) at (2.35,-1.45) {$\\omega=0$: the\\\\ $\\pi$-pulse runs\\\\ along a meridian};")
    b.append("\\draw[pointer] (p.west) to[bend left=12] %s;" % V.pt(rr[k][int(0.72 * k.sum())]))
    b.append("\\node[font=\\small] at (0,-2.95) {(a) two paths from $\\ket0$};")

    # ---- (b) length and distance against time
    Pl = N.Plot((0, 3.5), (0, 2.75), 5.6, 4.1, origin=(5.9, -2.25))
    b += Pl.axes("$t$", "", xticks=[(1, "$1$"), (2, "$2$"), (3, "$3$")],
                 yticks=[(np.pi / 4, "$\\frac\\pi4$"), (np.pi / 2, "$\\frac\\pi2$")])
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(0, np.pi / 2), Pl.pt(3.5, np.pi / 2)))
    b.append(Pl.curve(ts, L, "line width=1.2pt, ndInk"))
    b.append(Pl.curve(ts, d, "line width=1.5pt, ndGreen"))
    b.append(Pl.curve(ts, dr, "line width=1.5pt, ndBlue"))
    b.append("\\draw[ndGreen, line width=0.5pt, densely dotted] %s -- %s;" % (Pl.pt(ts[i], d[i]), Pl.pt(ts[i], L[i])))
    b.append("\\node[dot, fill=ndGreen] at %s {};" % Pl.pt(ts[i], d[i]))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % Pl.pt(np.pi / U, np.pi / 2))
    b.append("\\node[font=\\small, anchor=south east, rotate=%.1f] at %s {$L(t)=\\int_0^t\\Delta H$};"
             % (np.degrees(np.arctan2(Pl.sy * U / 2, Pl.sx)), Pl.pt(3.4, U / 2 * 3.4 + 0.04)))
    b.append("\\node[note, anchor=north west, text=ndGreen!80!ndInk] at %s {distance $d(t)$\\\\ with drift};" % Pl.pt(0.98, 0.55))
    b.append("\\node[note, anchor=south west, text=ndBlue] (b) at %s {without drift,\\\\ $d=L$ until $\\ket1$};" % Pl.pt(0.15, 1.75))
    b.append("\\draw[pointer] (b.south) to[bend right=15] %s;" % Pl.pt(1.2, 0.75 * 1.2 + 0.06))
    b.append("\\node[note, anchor=west] (gap) at %s {$L>d$};" % Pl.pt(ts[i] + 0.12, 0.5 * (d[i] + L[i])))
    b.append("\\node[font=\\small] at %s {(b) length and distance (Fubini--Study)};" % N.pt(Pl.P(1.8, 0) + np.array([0, -0.7])))
    N.write("ch07_speed", b)
    print("written figures/ch07_speed.tex")


if __name__ == "__main__":
    main()
