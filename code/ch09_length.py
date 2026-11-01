"""ch09_length.py -- length and distance along the time-optimal transfer.

The driven qubit of Chapter 4, H = (1/2)(Delta sigma_z + u sigma_x) with
Delta = -omega = -1 and |u| <= u0 = 1.5, and two controls from |0>:
  (i)  the time-optimal two-bang control of ch09_timeoptimal.py, u = -u0 for
       t1 = 1.127 and u = +u0 for t2 = 2.359, which reaches |1> at t* = 3.485;
  (ii) the constant pulse u = -u0, which keeps the state on the circle of the
       first bang and never reaches |1>.
For each, the Fubini--Study distance d(t) = arccos |<0|psi(t)>| from |0> and
the length L(t) = int_0^t Delta H travelled.  On a bang the Bloch vector turns
about h = (u, 0, Delta), h.r is constant and Delta H = (1/2) sqrt(Omega^2 - (h.r)^2).
Here h.r = Delta = -1 on the first bang (it starts at |0>) and h.r = +1 on the
second (it ends at |1>), so that Delta H = u0/2 = 0.75 on both controls
throughout, and L(t) = 0.75 t is the same line for both.  The speed limit of
Chapter 7 would need Delta H = Omega/2 = 0.901, which requires h.r = 0; along
the meridian (the Fubini--Study geodesic) the state would reach |1> at
pi/Omega = 1.743.  The optimal control reaches d = pi/2 at t* = 3.485 after the
length L(t*) = 0.75 t* = 2.614, an excess of 1.043 over the distance pi/2.
After the switch on the equator (d = pi/4) the second bang first carries the
state back towards |0>, to d = 0.588 at t = 1.743, before descending to |1>.
The constant pulse gets no farther than d = 0.983, at t = 1.743, and is back at
|0> at t* = 2 pi/Omega.
Writes figures/ch09_length.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N

I2, SX, SY, SZ = T.pauli()
DELTA, U0 = -1.0, 1.5
W = np.hypot(U0, DELTA)
T1 = (np.pi - np.arccos((DELTA / U0) ** 2)) / W
T2 = (np.pi + np.arccos((DELTA / U0) ** 2)) / W
TSTAR = T1 + T2


def Hm(u):
    return 0.5 * (DELTA * SZ + u * SX)


def run(control, ts):
    """States, distances and speeds along a piecewise-constant control given as a function of t."""
    psi0 = np.array([1, 0], dtype=complex)
    psis, dH = [], []
    psi, t_prev = psi0.copy(), 0.0
    for t in ts:
        # integrate exactly over [t_prev, t], splitting at the switch t1
        cuts = [t_prev] + [c for c in (T1,) if t_prev < c < t] + [t]
        for a, b in zip(cuts[:-1], cuts[1:]):
            psi = expm(-1j * Hm(control(0.5 * (a + b))) * (b - a)) @ psi
        H = Hm(control(t))
        psis.append(psi.copy())
        dH.append(np.sqrt(max(0.0, np.real(psi.conj() @ H @ H @ psi - (psi.conj() @ H @ psi) ** 2))))
        t_prev = t
    psis = np.array(psis)
    d = np.arccos(np.clip(np.abs(psis[:, 0]), 0, 1))
    dH = np.array(dH)
    L = np.concatenate([[0], np.cumsum(0.5 * (dH[1:] + dH[:-1]) * np.diff(ts))])
    return d, L, dH


def checks():
    ts = np.linspace(0, TSTAR, 2001)
    d, L, dH = run(lambda t: -U0 if t < T1 else U0, ts)
    assert np.allclose(dH, U0 / 2, atol=1e-9)                        # speed u0/2 on both bangs
    assert abs(d[-1] - np.pi / 2) < 1e-6 and np.all(L >= d - 1e-9)
    assert abs(L[-1] - U0 / 2 * TSTAR) < 1e-9 and abs(L[-1] - 2.614) < 1e-3
    dc, Lc, dHc = run(lambda t: -U0, ts)
    assert np.allclose(dHc, U0 / 2, atol=1e-9) and np.all(Lc >= dc - 1e-9)
    i = np.argmax(dc)
    assert abs(dc[i] - 0.983) < 1e-3 and abs(ts[i] - np.pi / W) < 2e-3
    assert np.all(dH <= W / 2 + 1e-12)                                # below the speed limit
    k = ts > T1
    j = np.argmin(d[k])                                              # the climb back after the switch
    assert abs(d[np.searchsorted(ts, T1)] - np.pi / 4) < 1e-3 and abs(d[k][j] - 0.588) < 1e-3
    assert abs(dc[-1]) < 1e-6                                        # the constant pulse is back at |0>
    return ts, d, L, dc, Lc


def main():
    ts, d, L, dc, Lc = checks()
    b = []
    pl = N.Plot((0, 3.75), (0, 2.85), 7.4, 4.6, origin=(0.0, 0.0))
    b += pl.axes(xlabel="{\\scriptsize $t$}", xticks=[(np.pi / W, "$\\frac{\\pi}{\\Omega}$"), (T1, "$t_1$"), (TSTAR, "$t^\\star$")],
                 yticks=[(np.pi / 4, "$\\frac\\pi4$"), (np.pi / 2, "$\\frac\\pi2$")])
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (pl.pt(0, np.pi / 2), pl.pt(3.75, np.pi / 2)))
    # the speed limit: the geodesic at the largest speed Omega/2
    tg = np.linspace(0, np.pi / W, 50)
    b.append(pl.curve([0, 3.12], [0, W / 2 * 3.12], "line width=0.7pt, ndGold!80, dash pattern=on 1pt off 1.5pt"))
    b.append(pl.curve(tg, W / 2 * tg, "line width=1.2pt, ndGold, dash pattern=on 3pt off 2pt"))
    b.append("\\node[dot, fill=ndGold] at %s {};" % pl.pt(np.pi / W, np.pi / 2))
    # the common length line, and the two distances
    b.append(pl.curve(ts, L, "line width=1.2pt, ndInk"))
    b.append(pl.curve(ts, dc, "line width=1.2pt, ndGreen"))
    b.append(pl.curve(ts, d, "line width=1.6pt, ndBlue"))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % pl.pt(TSTAR, np.pi / 2))
    b.append("\\draw[ndBlue, line width=0.6pt, <->, >=stealth] %s -- %s;" % (pl.pt(TSTAR + 0.1, np.pi / 2 + 0.02), pl.pt(TSTAR + 0.1, L[-1] - 0.02)))
    b.append("\\node[note, anchor=west, text=ndBlue] at %s {excess\\\\ $%.2f$};" % (pl.pt(TSTAR + 0.13, 0.5 * (np.pi / 2 + L[-1])), L[-1] - np.pi / 2))
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (pl.pt(T1, 0), pl.pt(T1, d[np.searchsorted(ts, T1)])))
    ang = lambda s: np.degrees(np.arctan2(pl.sy * s, pl.sx))
    b.append("\\node[font=\\scriptsize, anchor=north, rotate=%.1f] at %s {length $\\int_0^t\\Delta H=\\frac{u_0}{2}t$, both controls};" % (ang(U0 / 2), pl.pt(2.85, U0 / 2 * 2.85 - 0.04)))
    b.append("\\node[note, anchor=south east, text=ndGold!55!ndInk] (sl) at %s {the speed limit $\\frac\\Omega2t$:\\\\ the geodesic would\\\\ reach $\\ket1$ at $\\frac\\pi\\Omega$};" % pl.pt(1.25, 1.95))
    b.append("\\draw[pointer] (sl.south east) to[bend right=10] %s;" % pl.pt(1.35, W / 2 * 1.35 + 0.03))
    b.append("\\node[note, anchor=north west, text=ndBlue] (op) at %s {distance, optimal\\\\ control: $\\frac\\pi2$ at $t^\\star$};" % pl.pt(3.05, 1.2))
    b.append("\\draw[pointer] (op.north west) to[bend left=15] %s;" % pl.pt(2.95, d[np.searchsorted(ts, 2.95)] - 0.03))
    b.append("\\node[note, anchor=north, text=ndBlue] (dp) at %s {after the switch,\\\\ back towards $\\ket0$};" % pl.pt(2.0, 0.47))
    b.append("\\draw[pointer] (dp.north) to[bend right=20] %s;" % pl.pt(1.76, 0.565))
    b.append("\\node[note, anchor=west, text=ndGreen!80!ndInk] (cp) at %s {distance, constant\\\\ pulse: at most $%.2f$,\\\\ back at $\\ket0$ at $t^\\star$};" % (pl.pt(3.6, 0.42), dc.max()))
    b.append("\\draw[pointer] (cp.west) to[bend right=15] %s;" % pl.pt(3.2, dc[np.searchsorted(ts, 3.2)] + 0.03))
    N.write("ch09_length", b)
    return L[-1], dc.max()


if __name__ == "__main__":
    Lt, dmax = main()
    print("L(t*) = %.4f, excess over pi/2 = %.4f; constant pulse: max distance %.4f" % (Lt, Lt - np.pi / 2, dmax))
    print("written figures/ch09_length.tex")
