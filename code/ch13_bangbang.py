"""ch13_bangbang.py -- the two time-optimal transfers of a qubit, on the Bloch sphere.

Example B revisited.  The qubit is moved from |up> (north pole) to |down> (south
pole) in the least time.
  * Two controls and no drift, ||(u_x, u_z)|| <= u_0: the pi-pulse u_x = u_0,
    a rotation about x at the full speed u_0 along the great half-circle in the
    y-z plane, T = pi/u_0.  This half-circle is the geodesic of the
    Mandelstam--Tamm speed limit.
  * One control and a drift, H = (Delta/2) sigma_z + (u/2) sigma_x, |u| <= u_0:
    the Bloch vector turns at the rate Omega = sqrt(u_0^2 + Delta^2) about
    (+-u_0, 0, Delta)/Omega.  For Delta < u_0 the optimum has two bangs,
        t_1 = (pi - arccos(Delta^2/u_0^2))/Omega,
        t_2 = (pi + arccos(Delta^2/u_0^2))/Omega,
    with the switch on the equator and T* = t_1 + t_2 = 2 pi/Omega, twice the
    speed-limit bound pi/Omega: the geodesic is not a trajectory of the system.
The parameters Delta = 1/4, u_0 = 1/2 are those of the BEC experiment of the
next frame (lattice depth s = 0.5, Delta = s/2); there t_2/t_1 = 2.45, as for
the measured 156 us / 64 us.  The script checks that the two bangs end at the
south pole, that the switch lies on the equator, and that no sequence of up to
five bangs reaches the pole sooner.
Writes figures/ch13_bangbang.tex.
"""
import os
import sys

import warnings

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

DELTA, U0 = 0.25, 0.5
OMEGA = np.hypot(DELTA, U0)


def rotation(axis, angle):
    """Rotation of the Bloch ball by `angle` about `axis` (the action of exp(-i H t))."""
    n = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def bangs(durations, sign=+1, r0=(0.0, 0.0, 1.0), m=160):
    """Bloch trajectory under alternating bangs u = sign*u_0, -sign*u_0, ... of the given durations."""
    r, arcs = np.asarray(r0, dtype=float), []
    for k, d in enumerate(durations):
        axis = [sign * U0 * (-1) ** k, 0.0, DELTA]
        arcs.append(np.array([rotation(axis, OMEGA * s) @ r for s in np.linspace(0, d, m)]))
        r = arcs[-1][-1]
    return arcs


def least_time(k, starts=60, seed=3):
    """Least total duration of k alternating bangs that ends at the south pole (local searches)."""
    rng, best = np.random.default_rng(seed), np.inf
    end = lambda d: bangs(d, m=2)[-1][-1]
    for _ in range(starts):
        res = minimize(lambda d: d.sum(), rng.uniform(0, 2 * np.pi / OMEGA, k), method="SLSQP",
                       bounds=[(0, 4 * np.pi / OMEGA)] * k,
                       constraints=[{"type": "eq", "fun": lambda d: end(d)[:2]}], options=dict(maxiter=400, ftol=1e-14))
        r = end(res.x)
        if res.success and r[2] < 0 and np.hypot(r[0], r[1]) < 1e-9:
            best = min(best, res.x.sum())
    return best


def main():
    warnings.filterwarnings("ignore", message="Values in x were outside bounds")
    a = np.arccos(DELTA ** 2 / U0 ** 2)
    t1, t2 = (np.pi - a) / OMEGA, (np.pi + a) / OMEGA
    arcs = bangs([t1, t2])
    switch = arcs[0][-1]
    assert np.allclose(arcs[1][-1], [0, 0, -1], atol=1e-9)                # the two bangs reach |down>
    assert abs(switch[2]) < 1e-9                                           # the switch lies on the equator
    assert np.isclose(t1 + t2, 2 * np.pi / OMEGA)
    for k in (2, 3, 4, 5):                                                 # no sequence of bangs is faster
        assert least_time(k) > 2 * np.pi / OMEGA - 1e-9
    th = np.linspace(0, np.pi, 160)
    geodesic = np.stack([np.zeros_like(th), np.sin(th), np.cos(th)], 1)   # the pi-pulse u_x = -u_0, about -x
    print("t1 = %.4f  t2 = %.4f  T* = 2pi/Omega = %.4f  speed limit pi/Omega = %.4f  pi-pulse pi/u0 = %.4f  t2/t1 = %.3f"
          % (t1, t2, t1 + t2, np.pi / OMEGA, np.pi / U0, t2 / t1))

    V = N.View(az=20.0, el=18.0, R=2.0)
    b = V.sphere()
    b += V.great_circle([0, 0, 1])
    for v, lab, pos in [((0, 0, 1.3), "z", "above"), ((1.6, 0, 0), "x", "below left"), ((0, 1.3, 0), "y", "right")]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s] {$%s$};" % (V.pt((0, 0, 0)), V.pt(v), pos, lab))
    for s, lab, pos in [(+1, "+u_0", "below left"), (-1, "-u_0", "left")]:    # the two axes of rotation
        n = np.array([s * U0, 0.0, DELTA]) / OMEGA
        b.append("\\draw[->, >=stealth, line width=0.8pt, ndInk!70] %s -- %s node[%s, font=\\scriptsize, inner sep=1pt] {$%s$};"
                 % (V.pt((0, 0, 0)), V.pt(n), pos, lab))
    b += V.curve(geodesic, "line width=1.5pt, ndGreen", "line width=0.8pt, ndGreen, densely dotted")
    b += V.curve(arcs[0], "line width=1.5pt, ndBlue", "line width=0.8pt, ndBlue, densely dotted")
    b += V.curve(arcs[1], "line width=1.5pt, ndBlue", "line width=0.8pt, ndBlue, densely dotted")
    for path, col, f in [(geodesic, "ndGreen", 0.3), (arcs[0], "ndBlue", 0.5), (arcs[1], "ndBlue", 0.2), (arcs[1], "ndBlue", 0.9)]:
        k = int(f * len(path))
        if V.visible(path[k]) and V.visible(path[k + 4]):
            b.append("\\draw[->, >=stealth, line width=1.5pt, %s] %s -- %s;" % (col, V.pt(path[k]), V.pt(path[k + 4])))
    b.append("\\node[dot, label={[font=\\footnotesize]above left:$\\ket\\uparrow$}] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[dot, label={[font=\\footnotesize]below:$\\ket\\downarrow$}] at %s {};" % V.pt((0, 0, -1)))
    b.append("\\node[dot, fill=ndRed, minimum size=4.5pt] at %s {};" % V.pt(switch))
    # notes: the pi-pulse on the right, the two bangs on the left
    b.append("\\node[note, anchor=west, text=ndGreen!80!ndInk] (g) at (%.2f,1.15) {$\\pi$-pulse, no drift:\\\\ the geodesic, at full\\\\ speed, $T=\\pi/u_0=%.2f$};"
             % (V.R + 0.3, np.pi / U0))
    b.append("\\draw[pointer] (g.west) to[bend right=12] %s;" % V.pt(geodesic[38]))
    b.append("\\node[note, anchor=west] (sl) at (%.2f,-1.25) {with the drift, the\\\\ speed limit would allow\\\\ $\\pi/\\Omega=%.2f$ along it:\\\\ not a trajectory};"
             % (V.R + 0.3, np.pi / OMEGA))
    b.append("\\draw[pointer] (sl.west) to[bend left=12] %s;" % V.pt(geodesic[110]))
    b.append("\\node[note, anchor=east, text=ndRed] (s) at (%.2f,0.55) {the switch,\\\\ on the equator};" % (-V.R - 0.3))
    b.append("\\draw[pointer] (s.east) to[bend left=12] %s;" % V.pt(switch))
    kb = int(0.93 * len(arcs[1]))
    b.append("\\node[note, anchor=east, text=ndBlue] (bb) at (%.2f,-1.2) {two bangs with drift:\\\\ $t_1=%.2f$, $t_2=%.2f$,\\\\ $T^\\star=2\\pi/\\Omega=%.2f$};"
             % (-V.R - 0.3, t1, t2, t1 + t2))
    b.append("\\draw[pointer] (bb.east) to[bend right=12] %s;" % V.pt(arcs[1][kb]))
    N.write("ch13_bangbang", b)
    print("written figures/ch13_bangbang.tex")


if __name__ == "__main__":
    main()
