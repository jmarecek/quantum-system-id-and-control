"""ch02_bloch.py -- Bloch-sphere trajectories of a driven qubit.

Resonant drive (Delta = 0) versus detuned drive (Delta = 1) with the same
Rabi amplitude u = 1, both started in |0>.  The Bloch vector precesses about
the field vector h = (u, 0, Delta).  Writes figures/ch02_bloch.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N

I, X, Y, Z = T.pauli()
V = N.View(az=-60.0, el=18.0, R=2.6)


def trajectory(Delta, u, T_final=np.pi, steps=200):
    """Bloch vectors of the solution of  i d/dt psi = (Delta/2 Z + u/2 X) psi."""
    H = 0.5 * Delta * Z + 0.5 * u * X
    psi = np.array([1.0, 0.0], dtype=complex)          # |0>
    dt = T_final / steps
    U_dt = expm(-1j * H * dt)                           # propagator of one step
    out = []
    for _ in range(steps + 1):
        rho = np.outer(psi, psi.conj())
        out.append(T.bloch_vector(rho))
        psi = U_dt @ psi
    return np.array(out)


def figure(resonant, detuned):
    b = V.sphere()
    b += V.great_circle([0, 0, 1])                                    # equator
    for v, lab, pos in [((0, 0, 1.3), "z", "above"), ((1.4, 0, 0), "x", "above"),
                        ((0, 1.4, 0), "y", "above right")]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s] {$%s$};" % (V.pt((0, 0, 0)), V.pt(v), pos, lab))
    # the two field vectors (rotation axes), drawn to the sphere
    hb = np.array([1.0, 0.0, 0.0])
    hr = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    b.append("\\draw[vec, ndBlue!70] %s -- %s node[below, yshift=-2pt, text=ndBlue] {$\\vec h=(1,0,0)$};"
             % (V.pt((0, 0, 0)), V.pt(hb)))
    b.append("\\draw[vec, ndGold] %s -- %s node[above right, text=ndGold!80!black] {$\\vec h=(1,0,1)$};"
             % (V.pt((0, 0, 0)), V.pt(hr)))
    # trajectories: solid in front, dotted behind
    b += V.curve(resonant, "line width=1.6pt, ndBlue", "line width=0.8pt, ndBlue, densely dotted")
    b += V.curve(detuned, "line width=1.6pt, ndGold", "line width=0.8pt, ndGold, densely dotted")
    k = len(resonant) // 3                                           # direction arrow on the resonant path
    b.append("\\draw[->, >=stealth, line width=1.6pt, ndBlue] %s -- %s;" % (V.pt(resonant[k]), V.pt(resonant[k + 3])))
    b.append("\\node[dot, label={[font=\\footnotesize]above left:$\\ket0$}] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[dot, fill=ndBlue, label={[font=\\footnotesize]below:$\\ket1$}] at %s {};" % V.pt((0, 0, -1)))
    lowest = detuned[np.argmin(detuned[:, 2])]
    b.append("\\node[dot, fill=ndGold] at %s {};" % V.pt(lowest))
    # annotations
    x0 = V.R + 0.7
    b.append("\\node[note, anchor=east, text=ndBlue] (a) at (%.2f,1.4) {resonant, $\\Delta=0$:\\\\"
             "a great circle about $x$;\\\\ at $t=\\pi$ it reaches $\\ket1$:\\\\ a $\\pi$-pulse};" % (-V.R - 0.5))
    b.append("\\draw[pointer] (a.east) to[bend left=15] %s;" % V.pt(resonant[int(0.3 * len(resonant))]))
    b.append("\\node[note, anchor=west, text=ndGold!80!black] (c) at (%.2f,-1.6) {detuned, $\\Delta=u$:\\\\"
             "the axis tilts, the circle\\\\ shrinks and misses $\\ket1$};" % x0)
    b.append("\\draw[pointer] (c.west) to[bend left=10] %s;" % V.pt(lowest))
    b.append("\\node[note, anchor=east] at (%.2f,-2.4) {$\\dot{\\mathbf r}=\\vec h\\times\\mathbf r$:\\\\"
             "precession about $\\vec h$};" % (-0.9,))
    return b


if __name__ == "__main__":
    resonant = trajectory(Delta=0.0, u=1.0)            # a pi-pulse: |0> -> |1>
    detuned = trajectory(Delta=1.0, u=1.0, T_final=2 * np.pi / np.sqrt(2))
    assert np.allclose(resonant[-1], [0, 0, -1], atol=1e-8)
    N.write("ch02_bloch", figure(resonant, detuned))
    print("written figures/ch02_bloch.tex; detuned min z = %.3f" % detuned[:, 2].min())
