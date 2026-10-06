"""ch03_bitflip.py -- the bit-flip channel from a random Hamiltonian.

With probability p the noise Hamiltonian H^n = (pi/2) sigma_x acts for one unit of
time (a rotation by pi about the x-axis), with probability 1-p nothing happens.
Each sample path is a rotation; the average state has Bloch vector
(x, (1-2p) y, (1-2p) z): the ball is squeezed into a cigar along the x-axis.
Writes figures/ch03_bitflip.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N

I, X, Y, Z = T.pauli()
P = 0.3
V = N.View(az=30.0, el=20.0, R=2.5)


def main():
    r0 = np.array([0.35, 0.62, 0.70]); r0 /= np.linalg.norm(r0)
    rho0 = 0.5 * (I + r0[0] * X + r0[1] * Y + r0[2] * Z)
    s = np.linspace(0, 1, 80)
    path = [T.bloch_vector(expm(-0.5j * np.pi * X * a) @ rho0 @ expm(0.5j * np.pi * X * a)) for a in s]
    r1 = path[-1]
    avg = (1 - P) * r0 + P * r1
    kraus = (1 - P) * rho0 + P * X @ rho0 @ X
    assert np.allclose(T.bloch_vector(kraus), avg)
    assert np.allclose(avg, [r0[0], (1 - 2 * P) * r0[1], (1 - 2 * P) * r0[2]])
    b = V.sphere()
    b += V.great_circle([0, 0, 1])
    for v, lab, pos in [((1.45, 0, 0), "x", "below left"), ((0, 1.3, 0), "y", "right"), ((0, 0, 1.25), "z", "above")]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s, text=ndInk] {$%s$};" % (V.pt((0, 0, 0)), V.pt(v), pos, lab))
    out = N.ellipsoid_outline(V, (0, 0, 0), (1.0, 1 - 2 * P, 1 - 2 * P))
    b.append("\\filldraw[fill=ndGreen!20, draw=ndGreen, line width=1.0pt, fill opacity=0.8] %s -- cycle;" % " -- ".join(N.pt(p) for p in out))
    b += V.curve(np.array(path), "->, >=stealth, line width=1.4pt, ndRed", "line width=0.7pt, ndRed, densely dotted")
    b.append("\\draw[dashedink] %s -- %s;" % (V.pt(r0), V.pt(r1)))
    b.append("\\node[dot, label={[font=\\footnotesize]above right:$\\Bloch$}] at %s {};" % V.pt(r0))
    b.append("\\node[dot, fill=ndRed, label={[font=\\footnotesize, text=ndRed]below:flipped}] at %s {};" % V.pt(r1))
    b.append("\\node[dot, fill=ndGreen, minimum size=4.4pt] at %s {};" % V.pt(avg))
    x0 = V.R + 0.6
    b.append("\\node[note, anchor=west] (n) at (%.2f,2.0) {probability $1-p$:\\\\ nothing happens};" % x0)
    b.append("\\node[note, anchor=west, text=ndRed] (f) at (%.2f,0.6) {probability $p$: $H^{\\rm n}=\\tfrac\\pi2\\sigx$,\\\\ a rotation by $\\pi$ about $x$};" % x0)
    b.append("\\draw[pointer] (f.west) to[bend right=15] %s;" % V.pt(path[30]))
    b.append("\\node[note, anchor=west, text=ndGreen!60!black] (a) at (%.2f,-0.9) {the average: $(x,(1-2p)y,(1-2p)z)$,\\\\ on the chord, at weight $p$};" % x0)
    b.append("\\draw[pointer] (a.west) to[bend left=15] %s;" % V.pt(avg))
    b.append("\\node[note, anchor=west, text=ndGreen!60!black] at (%.2f,-2.3) {all states: the ball becomes\\\\ a cigar along $x$ ($p=%.1f$)};" % (x0, P))
    N.write("ch03_bitflip", b)
    print("written figures/ch03_bitflip.tex")


if __name__ == "__main__":
    main()
