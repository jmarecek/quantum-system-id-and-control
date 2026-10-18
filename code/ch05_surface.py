"""ch05_surface.py -- why poles are called poles.

The gain |G(s)| of the weakly driven qubit (ch05_response.py),
    G(s) = omega / ((s + Gamma)^2 + omega^2),
drawn as a surface over the left half of the complex s-plane.  It rises like
a tent on two poles at s = -Gamma +- i omega, where G is infinite (the spikes
are cut off), and its front edge, the section along the imaginary axis s = i nu,
is the frequency response |G(i nu)| of ch05_response.  Hidden parts are removed
by the painter's algorithm: the cells are filled from back to front.
Writes figures/ch05_surface.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import OMEGA, GAMMA
from ch05_response import G as G_qubit

SMIN, SMAX = -1.4, 0.0             # Re s on the surface
FLOOR = 0.6                        # the floor continues into the right half-plane
NMAX = 4.5                         # Im s in [-NMAX, NMAX]
CLIP = 2.6                         # spikes cut at this height
ZS = 1.25                          # page cm per unit of |G|


def G(s):
    return OMEGA / ((s + GAMMA) ** 2 + OMEGA ** 2)


def main():
    for s in (0.0, 1j * OMEGA, -0.3 + 2j, 1j * 1.7):
        assert np.isclose(G(s), G_qubit(s))
    sig = np.linspace(SMIN, SMAX, 29)
    nus = np.linspace(-NMAX, NMAX, 49)
    assert np.isclose(sig[np.argmin(abs(sig + GAMMA))], -GAMMA) and np.isclose(nus[np.argmin(abs(nus - OMEGA))], OMEGA)
    S, Nu = np.meshgrid(sig, nus, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        Hgt = np.abs(G(S + 1j * Nu))
    Hgt = np.where(np.isfinite(Hgt), np.minimum(Hgt, CLIP), CLIP)      # the grid passes through both poles
    assert (Hgt >= CLIP).sum() >= 2
    V = N.View(az=18.0, el=24.0, R=1.0, shift=(0.0, 0.0))
    # stretch Re s so that the decay direction is legible: page = View of (2 sigma, nu, ZS |G|)
    P3 = lambda s, n, h: V.P(np.array([2.2 * s, 0.9 * n, ZS * h]))
    pt = lambda s, n, h: N.pt(P3(s, n, h))
    b = []
    # ---- floor: the s-plane
    corners = [(SMIN, -NMAX), (FLOOR, -NMAX), (FLOOR, NMAX), (SMIN, NMAX)]
    b.append("\\fill[ndPaper] %s -- cycle;" % " -- ".join(pt(s, n, 0) for s, n in corners))
    b.append("\\fill[ndRose, fill opacity=0.5] %s -- cycle;" % " -- ".join(pt(s, n, 0) for s, n in [(0, -NMAX), (FLOOR, -NMAX), (FLOOR, NMAX), (0, NMAX)]))
    b.append("\\draw[faint] %s -- cycle;" % " -- ".join(pt(s, n, 0) for s, n in corners))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[below left, text=ndInk, font=\\small] {$\\mathrm{Re}\\,s$};" % (pt(SMIN, -NMAX - 0.3, 0), pt(FLOOR + 0.35, -NMAX - 0.3, 0)))
    b.append("\\node[note, anchor=north] at %s {right half-plane: no Lindbladian has poles here};" % N.pt(P3(FLOOR, 0.8, 0) + np.array([0.0, -0.05])))
    # ---- the surface, back to front
    cells = []
    for i in range(len(sig) - 1):
        for j in range(len(nus) - 1):
            q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            c3 = np.mean([[2.2 * S[a], 0.9 * Nu[a], ZS * Hgt[a]] for a in q], axis=0)
            cells.append((c3 @ V.d, q))
    cells.sort(key=lambda c: c[0])
    for _, q in cells:
        top = max(Hgt[a] for a in q) >= CLIP - 1e-9
        fill = "ndRose!70" if top else "ndSky!55"
        b.append("\\filldraw[fill=%s, draw=ndInk!45, line width=0.25pt] %s -- cycle;"
                 % (fill, " -- ".join(pt(S[a], Nu[a], Hgt[a]) for a in q)))
    # ---- the section along the imaginary axis: a curtain and the red curve
    nn = np.linspace(-NMAX, NMAX, 361)
    gain = np.abs(G(1j * nn))
    assert np.allclose(gain, [abs(G_qubit(1j * v)) for v in nn])
    assert np.allclose(gain[::45], Hgt[-1, ::6])                    # the front edge of the surface
    curt = [P3(0, n, h) for n, h in zip(nn, gain)] + [P3(0, NMAX, 0), P3(0, -NMAX, 0)]
    b.append("\\fill[ndRose, fill opacity=0.35] %s -- cycle;" % " -- ".join(N.pt(p) for p in curt))
    b.append(N.polyline([P3(0, n, h) for n, h in zip(nn, gain)], "line width=1.6pt, ndRed"))
    b.append("\\draw[line width=0.9pt, ndRed] %s -- %s;" % (pt(0, -NMAX, 0), pt(0, NMAX, 0)))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk, font=\\small] {$\\mathrm{Im}\\,s$};" % (pt(0, NMAX, 0), pt(0, NMAX + 0.7, 0)))
    for v, lab in [(-OMEGA, "$-\\omega$"), (0, "$0$"), (OMEGA, "$\\omega$")]:
        b.append("\\draw[ink] %s -- ++(0,-0.08) node[below, font=\\scriptsize] {%s};" % (pt(0, v, 0), lab))
    # ---- notes
    b.append("\\node[note, anchor=south] (p) at %s {poles at $-\\Gamma\\pm\\imath\\omega$: $\\TF$ is infinite\\\\ there (the peaks are truncated)};" % N.pt(P3(-GAMMA, 0.0, CLIP) + np.array([0.0, 0.55])))
    for sgn in (-1, 1):
        b.append("\\draw[pointer] (p.%s) to[bend %s=15] %s;" % ("south west" if sgn < 0 else "south east", "right" if sgn < 0 else "left",
                                                              pt(-GAMMA, sgn * OMEGA, CLIP + 0.05)))
    b.append("\\node[note, anchor=north west, text=ndRed] (c) at %s {the section along $s=\\imath\\nu$:\\\\ the frequency response $\\abs{\\TF(\\imath\\nu)}$};" % N.pt(P3(0, NMAX, 0) + np.array([0.5, 1.35])))
    b.append("\\draw[pointer, ndRed!70] (c.west) to[bend right=20] %s;" % pt(0, 3.6, abs(G(3.6j)) + 0.02))
    b.append("\\node[note, anchor=north east] at %s {height: $\\abs{\\TF(s)}$ over\\\\ the left half-plane};" % N.pt(P3(SMIN, -NMAX, 0) + np.array([-0.15, 1.9])))
    N.write("ch05_surface", b)
    print("written figures/ch05_surface.tex; %d cells, peak of the cut %.3f" % (len(cells), gain.max()))


if __name__ == "__main__":
    main()
