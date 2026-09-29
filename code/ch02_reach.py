"""ch02_reach.py -- reachable sets grow with time; the minimum time is first contact.

A resonant qubit (no drift) with a bounded drive in the equatorial plane,
H = (u_x sx + u_y sy)/2 with u_x^2 + u_y^2 <= M^2.  The Bloch vector moves
with speed |h x r| <= M, so the states reachable from |0> at time T form the
spherical cap of angular radius min(M T, pi) about the north pole (any point of
the cap is reached by a rotation about a horizontal axis, slowed down if need
be).  A target at polar angle alpha is first reached at T_min = alpha / M,
along a meridian.  As a sanity check, random bounded piecewise-constant controls
never leave the cap.
Writes figures/ch02_reach.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N

M = 1.0
ALPHA = np.radians(125.0)                  # polar angle of the target
AZT = np.radians(20.0)                     # azimuth of the target (faces the viewer)
TIMES = [0.35 * ALPHA, 0.65 * ALPHA, ALPHA]
V = N.View(az=25.0, el=18.0, R=2.5)
I, X, Y, Z = T.pauli()


def cap_points(polar, n=361):
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([np.sin(polar) * np.cos(t), np.sin(polar) * np.sin(t),
                     np.cos(polar) * np.ones_like(t)], axis=1)


def visible_cap_polygon(polar, n=721):
    """Page polygon of the visible part of the cap {angle to north pole <= polar}."""
    c = cap_points(polar, n)[:-1]
    vis = c @ V.d >= 0
    if vis.all():
        return [V.P(p) for p in c]
    # rotate so the visible run is contiguous, then take it
    k = np.argmin(vis)                      # an invisible index
    c, vis = np.roll(c, -k, axis=0), np.roll(vis, -k)
    arc = c[vis]
    # silhouette points (v.d = 0) lying inside the cap, from the end of arc back to its start
    phi = np.linspace(0, 2 * np.pi, n)[:-1]
    sil = np.outer(np.cos(phi), V.e1) + np.outer(np.sin(phi), V.e2)
    inside = sil[:, 2] >= np.cos(polar)
    k = np.argmin(inside)
    sil, inside = np.roll(sil, -k, axis=0), np.roll(inside, -k)
    rim = sil[inside]
    if np.linalg.norm(rim[0] - arc[-1]) > np.linalg.norm(rim[-1] - arc[-1]):
        rim = rim[::-1]
    return [V.P(p) for p in np.vstack([arc, rim])]


def sampled_max_polar(Tf, samples=400, steps=12, rng=np.random.default_rng(1)):
    """Largest polar angle reached from |0> by random piecewise-constant bounded controls."""
    best = 0.0
    for _ in range(samples):
        psi = np.array([1, 0], dtype=complex)
        for _ in range(steps):
            a, r = rng.uniform(0, 2 * np.pi), M * np.sqrt(rng.uniform())
            H = 0.5 * r * (np.cos(a) * X + np.sin(a) * Y)
            psi = expm(-1j * H * Tf / steps) @ psi
        z = T.bloch_vector(np.outer(psi, psi.conj()))[2]
        best = max(best, np.arccos(np.clip(z, -1, 1)))
    return best


def main():
    for Tf in TIMES:                        # sampled controls stay inside the cap
        assert sampled_max_polar(Tf) <= M * Tf + 1e-9
    b = V.sphere(shade=False)
    b.insert(0, "\\fill[ndPaper] %s circle (%.3f);" % (N.pt(V.shift), V.R))
    b += V.great_circle([0, 0, 1], "faint", "faint, densely dotted")
    washes = ["ndSky!60", "ndSky!85", "ndBlue!30"]
    for Tf, w in sorted(zip(TIMES, washes), key=lambda z: -z[0]):
        poly = visible_cap_polygon(M * Tf)
        b.append("\\fill[%s, fill opacity=0.7] %s -- cycle;" % (w, " -- ".join(N.pt(p) for p in poly)))
    for k, Tf in enumerate(TIMES):
        col = "ndBlue" if k < len(TIMES) - 1 else "ndRed"
        b += V.small_circle([0, 0, 1], M * Tf, "line width=0.9pt, %s" % col, "line width=0.5pt, %s, densely dotted" % col)
        lab = V.P([np.sin(M * Tf) * np.cos(-0.35), np.sin(M * Tf) * np.sin(-0.35), np.cos(M * Tf)])
        name = "\\Reach_{T_%d}" % (k + 1) if k < len(TIMES) - 1 else "\\Reach_{T_{\\min}}"
        b.append("\\node[font=\\footnotesize, text=%s, fill=white, fill opacity=0.7, text opacity=1, inner sep=1pt] at %s {$%s$};"
                 % (col, N.pt(lab), name))
    # a time-optimal path: the meridian from |0> to the target
    t = np.linspace(0, ALPHA, 100)
    mer = np.stack([np.sin(t) * np.cos(AZT), np.sin(t) * np.sin(AZT), np.cos(t)], axis=1)
    b += V.curve(mer, "->, >=stealth, line width=1.3pt, ndRed", "line width=0.6pt, ndRed, densely dotted")
    target = mer[-1]
    b.append("\\node[dot, label={[font=\\footnotesize]above:$\\ket0$}] at %s {};" % V.pt([0, 0, 1]))
    b.append("\\node[dot, fill=ndRed, minimum size=4.4pt, label={[font=\\footnotesize, text=ndRed]below right:target}] at %s {};"
             % V.pt(target))
    # annotations
    x0 = V.R + 0.6
    b.append("\\node[note, anchor=west] (a) at (%.2f,1.9) {bounded drive, $|u|\\le M$:\\\\"
             "at time $T$ the reachable states\\\\ form a cap of radius $MT$};" % x0)
    b.append("\\draw[pointer] (a.west) to[bend right=15] %s;"
             % V.pt([np.sin(M * TIMES[0]) * np.cos(0.5), np.sin(M * TIMES[0]) * np.sin(0.5), np.cos(M * TIMES[0])]))
    b.append("\\node[note, anchor=west] (b) at (%.2f,-0.3) {the caps grow with $T$:\\\\"
             "$\\Reach_{T_1}\\subset\\Reach_{T_2}\\subset\\Reach_{T_{\\min}}$};" % x0)
    b.append("\\node[note, anchor=west] (c) at (%.2f,-2.1) {\\emph{first contact} with the\\\\"
             "target: minimum time\\\\ $T_{\\min}=\\alpha/M$, along a meridian};" % x0)
    b.append("\\draw[pointer] (c.west) to[bend left=15] %s;" % V.pt(target))
    N.write("ch02_reach", b)
    print("written figures/ch02_reach.tex; T_min = %.3f" % (ALPHA / M))


if __name__ == "__main__":
    main()
