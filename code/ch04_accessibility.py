"""ch04_accessibility.py -- accessibility is not controllability.

A resonant qubit with bounded controls, H = (u_x sigma_x + u_y sigma_y)/2,
u_x^2 + u_y^2 <= M^2, piecewise constant on K intervals, starts from the mixed
state r0 = (0.6, 0, 0) and is subject to
  (a) dephasing at the rate gamma_phi (unital), or
  (b) amplitude damping towards |0> at the rate gamma_1 (not unital).
The Bloch vectors reached at time T by random controls fill a region with
non-empty interior in both cases (accessibility), but never the whole ball
(no controllability): under (a) |r| never exceeds |r0|; under (b) the state
can become purer, but only by moving towards the ground state.
Writes figures/ch04_accessibility.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N

M_AMP, K, T_END = 2.0, 10, 3.0
GAMMA = 0.5
R0 = np.array([0.6, 0.0, 0.0])
N_RUNS = 900


def generator(ux, uy, kind):
    """Affine generator of r' = h x r + dissipation, as a 4x4 matrix acting on (r, 1)."""
    h = np.array([ux, uy, 0.0])
    G = np.zeros((4, 4))
    G[:3, :3] = np.array([[0, -h[2], h[1]], [h[2], 0, -h[0]], [-h[1], h[0], 0]])    # r' = h x r
    if kind == "dephasing":
        G[0, 0] -= GAMMA; G[1, 1] -= GAMMA
    else:                                                                              # amplitude damping to z = +1
        G[0, 0] -= GAMMA / 2; G[1, 1] -= GAMMA / 2; G[2, 2] -= GAMMA; G[2, 3] += GAMMA
    return G


def reachable(kind, rng):
    pts = []
    dt = T_END / K
    for _ in range(N_RUNS):
        v = np.append(R0, 1.0)
        for _ in range(K):
            a, rad = rng.uniform(0, 2 * np.pi), M_AMP * np.sqrt(rng.uniform())
            v = expm(generator(rad * np.cos(a), rad * np.sin(a), kind) * dt) @ v
        pts.append(v[:3])
    return np.array(pts)


def panel(b, V, pts, col, title, notes):
    b += V.sphere()
    b += V.great_circle([0, 0, 1])
    b.append("\\draw[dashedink, line width=0.7pt] %s circle (%.3f);" % (N.pt(V.shift), np.linalg.norm(R0) * V.R))
    order = np.argsort(pts @ V.d)                                  # back to front
    for p in pts[order]:
        depth = 0.5 + 0.5 * (p @ V.d)
        b.append("\\fill[%s, opacity=%.2f] %s circle (0.75pt);" % (col, 0.25 + 0.55 * depth, V.pt(p)))
    b.append("\\node[dot, fill=ndInk, minimum size=4.6pt, label={[font=\\footnotesize]below:$\\vec r_0$}] at %s {};" % V.pt(R0))
    b.append("\\node[dot, minimum size=3.6pt, label={[font=\\footnotesize]above:$\\ket0$}] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[font=\\small] at %s {%s};" % (N.pt(V.shift + np.array([0, -V.R - 0.55])), title))
    for (x, y), text, target in notes:
        b.append("\\node[note, anchor=west] (n) at (%.2f,%.2f) {%s};" % (x, y, text))
        if target is not None:
            b.append("\\draw[pointer] (n.west) to[bend right=12] %s;" % V.pt(target))


def main():
    rng = np.random.default_rng(11)
    deph, damp = reachable("dephasing", rng), reachable("damping", rng)
    rd, ra = np.linalg.norm(deph, axis=1), np.linalg.norm(damp, axis=1)
    assert rd.max() <= np.linalg.norm(R0) + 1e-12                   # unital: purity never increases
    assert ra.max() > np.linalg.norm(R0) + 0.15 and damp[ra.argmax(), 2] > 0.5   # damping purifies towards |0>
    for pts in (deph, damp):                                          # non-empty interior: accessible
        assert np.linalg.eigvalsh(np.cov(pts.T)).min() > 1e-3
    print("max |r|: dephasing %.3f, damping %.3f (|r0| = %.1f)" % (rd.max(), ra.max(), np.linalg.norm(R0)))

    b = []
    Va = N.View(az=25.0, el=15.0, R=2.0, shift=(0.0, 0.0))
    Vb = N.View(az=25.0, el=15.0, R=2.0, shift=(7.4, 0.0))
    panel(b, Va, deph, "ndBlue", "(a) dephasing: unital",
          [((2.3, 1.7), "a region with interior:\\\\ \\emph{accessible}", deph[np.argmax(deph @ Va.d)] * 0.9),
           ((2.3, -1.3), "but never outside the\\\\ dashed sphere $\\abs{\\vec r}=\\abs{\\vec r_0}$:\\\\ not controllable", None)])
    panel(b, Vb, damp, "ndRed", "(b) amplitude damping: not unital",
          [((9.7, 1.7), "purer states are reached,\\\\ but only towards $\\ket0$", damp[np.argmax(damp[:, 2])] * 0.97),
           ((9.7, -1.3), "still not the whole ball:\\\\ not controllable", None)])
    N.write("ch04_accessibility", b)
    print("written figures/ch04_accessibility.tex")


if __name__ == "__main__":
    main()
