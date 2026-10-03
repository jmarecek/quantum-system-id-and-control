"""ch03_channels.py -- three qubit channels as images of the Bloch ball.

Each Lindblad evolution, run for a time t, maps the Bloch ball to an ellipsoid:
  amplitude damping  L = sqrt(g) |0><1|:      axes (e^{-gt/2}, e^{-gt/2}, e^{-gt}),
                                              centre (0, 0, 1 - e^{-gt}) -- pushed to |0>;
  dephasing          L = sqrt(g/2) sigma_z:   axes (e^{-gt}, e^{-gt}, 1) -- squashed onto the z-axis;
  depolarising       L_k = sqrt(g/4) sigma_k: axes (e^{-gt}, e^{-gt}, e^{-gt}) -- shrunk uniformly.
The formulas are checked against the exact evolution of random pure states.
Writes figures/ch03_channels.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N
from ch03_decoherence import lindbladian, vec, unvec

I, X, Y, Z = T.pauli()
SM = np.array([[0, 1], [0, 0]], complex)                 # |0><1|
GT = 0.9                                                 # g t
CASES = [
    ("(a) amplitude damping", [SM], (np.exp(-GT / 2), np.exp(-GT / 2), np.exp(-GT)), (0, 0, 1 - np.exp(-GT)), "ndRed"),
    ("(b) dephasing", [Z / np.sqrt(2)], (np.exp(-GT), np.exp(-GT), 1.0), (0, 0, 0), "ndBlue"),
    ("(c) depolarising", [X / 2, Y / 2, Z / 2], (np.exp(-GT),) * 3, (0, 0, 0), "ndGreen"),
]


def check(jumps, axes, centre, rng):
    L = lindbladian(0 * I, jumps)
    E = expm(L * GT)
    for _ in range(20):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        rho = 0.5 * (I + v[0] * X + v[1] * Y + v[2] * Z)
        r = T.bloch_vector(unvec(E @ vec(rho)))
        assert np.isclose(sum(((r - centre) / axes) ** 2), 1.0)


def panel(title, axes, centre, col, shift):
    V = N.View(az=25.0, el=18.0, R=1.7, shift=shift)
    b = [f for f in V.sphere(shade=False)]
    b[0] = b[0].replace("\\draw[ink]", "\\draw[faint, line width=0.6pt]")
    b += V.great_circle([0, 0, 1], "faint", "faint, densely dotted")
    out = N.ellipsoid_outline(V, centre, axes)
    b.append("\\filldraw[fill=%s!25, draw=%s, line width=1.0pt] %s -- cycle;" % (col, col, " -- ".join(N.pt(p) for p in out)))
    s = np.linspace(0, 2 * np.pi, 120)
    ring = np.stack([axes[0] * np.cos(s), axes[1] * np.sin(s), 0 * s], 1) + centre
    b += V.curve(ring, "line width=0.6pt, %s" % col, "line width=0.4pt, %s, densely dotted" % col)
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, text=ndInk, font=\\scriptsize] {$\\ket0$};" % (V.pt((0, 0, 0)), V.pt((0, 0, 1.3))))
    b.append("\\node[dot] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[dot, fill=%s] at %s {};" % (col, V.pt(centre)))
    b.append("\\node[font=\\small] at %s {%s};" % (N.pt(V.shift + (0, -2.25)), title))
    return b


def main():
    rng = np.random.default_rng(0)
    b = []
    for k, (title, jumps, axes, centre, col) in enumerate(CASES):
        check(jumps, np.array(axes), np.array(centre), rng)
        b += panel(title, axes, centre, col, (4.6 * k, 0.0))
    b.append("\\node[note] at (0,2.95) {shrunk, and pushed\\\\ towards the ground state};")
    b.append("\\node[note] at (4.6,2.95) {coherences die,\\\\ populations stay};")
    b.append("\\node[note] at (9.2,2.95) {every direction\\\\ shrinks alike};")
    N.write("ch03_channels", b)
    print("written figures/ch03_channels.tex")


if __name__ == "__main__":
    main()
