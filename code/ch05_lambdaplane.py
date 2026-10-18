"""ch05_lambdaplane.py -- the lambda-plane as a dictionary of behaviours.

Each marked point lambda of the complex plane comes with a thumbnail of the
mode Re e^{lambda t} it stands for: a constant at 0, decays on the negative real
axis (faster further left), undamped oscillations on the imaginary axis (faster
higher up), damped oscillations in between, and growth in the right half-plane,
which never occurs for a Lindbladian.  The four poles of the running example,
0, -gamma_1 and -Gamma +- i omega, are overlaid in red.  The two axes are drawn
to different scales.  Writes figures/ch05_lambdaplane.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import GAMMA, GAMMA1, OMEGA, POLES, qubit_A

SX, SY = 2.4, 0.72                      # cm per unit of Re lambda, Im lambda
TT = 6.0                                # time span of every thumbnail
TW, TH = 1.55, 0.5                      # thumbnail width, half-height (cm)


def P(z):
    return np.array([SX * z.real, SY * z.imag])


def thumbnail(lam, centre, col, text, label_pos="below"):
    """A small boxed plot of Re e^{lam t}, t in [0, TT], with values in [-1, 1] (clipped)."""
    cx, cy = centre
    x0, x1 = cx - TW / 2, cx + TW / 2
    out = ["\\fill[white] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0 - 0.06, cy - TH - 0.06, x1 + 0.06, cy + TH + 0.06),
           "\\draw[line width=0.4pt, ndInk!45] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0 - 0.06, cy - TH - 0.06, x1 + 0.06, cy + TH + 0.06),
           "\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (x0, cy, x1, cy)]
    t = np.linspace(0, TT, 300)
    v = np.exp(lam * t).real
    v = v / np.abs(v).max()              # a growing mode is shrunk to fit its box
    pts = [(x0 + TW * s / TT, cy + TH * w) for s, w in zip(t, v)]
    out.append(N.polyline(pts, "line width=0.9pt, %s" % col))
    out.append("\\node[font=\\scriptsize\\itshape, text=%s, %s=1pt, align=center] at (%.3f,%.3f) {%s};"
               % (col, label_pos, cx, cy - TH - 0.06 if label_pos == "below" else cy + TH + 0.06, text))
    return out


def main():
    lam = np.linalg.eigvals(qubit_A())
    assert all(np.min(abs(lam - p)) < 1e-12 for p in POLES)
    assert np.isclose(POLES[1], -GAMMA1) and np.isclose(POLES[2], -GAMMA + 1j * OMEGA)
    b = []
    xl, xr, yb, yt = -2.35, 1.0, -4.6, 4.6
    # the stable half-plane, axes, ticks
    b.append("\\fill[ndSky, fill opacity=0.5] (%.2f,%.2f) rectangle (0,%.2f);" % (SX * xl, SY * yb, SY * yt))
    b.append("\\fill[ndInk, fill opacity=0.06] (0,%.2f) rectangle (%.2f,%.2f);" % (SY * yb, SX * xr, SY * yt))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[right, text=ndInk] {$\\mathrm{Re}\\,\\lambda$};" % (SX * xl, SX * xr + 0.2))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[right, text=ndInk] {$\\mathrm{Im}\\,\\lambda$};" % (SY * yb, SY * yt + 0.2))
    for v in (-2, -1):
        b.append("\\draw[faint] (%.2f,0.06) -- (%.2f,-0.06);" % (SX * v, SX * v))
        b.append("\\node[font=\\scriptsize, text=ndInk!70, above right=-1pt] at (%.2f,0) {$%d$};" % (SX * v, v))
    for v in (-4, -2, 2, 4):
        b.append("\\draw[faint] (0.06,%.2f) -- (-0.06,%.2f);" % (SY * v, SY * v))
        b.append("\\node[font=\\scriptsize, text=ndInk!70, right=-1pt] at (0.04,%.2f) {$%d$};" % (SY * v, v))
    # sample points with their thumbnails: (lambda, thumbnail centre, colour, text, label side)
    samples = [
        (-1.6 + 0j, (-4.45, -1.1), "ndInk", "fast decay", "below"),
        (-1.6 + 3j, (-4.45, 3.0), "ndInk", "heavily damped", "above"),
        (0 + 1.5j, (-2.55, 1.35), "ndInk", "undamped, slow", "below"),
        (0 + 4j, (1.95, 2.8), "ndInk", "undamped, fast", "above"),
        (0.25 + 1.6j, (1.45, 0.95), "ndInk!45", "growth: impossible\\\\ for a Lindbladian", "above"),
    ]
    poles = [
        (POLES[0], (1.45, -1.1), "ndRed", "constant:\\\\ steady state", "below"),
        (POLES[1], (-2.25, -1.1), "ndRed", "slow decay", "below"),
        (POLES[2], (-1.6, 3.0), "ndRed", "damped oscillation", "above"),
    ]
    for z, c, col, text, side in samples + poles:
        p, c = P(z), np.array(c)
        if abs(p[0] - c[0]) < TW / 2:
            q = c + np.array([0.0, np.sign(p[1] - c[1]) * (TH + 0.06)])
        else:
            q = c + np.array([np.sign(p[0] - c[0]) * (TW / 2 + 0.06), 0.0])
        b.append("\\draw[pointer, ndInk!45] %s -- %s;" % (N.pt(p), N.pt(q)))
    for z, c, col, text, side in samples + poles:
        b += thumbnail(z, c, col, text, side)
    for z, c, col, text, side in samples:
        b.append("\\node[dot, fill=%s] at %s {};" % (col, N.pt(P(z))))
    # the running example in red, with labels
    for z, lab, pos in [(POLES[0], "$0$", "below right"), (POLES[1], "$-\\gamma_1$", "above left"),
                        (POLES[2], "$-\\Gamma+\\imath\\omega$", "left"), (POLES[3], "$-\\Gamma-\\imath\\omega$", "right")]:
        b.append("\\node[dot, fill=ndRed, minimum size=5pt, label={[font=\\scriptsize, text=ndRed, inner sep=1pt]%s:%s}] at %s {};" % (pos, lab, N.pt(P(z))))
    # the mirror pair
    p3 = P(POLES[3])
    b.append("\\node[note, anchor=east] (m) at (%.2f,%.2f) {conjugate pairs: $\\lambda$ and $\\bar\\lambda$\\\\ together give a real signal};" % (-1.75, -2.7))
    b.append("\\draw[pointer] (m.east) to[bend right=20] %s;" % N.pt(p3 + np.array([-0.12, -0.06])))
    # direction arrows along the edges
    ya = SY * yb - 0.35
    b.append("\\draw[vec, ndInk!70, line width=0.8pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (SX * -0.3, ya, SX * -2.2, ya))
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {further left: faster decay};" % (SX * -1.25, ya - 0.05))
    xa = SX * xl - 0.35
    b.append("\\draw[vec, ndInk!70, line width=0.8pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (xa, 0.3, xa, SY * 4.3))
    b.append("\\node[note, anchor=south, rotate=90] at (%.2f,%.2f) {higher: faster oscillation};" % (xa - 0.05, SY * 2.3))
    # legend
    xn = SX * xr + 0.9
    b.append("\\node[note, anchor=north west] at (%.2f,%.2f) {each box shows\\\\ $\\mathrm{Re}\\,\\ee^{\\lambda t}$ for the\\\\ point $\\lambda$ beside it};" % (xn, SY * 4.4))
    b.append("\\node[note, anchor=west, text=ndRed] at (%.2f,%.2f) {red: the four poles\\\\ of the example,\\\\ $\\Gamma=%.1f$, $\\gamma_1=%.1f$, $\\omega=%d$};" % (xn, -0.4, GAMMA, GAMMA1, OMEGA))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {blue: the stable half-plane,\\\\ where every Lindbladian\\\\ has its eigenvalues};" % (xn, -1.9))
    b.append("\\node[note, anchor=west, text=ndInk!60] at (%.2f,%.2f) {(the two axes have\\\\ different scales)};" % (xn, SY * yb - 0.2))
    N.write("ch05_lambdaplane", b)
    print("written figures/ch05_lambdaplane.tex")


if __name__ == "__main__":
    main()
