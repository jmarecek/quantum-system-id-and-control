"""ch01_duality.py -- a conic program in the plane: minimise c^T x over the
intersection of the second-order cone {(x1,x2,x3): x3 >= |(x1,x2)|} with the
affine plane x3 = 1 (a disc) and a line a^T x = b; the optimum, the level sets of
the objective and the dual certificate.  The dual is max_y  b y - ||c - a y||_2;
at the optimum y*, the vector s = c - a y* is a positive multiple of -x*, the
inward normal of the disc at x*, so that c = y* a + s certifies the bound.
Writes figures/ch01_duality.tex.
"""
import os, sys
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N


def main():
    c = np.array([1.0, 0.6]); c /= np.linalg.norm(c)
    a = np.array([0.3, 1.0]); a /= np.linalg.norm(a); b = 0.35
    # feasible set: unit disc intersect line a.x = b  -> a chord; minimise c.x on the chord
    p0 = b * a; tdir = np.array([-a[1], a[0]]); h = np.sqrt(1 - b ** 2)
    e1, e2 = p0 - h * tdir, p0 + h * tdir
    xopt = e1 if c @ e1 < c @ e2 else e2
    val = c @ xopt
    dual = minimize_scalar(lambda y: -(b * y - np.linalg.norm(c - a * y)), bounds=(-10, 10), method="bounded",
                           options=dict(xatol=1e-12))
    ystar = dual.x
    s = c - a * ystar
    assert abs((b * ystar - np.linalg.norm(s)) - val) < 1e-9      # no duality gap
    assert np.allclose(s / np.linalg.norm(s), -xopt, atol=1e-6)  # s is the inward normal at x*

    S = 2.2
    P = lambda p: S * np.asarray(p)
    out = []
    out.append("\\fill[ndSky, fill opacity=0.6] (0,0) circle (%.2f);" % S)
    out.append("\\draw[ink] (0,0) circle (%.2f);" % S)
    nrm = np.array([-c[1], c[0]])
    for k in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        hl = np.sqrt(1.2 ** 2 - k ** 2)
        out.append("\\draw[dashedink] %s -- %s;" % (N.pt(P(k * c - hl * nrm)), N.pt(P(k * c + hl * nrm))))
    hl = np.sqrt(1.2 ** 2 - val ** 2)
    out.append("\\draw[line width=0.9pt, ndRed] %s -- %s;" % (N.pt(P(val * c - hl * nrm)), N.pt(P(val * c + hl * nrm))))
    out.append("\\draw[line width=2pt, ndBlue] %s -- %s;" % (N.pt(P(e1)), N.pt(P(e2))))
    o = np.array([0.05, -0.6])
    out.append("\\draw[vec, ndInk] %s -- %s node[pos=1.0, right, font=\\small] {$c$};" % (N.pt(P(o)), N.pt(P(o + 0.45 * c))))
    # the certificate at x*: c = y* a + s, drawn to half scale
    k = 0.85
    out.append("\\draw[vec, ndGold] %s -- %s;" % (N.pt(P(xopt)), N.pt(P(xopt + k * s))))
    out.append("\\node[font=\\small, anchor=north, text=ndGold] at %s {$s$};" % N.pt(P(xopt + k * s) + np.array([0.0, -0.08])))
    out.append("\\draw[vec, ndGreen] %s -- %s node[pos=0.6, right, font=\\small, text=ndGreen] {$y^\\star a$};" % (N.pt(P(xopt + k * s)), N.pt(P(xopt + k * c))))
    out.append("\\draw[vec, ndInk!70, line width=0.8pt] %s -- %s;" % (N.pt(P(xopt)), N.pt(P(xopt + k * c))))
    out.append("\\node[dot, fill=ndRed, minimum size=4.5pt] at %s {};" % N.pt(P(xopt)))
    out.append("\\node[font=\\small, anchor=east, text=ndRed] at %s {$x^\\star$};" % N.pt(P(xopt) + np.array([-0.12, -0.05])))
    out.append("\\node[note, anchor=south, text=ndBlue] (fs) at %s {the feasible set: the chord\\\\ $a^\\top x=b$ of the disc $\\norm{x}_2\\le1$};" % N.pt((1.75, 2.55)))
    xo = e2 if xopt is e1 else e1
    out.append("\\draw[pointer] (fs.south) to[bend left=20] %s;" % N.pt(P(xopt + 0.8 * (xo - xopt)) + np.array([0.0, 0.06])))
    out.append("\\node[note, anchor=north, text=ndRed] (ls) at %s {the level set $c^\\top x=c^\\top x^\\star$\\\\ meets the chord only at $x^\\star$};" % N.pt((-2.3, -2.25)))
    out.append("\\draw[pointer] (ls.north) to[bend left=15] %s;" % N.pt(P(val * c - 0.95 * nrm)))
    tx = 3.15
    out.append("\\node[anchor=west, font=\\small] at (%.2f,1.7) {primal: $\\min\\ c^\\top x$ s.t. $a^\\top x=b$, $\\norm{x}_2\\le1$};" % tx)
    out.append("\\node[anchor=west, font=\\small] at (%.2f,1.1) {dual: $\\max_y\\ b\\,y-\\norm{c-a\\,y}_2$};" % tx)
    out.append("\\node[note, anchor=north west, align=left] at (%.2f,0.6) {both equal $%.3f$: no duality gap.\\\\[3pt] The certificate is $c=y^\\star a+s$: the gold\\\\ vector $s=c-a\\,y^\\star$ points from $x^\\star$ to the\\\\ centre, normal to the disc, so $s^\\top x\\ge s^\\top x^\\star$\\\\ on the disc and $c^\\top x\\ge c^\\top x^\\star$ on the chord};" % (tx, val))
    N.write("ch01_duality", out)
    print("written figures/ch01_duality.tex  value %.3f  y* = %.4f" % (val, ystar))


if __name__ == "__main__":
    main()
