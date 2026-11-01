"""ch10_gamma.py -- why pointwise convergence is not enough.

The functions on the real line
    F_n(x) = (x - 1)^2 - 2 exp(-n^2 (x - 1/n)^2)
have a narrow well of depth about 2 and width 1/n at x = 1/n.  Their
minimisers x_n lie near 1/n (0.556, 0.273, 0.132, 0.064 for n = 2, 4, 8, 16)
and F_n(x_n) -> -1 (-1.778, -1.455, -1.240, -1.123).

(a) F_n for n = 2, 4 and 8, with their minimisers.
(b) The two limits.  The pointwise limit is F(x) = (x - 1)^2 for x != 0, since
    the well moves away from every fixed x != 0, and F(0) = 1 - 2/e = 0.264,
    since the well's flank passes through 0 at the same height for every n.  Its
    minimiser is x = 1, where F = 0, and not the point 0 at which the minimisers
    of F_n accumulate.  The Gamma-limit agrees with F except at 0, where it is
    -1: along the recovery sequence x_n = 1/n, F_n(x_n) = (1/n - 1)^2 - 2 -> -1,
    and no sequence x_n -> 0 does better, since F_n >= (x - 1)^2 - 2.  The
    minimiser of the Gamma-limit is 0, as the fundamental theorem requires.
Writes figures/ch10_gamma.tex.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import needham as N

NS = (2, 4, 8)
COLS = {2: "ndGold!85!ndInk", 4: "ndGreen", 8: "ndBlue"}


def F(x, n):
    return (x - 1) ** 2 - 2 * np.exp(-n ** 2 * (x - 1 / n) ** 2)


def minimiser(n):
    xs = np.linspace(-1, 3, 400001)
    x0 = xs[np.argmin(F(xs, n))]
    r = minimize_scalar(lambda x: F(x, n), bounds=(x0 - 0.5 / n, x0 + 0.5 / n), method="bounded",
                        options={"xatol": 1e-12})
    return r.x, r.fun


def gamma_limit(x):
    return np.where(np.abs(x) < 1e-12, -1.0, (x - 1) ** 2)


def checks():
    expected = {2: (0.556, -1.778), 4: (0.273, -1.455), 8: (0.132, -1.240), 16: (0.064, -1.123)}
    for n, (xe, fe) in expected.items():
        x, f = minimiser(n)
        assert abs(x - xe) < 1e-3 and abs(f - fe) < 1e-3
        assert abs(F(0.0, n) - (1 - 2 / np.e)) < 1e-12                 # F_n(0) does not depend on n
    # minimisers accumulate at 0 and the minima tend to -1
    for n in (64, 256, 1024):
        x, f = minimiser(n)
        assert abs(x) < 1.1 / n and abs(f + 1) < 2.5 / n
    # pointwise limit at x != 0 is (x - 1)^2
    for x in (-0.3, 0.05, 0.5, 1.0, 2.0):
        assert abs(F(x, 4000) - (x - 1) ** 2) < 1e-9
    # recovery sequence x_n = 1/n, and the lower bound F_n >= (x-1)^2 - 2
    for n in (10, 100, 1000):
        assert abs(F(1 / n, n) - ((1 / n - 1) ** 2 - 2)) < 1e-12
        xs = np.linspace(-0.2, 0.2, 2001)
        assert np.all(F(xs, n) >= (xs - 1) ** 2 - 2 - 1e-12)
    # the minimiser of the pointwise limit is 1 (value 0 < F(0) = 0.264), that of the Gamma-limit is 0
    assert 0 < 1 - 2 / np.e


def main():
    checks()
    b = []
    xl, yl = (-0.45, 2.1), (-2.0, 2.2)
    # ---- (a) the functions F_n
    pa = N.Plot(xl, yl, 6.4, 4.2, origin=(0.6, 0.6))
    b += pa.axes(xlabel="{\\scriptsize $x$}", xticks=[(1, "$1$"), (2, "$2$")], yticks=[(-1, "$-1$"), (1, "$1$")], yzero=0)
    b.append("\\draw[faint] %s -- %s;" % (pa.pt(0, yl[0]), pa.pt(0, yl[1])))
    for n in NS:
        xs = np.linspace(xl[0], xl[1], 1601)
        b.append(pa.curve(xs, F(xs, n), "line width=1.0pt, %s" % COLS[n]))
        x, f = minimiser(n)
        b.append("\\node[dot, fill=%s] at %s {};" % (COLS[n], pa.pt(x, f)))
        b.append("\\node[font=\\scriptsize, anchor=north, text=%s] at %s {$x_{%d}$};" % (COLS[n], pa.pt(x, f - 0.06), n))
    for n, (xt, yt, anc) in {2: (0.95, -1.45, "west"), 4: (1.18, -0.42, "west"), 8: (0.44, 0.42, "south west")}.items():
        b.append("\\node[font=\\scriptsize, anchor=%s, text=%s] at %s {$F_{%d}$};" % (anc, COLS[n], pa.pt(xt, yt), n))
    b.append("\\node[note, anchor=west] at %s {a well of width $1/n$\\\\ moving towards $0$};" % pa.pt(0.75, 1.65))
    b.append("\\node[note, anchor=east] (all) at %s {all pass through\\\\ $(0,\\,1-2/e)$};" % pa.pt(-0.12, -1.75))
    b.append("\\draw[pointer] (all.north) to[bend left=25] %s;" % pa.pt(-0.02, 1 - 2 / np.e - 0.05))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) the functions $F_n$ and their minimisers $x_n$};" % pa.pt(0.8, yl[0] - 0.25))

    # ---- (b) the pointwise limit and the Gamma-limit
    pb = N.Plot(xl, yl, 6.4, 4.2, origin=(9.0, 0.6))
    b += pb.axes(xlabel="{\\scriptsize $x$}", xticks=[(1, "$1$"), (2, "$2$")], yticks=[(-1, "$-1$"), (1, "$1$")], yzero=0)
    b.append("\\draw[faint] %s -- %s;" % (pb.pt(0, yl[0]), pb.pt(0, yl[1])))
    xs = np.linspace(xl[0], xl[1], 801)
    b.append(pb.curve(xs, (xs - 1) ** 2, "line width=1.1pt, ndInk"))
    b.append("\\draw[ndInk, fill=white, line width=0.7pt] %s circle (1.8pt);" % pb.pt(0, 1.0))
    b.append("\\node[dot, fill=ndRose!90!ndRed] at %s {};" % pb.pt(0, 1 - 2 / np.e))
    b.append("\\node[dot, fill=ndRed, minimum size=4pt] at %s {};" % pb.pt(0, -1))
    b.append("\\node[dot, fill=ndGold, minimum size=4pt] at %s {};" % pb.pt(1, 0))
    # the recovery sequence: the minima of F_n descend onto (0, -1)
    for n in (2, 4, 8, 16):
        x, f = minimiser(n)
        b.append("\\node[circle, fill=%s, inner sep=0pt, minimum size=2.2pt] at %s {};" % ("ndBlue!70", pb.pt(x, f)))
    pts = [minimiser(n) for n in (2, 4, 8, 16, 32, 64)]
    b.append(N.polyline([pb.P(x, f) for x, f in pts] + [pb.P(0, -1)], "->, >=stealth, line width=0.6pt, ndBlue!70, dash pattern=on 2pt off 1.5pt"))
    b.append("\\node[note, anchor=west, text=ndRose!40!ndRed] (pw) at %s {pointwise limit at $0$:\\\\ $1-2/e=0.264$};" % pb.pt(0.38, 1.55))
    b.append("\\draw[pointer] (pw.west) to[bend right=15] %s;" % pb.pt(0.03, 1 - 2 / np.e + 0.05))
    b.append("\\node[note, anchor=west, text=ndRed] (gl) at %s {$\\Gamma$-limit at $0$:\\\\ $-1$, its minimum};" % pb.pt(0.08, -0.5))
    b.append("\\draw[pointer] (gl.south west) to[bend right=25] %s;" % pb.pt(0.03, -0.94))
    b.append("\\node[note, anchor=west, text=ndBlue] (rs) at %s {minima of $F_n$:\\\\ a recovery sequence};" % pb.pt(0.78, -1.6))
    b.append("\\draw[pointer] (rs.west) to[bend left=10] %s;" % pb.pt(0.6, -1.74))
    b.append("\\node[note, anchor=south, text=ndGold!55!ndInk] (pm) at %s {minimum of the\\\\ pointwise limit};" % pb.pt(1.0, 0.38))
    b.append("\\draw[pointer] (pm.south) -- %s;" % pb.pt(1.0, 0.07))
    b.append("\\node[font=\\scriptsize, anchor=south west] at %s {$(x-1)^2$};" % pb.pt(1.78, 1.05))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the pointwise limit and the $\\Gamma$-limit};" % pb.pt(0.8, yl[0] - 0.25))
    N.write("ch10_gamma", b)


if __name__ == "__main__":
    main()
    for n in (2, 4, 8, 16):
        print(n, "minimiser %.4f, minimum %.4f" % minimiser(n))
    print("F(0) = 1 - 2/e = %.4f" % (1 - 2 / np.e))
    print("written figures/ch10_gamma.tex")
