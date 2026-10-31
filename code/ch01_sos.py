"""ch01_sos.py -- a nonnegative univariate polynomial, its minimum, and its
sum-of-squares certificate f(x) - f* = s1(x)^2 + s2(x)^2 (computed from the roots).
f(x) = x^4 - 3x^2 + x + 3; f - f* has a double root at the minimiser x*, and the
remaining quadratic factor is positive, which gives the two squares.
Writes figures/ch01_sos.tex.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N


def main():
    # f(x) = x^4 - 3x^2 + x + 3 ; f* = min f
    c = np.array([1, 0, -3, 1, 3.0])
    x = np.linspace(-2.2, 2.2, 500)
    f = np.polyval(c, x)
    xs = np.roots(np.polyder(c)); xs = xs[np.isreal(xs)].real
    fstar = np.polyval(c, xs).min(); xstar = xs[np.argmin(np.polyval(c, xs))]
    # g = f - f* >= 0 has a double root at x*; factor g = (x-x*)^2 (x^2 + b x + d) with x^2+bx+d = (x + b/2)^2 + (d - b^2/4)
    g = c.copy(); g[-1] -= fstar
    q, r = np.polydiv(g, np.array([1, -2 * xstar, xstar ** 2]))
    b, d = q[1], q[2]
    s1 = (x - xstar) * (x + b / 2)                # first square
    s2 = (x - xstar) * np.sqrt(max(d - b ** 2 / 4, 0))   # second square
    assert np.allclose(r, 0, atol=1e-9) and d - b ** 2 / 4 > 0
    assert np.allclose(f - fstar, s1 ** 2 + s2 ** 2, atol=1e-9)        # the certificate is an identity
    assert np.all(f - fstar >= -1e-12)

    out = []
    P = N.Plot((-2.2, 2.2), (-1.0, 8.0), 8.0, 5.2)
    out += P.axes("$x$", "", xticks=[(k, "$%d$" % k) for k in (-2, -1, 1, 2)],
                  yticks=[(0, "$0$"), (4, "$4$"), (8, "$8$")])
    out.append("\\draw[faint] %s -- %s;" % (P.pt(-2.2, 0.0), P.pt(2.2, 0.0)))
    clip = lambda y: np.where(y <= 8.0, y, np.nan)
    for y, style in [(fstar + s1 ** 2, "line width=1.0pt, ndGold"), (fstar + s2 ** 2, "line width=1.0pt, ndGreen"),
                     (f, "line width=1.4pt, ndBlue")]:
        m = clip(y) == clip(y)
        out.append(P.curve(x[m], y[m], style))
    out.append("\\draw[line width=0.9pt, ndRed, dash pattern=on 4pt off 2.5pt] %s -- %s;" % (P.pt(-2.2, fstar), P.pt(2.2, fstar)))
    out.append("\\node[dot, fill=ndRed, minimum size=4.5pt] at %s {};" % P.pt(xstar, fstar))
    out.append("\\node[font=\\small, anchor=north, text=ndRed] at %s {$x^\\star$};" % P.pt(xstar, fstar - 0.15))
    out.append("\\node[font=\\small, anchor=west, text=ndRed] at %s {$f^\\star=%.3f$};" % (P.pt(2.3, fstar), fstar))
    out.append("\\node[note, anchor=west, text=ndBlue] (f) at %s {$f(x)=x^4-3x^2+x+3$};" % P.pt(-1.78, 3.7))
    out.append("\\draw[pointer] (f.west) to[bend left=25] %s;" % P.pt(-1.95, np.polyval(c, -1.95) - 0.15))
    k1 = np.argmin(np.abs(fstar + s1 ** 2 - 5.0) + 10 * (x < 0))
    out.append("\\node[note, anchor=west, text=ndGold] (a) at %s {$f^\\star+s_1(x)^2$};" % P.pt(2.35, 5.6))
    out.append("\\draw[pointer] (a.west) to[bend left=20] %s;" % P.pt(x[k1] + 0.05, fstar + s1[k1] ** 2))
    k2 = np.argmin(np.abs(fstar + s2 ** 2 - 3.0) + 10 * (x < 0))
    out.append("\\node[note, anchor=west, text=ndGreen] (b2) at %s {$f^\\star+s_2(x)^2$};" % P.pt(2.35, 2.6))
    out.append("\\draw[pointer] (b2.west) to[bend left=15] %s;" % P.pt(x[k2] + 0.05, fstar + s2[k2] ** 2))
    out.append("\\node[note, anchor=north west, align=left] at %s {$f-f^\\star=s_1^2+s_2^2$ with\\\\ $s_1=(x-x^\\star)(x%+.3f)$,\\\\ $s_2=%.3f\\,(x-x^\\star)$: a certificate\\\\ that $f\\ge f^\\star$ everywhere};"
               % (P.pt(-1.35, 7.9), b / 2, np.sqrt(d - b ** 2 / 4)))
    N.write("ch01_sos", out)
    print("written figures/ch01_sos.tex  x*=%.3f f*=%.3f  s1=(x-x*)(x+%.3f) s2=%.3f(x-x*)" % (xstar, fstar, b / 2, np.sqrt(max(d - b ** 2 / 4, 0))))


if __name__ == "__main__":
    main()
