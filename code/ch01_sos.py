"""ch01_sos.py -- a nonnegative univariate polynomial, its minimum, and its
sum-of-squares certificate f(x) - f* = s1(x)^2 + s2(x)^2 (computed from the roots).
Writes figures/ch01_sos.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

# f(x) = x^4 - 3x^2 + x + 3 ; f* = min f
c = np.array([1, 0, -3, 1, 3.0])
x = np.linspace(-2.2, 2.2, 500)
f = np.polyval(c, x)
xs = np.roots(np.polyder(c)); xs = xs[np.isreal(xs)].real; fstar = np.polyval(c, xs).min(); xstar = xs[np.argmin(np.polyval(c, xs))]
# g = f - f* >= 0 has a double root at x*; factor g = (x-x*)^2 (x^2 + b x + d) with x^2+bx+d > 0 = (x + b/2)^2 + (d - b^2/4)
g = c.copy(); g[-1] -= fstar
q, r = np.polydiv(g, np.array([1, -2 * xstar, xstar ** 2]))
b, d = q[1], q[2]
s1 = (x - xstar) * (x + b / 2)                # first square
s2 = (x - xstar) * np.sqrt(max(d - b ** 2 / 4, 0))   # second square
series = [dict(x=x, y=f, label="$f(x)=x^4-3x^2+x+3$"),
          dict(x=x, y=np.full_like(x, fstar), label="$f^\\star=%.3f$" % fstar, style="dashed"),
          dict(x=x, y=fstar + s1 ** 2, label="$f^\\star+s_1(x)^2$", style="dotted, thick"),
          dict(x=x, y=fstar + s2 ** 2, label="$f^\\star+s_2(x)^2$", style="dotted, thick")]
body = T.pgfplots_axis(series, xlabel="$x$", ylabel="", width="0.8\\linewidth", height="0.45\\linewidth", legend_pos="north west",
                       axis_options="ymin=0, ymax=9")
T.write_tikz("ch01_sos", body)
print("written figures/ch01_sos.tex  x*=%.3f f*=%.3f  s1=(x-x*)(x+%.3f) s2=%.3f(x-x*)" % (xstar, fstar, b / 2, np.sqrt(max(d - b ** 2 / 4, 0))))
