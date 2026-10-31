"""ch06_flat.py -- a flat approximation of the exponential, after Wodecki and Marecek.

A polynomial p is an (eps, eta, K)-flat approximation of e^x if |p - e^x| <= eps
on [-K, K] and |p(x)| <= max(1, e^x) e^{eta |x|} for every real x.  The
Chebyshev truncation of e^x on [-t, t] is
    f_n(x) = I_0(t) + 2 sum_{j=1}^n I_j(t) T_j(x / t),
and the construction of arXiv:2402.05552 is the product
    Q_{k,l}(x) = prod_{i=1}^k f_{2^i l}(x / k),
of degree D = (2^{k+1} - 2) l, which approximates e^x on [-kt, kt].  With
t = 1, k = 3 and l = 6 (the value of l that the paper prescribes for eps = 1e-3),
D = 84 and eta = 1/k = 1/3 is claimed.
  (a) log10 |p(x)| on [-150, 8] for e^x, for its Taylor polynomial and its
      Chebyshev truncation on [-3, 3], both of degree 84, and for Q_{3,6}, with
      the envelope max(1, e^x) e^{|x|/3} (shaded).  All three polynomials agree
      with e^x on [-3, 3] (Q to 3e-13); beyond it the truncations grow like
      |x|^84 / 84!, and Q grows less, but it leaves the envelope for x <= -50
      and exceeds it by a factor of 1.7e31 near x = -255.  The smallest eta for
      which Q is flat is 0.78 (dashed), against 0.96 for the truncations.
  (b) the least degree n for which the truncation is within eps = 1e-6 of e^x
      on [-t, t], for Taylor and for Chebyshev, against t, with the sufficient
      conditions e t + ln(1/eps) (Chebyshev) and 10 t + ln(1/eps) (Taylor) as
      quoted in the paper.  Both errors are attained at x = t, where every term
      is positive: they are the tails sum_{j>n} t^j/j! and sum_{j>n} 2 I_j(t).
The polynomials of (a) are evaluated in 80-digit decimal arithmetic, since
near the interval of approximation their terms cancel to many digits.
Writes figures/ch06_flat.tex.
"""
import math
from decimal import Decimal, getcontext

import numpy as np
from scipy.special import iv
import needham as N

getcontext().prec = 80
T_, K_, L_ = 1, 3, 6
ETA = 1.0 / K_
DEGS = [2 ** i * L_ for i in range(1, K_ + 1)]
DEG = sum(DEGS)                                        # 84
EPS_B = 1e-6


def bessel_i(j, t, terms=80):
    """I_j(t) in decimal arithmetic, from its power series."""
    t2 = Decimal(t) / 2
    term = t2 ** j / math.factorial(j)
    s = Decimal(0)
    for m in range(terms):
        s += term
        term = term * t2 * t2 / ((m + 1) * (m + 1 + j))
    return s


def cheb_coefs(t, n):
    return [bessel_i(0, t)] + [2 * bessel_i(j, t) for j in range(1, n + 1)]


def clenshaw(y, c):
    b1 = b2 = Decimal(0)
    for a in reversed(c[1:]):
        b1, b2 = 2 * y * b1 - b2 + a, b1
    return y * b1 - b2 + c[0]


def taylor(x, n):
    s, term = Decimal(0), Decimal(1)
    for j in range(n + 1):
        s += term
        term = term * x / (j + 1)
    return s


COEF_T = {n: cheb_coefs(T_, n) for n in DEGS}
COEF_K = cheb_coefs(K_ * T_, DEG)


def Q(x):
    y = Decimal(x) / (K_ * T_)
    p = Decimal(1)
    for n in DEGS:
        p *= clenshaw(y, COEF_T[n])
    return p


def cheb_K(x):
    return clenshaw(Decimal(x) / (K_ * T_), COEF_K)


def log10abs(v):
    return float(abs(v).log10())


def least_degree_taylor(t, eps):
    terms = np.cumprod([1.0] + [t / j for j in range(1, 400)])      # t^j / j!
    tail = np.cumsum(terms[::-1])[::-1]                 # tail[n] = sum_{j>=n}
    return next(n for n in range(399) if tail[n + 1] < eps)


def least_degree_cheb(t, eps):
    c = 2 * iv(np.arange(0, 400), t)
    tail = np.cumsum(c[::-1])[::-1]
    return next(n for n in range(399) if tail[n + 1] < eps)


def main():
    # ---- the claims checked
    xs_in = np.linspace(-K_ * T_, K_ * T_, 121)
    err_in = max(abs(float(Q(x)) - math.exp(x)) for x in xs_in)
    assert err_in < 1e-12
    for f in (lambda x: taylor(Decimal(x), DEG), cheb_K):
        assert max(abs(float(f(x)) - math.exp(x)) for x in xs_in) < 1e-40
    far = np.arange(-600, 0)
    ln_q = np.array([log10abs(Q(int(x))) * math.log(10) for x in far])
    ln_t = np.array([log10abs(taylor(Decimal(int(x)), DEG)) * math.log(10) for x in far])
    eta_q = np.max(ln_q / np.abs(far))
    eta_t = np.max(ln_t / np.abs(far))
    excess = np.max((ln_q - ETA * np.abs(far)) / math.log(10))          # log10 of the worst |Q|/envelope
    x_excess = far[np.argmax(ln_q - ETA * np.abs(far))]
    outside = far[ln_q > ETA * np.abs(far)]
    assert abs(eta_q - 0.777) < 0.005 and abs(eta_t - 0.956) < 0.005
    assert abs(excess - 31.24) < 0.1 and abs(x_excess + 255) <= 2
    assert outside.max() == -50 and outside.min() == -600
    # ---- (a) the polynomials on a logarithmic scale
    b = []
    xlo, xhi, ylo, yhi = -150.0, 8.0, -12.0, 60.0
    Pl = N.Plot((xlo, xhi), (ylo, yhi), 7.4, 4.8, origin=(0.0, 0.0))
    b += Pl.axes("$x$", "$\\log_{10}\\abs{p(x)}$", xticks=[(-150, "$-150$"), (-100, "$-100$"), (-50, "$-50$"), (0, "$0$")],
                 yticks=[(-10, "$-10$"), (0, "$0$"), (20, "$20$"), (40, "$40$"), (60, "$60$")])
    xx = np.linspace(xlo, xhi, 400)
    env = lambda x, eta: (max(0.0, x) + eta * abs(x)) / math.log(10)
    poly = [N.pt(Pl.P(x, ylo)) for x in xx] + [N.pt(Pl.P(x, min(env(x, ETA), yhi))) for x in xx[::-1]]
    b.append("\\fill[ndSky, fill opacity=0.6] %s -- cycle;" % " -- ".join(poly))
    b.append(Pl.curve(xx, [env(x, eta_q) for x in xx], "dashedink"))
    b.append("\\draw[faint] %s -- %s;" % (Pl.pt(xlo, 0), Pl.pt(xhi, 0)))
    ex = xx[xx / math.log(10) >= ylo]
    b.append(Pl.curve(ex, ex / math.log(10), "line width=1.0pt, ndInk"))
    clip = lambda v: min(max(v, ylo), yhi)
    yt = [clip(log10abs(taylor(Decimal(x), DEG))) for x in xx]
    yc = [clip(log10abs(cheb_K(x))) for x in xx]
    yq = [clip(log10abs(Q(x))) for x in xx]
    b.append(Pl.curve(xx, yc, "line width=1.0pt, ndGold"))
    b.append(Pl.curve(xx, yt, "line width=0.8pt, ndGold!60!ndInk, dash pattern=on 3pt off 2pt"))
    b.append(Pl.curve(xx, yq, "line width=1.3pt, ndRed"))
    b.append("\\draw[line width=2pt, ndGreen] %s -- %s;" % (Pl.pt(-3, ylo), Pl.pt(3, ylo)))
    b.append("\\node[note, anchor=west, text=ndGold!80!ndInk] (c) at %s {Taylor and Chebyshev\\\\ truncations, degree $%d$};" % (Pl.pt(-42, 56.0), DEG))
    b.append("\\draw[pointer] (c.west) to[bend right=15] %s;" % Pl.pt(-70, log10abs(cheb_K(-70)) + 1.0))
    b.append("\\node[note, anchor=west, text=ndRed] (q) at %s {the product $Q_{3,6}$,\\\\ also of degree $%d$};" % (Pl.pt(-30, 38.0), DEG))
    b.append("\\draw[pointer] (q.west) to[bend right=15] %s;" % Pl.pt(-62, log10abs(Q(-62)) + 1.0))
    b.append("\\node[note, anchor=west] (d) at %s {dashed: $\\eta=%.2f$,\\\\ the least that $Q$ admits};" % (Pl.pt(-22, 22.0), eta_q))
    b.append("\\draw[pointer] (d.west) to[bend right=15] %s;" % Pl.pt(-36, env(-36, eta_q) + 0.8))
    b.append("\\node[note, anchor=west] at %s {shaded: $\\abs{p}\\le\\ee^{\\abs x/3}$,\\\\ flatness with $\\eta=1/3$};" % Pl.pt(-147, 8.0))
    b.append("\\node[font=\\small, anchor=west] at %s {$\\ee^x$};" % Pl.pt(-13, -9.8))
    b.append("\\node[font=\\small] at %s {(a) $\\abs{p(x)}$ on a logarithmic scale};" % N.pt(Pl.P((xlo + xhi) / 2, ylo) + np.array([0.0, -0.95])))
    # ---- (b) the least degree against t
    ts = np.arange(1, 41)
    nt = np.array([least_degree_taylor(float(t), EPS_B) for t in ts])
    nc = np.array([least_degree_cheb(float(t), EPS_B) for t in ts])
    L = math.log(1 / EPS_B)
    assert np.all(nc < nt) and np.all(nt <= math.e * ts + L + 1) and np.all(nc <= math.e * ts + L)
    assert nt[-1] == 119 and nc[-1] == 69
    R = N.Plot((0, 40), (0, 160), 6.2, 4.8, origin=(9.9, 0.0))
    b += R.axes("$t$", "least degree $n$", xticks=[(10, "$10$"), (20, "$20$"), (30, "$30$"), (40, "$40$")],
                yticks=[(50, "$50$"), (100, "$100$"), (150, "$150$")])
    tt = np.linspace(0, 40, 50)
    b.append(R.curve(tt, math.e * tt + L, "dashedink"))
    t10 = tt[10 * tt + L <= 160]
    b.append(R.curve(t10, 10 * t10 + L, "dashedink"))
    b += ["\\fill[ndGold!60!ndInk] %s circle (1.1pt);" % R.pt(t, n) for t, n in zip(ts, nt)]
    b += ["\\fill[ndBlue] %s circle (1.1pt);" % R.pt(t, n) for t, n in zip(ts, nc)]
    b.append("\\node[note, anchor=east] at %s {$10t+\\ln\\frac1\\eps$};" % R.pt(12.3, 140))
    b.append("\\node[note, anchor=west] at %s {$\\ee t+\\ln\\frac1\\eps$};" % R.pt(40.5, math.e * 40 + L - 4))
    b.append("\\node[note, anchor=east, text=ndGold!80!ndInk] (a) at %s {Taylor};" % R.pt(21, 98))
    b.append("\\draw[pointer] (a.east) to[bend left=15] %s;" % R.pt(25.6, nt[25] + 3))
    b.append("\\node[note, anchor=west, text=ndBlue] (bb) at %s {Chebyshev};" % R.pt(26, 22))
    b.append("\\draw[pointer] (bb.north) to[bend right=15] %s;" % R.pt(28, nc[27] - 3))
    b.append("\\node[note, anchor=west] at %s {dots: the least degree\\\\ found; dashed: the\\\\ sufficient conditions};" % R.pt(16, 150))
    b.append("\\node[font=\\small, align=center] at %s {(b) the least degree for an error of $10^{-6}$ on $[-t,t]$};" % N.pt(R.P(20, 0) + np.array([0.0, -0.95])))
    N.write("ch06_flat", b)
    print("written figures/ch06_flat.tex; Q_{3,6}: degree %d, error on [-3,3] %.1e, least eta %.3f (truncations %.3f);"
          " |Q| exceeds e^{|x|/3} for x <= %d, by up to 10^%.1f at x = %d" % (DEG, err_in, eta_q, eta_t, outside.max(), excess, x_excess))
    print("least degrees at t = 10, 20, 40: Taylor", nt[[9, 19, 39]], " Chebyshev", nc[[9, 19, 39]])


if __name__ == "__main__":
    main()
