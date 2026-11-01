"""ch09_reach.py -- the maximum principle seen through the reachable sets.

The particle of the example in Chapter 9: x' = p, p' = f with |f| <= 1, to be
steered from X0 = (1, 0) to the origin in the least time.  The answer is
f = -1 on [0, 1] and f = +1 on [1, 2], so that t* = 2.

(a) The reachable sets R(t) from X0.  Since the system is linear and the
    controls form a convex set, R(t) is convex, with the support function
        h_t(a, b) = max_{Y in R(t)} (a, b).Y = a + int_0^t |a (t - s) + b| ds,
    attained by f(s) = sgn(a (t - s) + b): computing the boundary of R(t) is
    itself an application of the maximum principle, and every boundary point
    is reached by a control with at most one switch.  The optimal trajectory
    runs along the boundaries: X(t) lies on the boundary of R(t) for every t,
    and the adjoint lambda(t) = (-1, t - 1), which solves the adjoint
    equations and makes H_P = lambda.F - 1 vanish, is an outward normal there,
    h_t(lambda(t)) = lambda(t).X(t).  At t* = 2 the target is on the boundary
    of R(2), and R(2) lies on one side of the line through it orthogonal to
    lambda(2) = (-1, 1).
(b) The optimal synthesis.  Above the switching curve x = -p|p|/2 the optimal
    control is f = -1 and below it f = +1; every optimal trajectory switches at
    most once, on the curve, and then follows it to the origin.
(c) The switching function lambda_p(t) = t - 1, affine in t, and the control
    f(t) = sgn lambda_p(t).
Writes figures/ch09_reach.tex.
"""
import numpy as np
from scipy.integrate import quad
import needham as N

X0 = np.array([1.0, 0.0])
TSTAR = 2.0


def boundary(t, n=121):
    """The boundary of R(t), from the controls with one switch at tau (two arcs)."""
    tau = np.linspace(0, t, n)
    up = np.stack([1 + t * tau - tau ** 2 / 2 - (t - tau) ** 2 / 2, 2 * tau - t], 1)        # +1 then -1
    down = np.stack([1 - t * tau + tau ** 2 / 2 + (t - tau) ** 2 / 2, t - 2 * tau], 1)    # -1 then +1
    return np.vstack([up, down])


def support(t, a, b):
    return a * X0[0] + b * X0[1] + quad(lambda s: abs(a * (t - s) + b), 0, t, points=[t + b / a] if a else None)[0]


def optimal(t):
    """The optimal state X(t) and adjoint lambda(t)."""
    if t <= 1:
        X = np.array([1 - t ** 2 / 2, -t])
    else:
        s = t - 1
        X = np.array([0.5 - s + s ** 2 / 2, -1 + s])
    return X, np.array([-1.0, t - 1])


def synthesis(x0, p0, n=80):
    """The optimal trajectory from (x0, p0) to the origin: one bang to the curve x = -p|p|/2, then along it."""
    if x0 > -p0 * abs(p0) / 2:                       # above the curve: f = -1 until x = p^2/2, p < 0
        c = x0 + p0 ** 2 / 2
        p1 = -np.sqrt(c)
        ps = np.linspace(p0, p1, n)
        first = np.stack([c - ps ** 2 / 2, ps], 1)
        ps = np.linspace(p1, 0, n)
        second = np.stack([ps ** 2 / 2, ps], 1)
    else:                                            # below: f = +1 until x = -p^2/2, p > 0
        c = x0 - p0 ** 2 / 2
        p1 = np.sqrt(-c)
        ps = np.linspace(p0, p1, n)
        first = np.stack([c + ps ** 2 / 2, ps], 1)
        ps = np.linspace(p1, 0, n)
        second = np.stack([-ps ** 2 / 2, ps], 1)
    return first, second


def checks():
    # the optimal control reaches the origin at t* = 2, and the adjoint solves the adjoint equations with H_P = 0
    X, L = optimal(TSTAR)
    assert np.allclose(X, [0, 0])
    for t in np.linspace(0, 2, 41):
        X, L = optimal(t)
        f = np.sign(L[1]) if abs(L[1]) > 1e-12 else 1.0
        assert abs(L[0] * X[1] + L[1] * f - 1) < 1e-12              # H_P = lambda.F - 1 = 0 (f maximises it)
        # X(t) is on the boundary of R(t) and lambda(t) is an outward normal there
        if t > 0:
            assert abs(support(t, *L) - L @ X) < 1e-9
            for th in np.linspace(0, 2 * np.pi, 13):                # every point of R(t) is on the inner side
                d = np.array([np.cos(th), np.sin(th)])
                Y = boundary(t)[np.argmax(boundary(t) @ d)]
                assert L @ Y <= L @ X + 1e-9
    # the support function and the two-arc boundary agree
    for t in (0.5, 1.0, 2.0):
        B = boundary(t, 2001)
        for th in np.linspace(0, 2 * np.pi, 25):
            d = np.array([np.cos(th), np.sin(th)])
            assert abs((B @ d).max() - support(t, *d)) < 2e-5
    # no time shorter than 2 reaches the origin: the origin lies outside R(t) for t < 2
    for t in (1.5, 1.9, 1.99):
        X, L = optimal(t)
        assert support(t, -1.0, 1.0) < 0                             # the half-plane p - x <= h < 0 misses (0, 0)
    # the synthesis reaches the origin with one switch, on the switching curve
    for x0, p0 in [(1, 0), (-1.5, 0.6), (2.2, 1.2), (-0.6, -1.3), (0.3, 1.5)]:
        a, b = synthesis(x0, p0)
        assert np.allclose(b[-1], [0, 0]) and abs(a[-1][0] + a[-1][1] * abs(a[-1][1]) / 2) < 1e-12


def main():
    checks()
    b = []
    S = 1.2                                                        # cm per unit in (a) and (b)
    # ---- (a) reachable sets
    oa = np.array([1.5, 2.6])
    P = lambda v: oa + S * np.asarray(v)
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$x$};" % (N.pt(P([-1.25, 0])), N.pt(P([3.3, 0]))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$p$};" % (N.pt(P([0, -2.15])), N.pt(P([0, 2.25]))))
    for t, style, fill in [(2.0, "ink", "ndSky, fill opacity=0.55"), (1.0, "line width=0.7pt, ndBlue", "ndBlue!14")]:
        pts = [P(q) for q in boundary(t)]
        b.append("\\fill[%s] %s -- cycle;" % (fill, " -- ".join(N.pt(q) for q in pts)))
        b.append("\\draw[%s] %s -- cycle;" % (style, " -- ".join(N.pt(q) for q in pts)))
    # the supporting line at the target
    b.append("\\draw[dashedink] %s -- %s;" % (N.pt(P([-1.15, -1.15])), N.pt(P([1.9, 1.9]))))
    ts = np.linspace(0, 2, 121)
    b.append(N.polyline([P(optimal(t)[0]) for t in ts], "line width=1.3pt, ndRed"))
    for t in (1.0, 2.0):
        X, L = optimal(t)
        u = L / np.linalg.norm(L)
        b.append("\\draw[vec, ndGold!85!ndInk] %s -- %s;" % (N.pt(P(X)), N.pt(P(X) + 0.7 * u)))
        b.append("\\node[dot] at %s {};" % N.pt(P(X)))
    b.append("\\node[dot, fill=ndRed] at %s {};" % N.pt(P(X0)))
    b.append("\\node[font=\\scriptsize, anchor=south west] at %s {$X_0$};" % N.pt(P(X0) + [0.0, 0.03]))
    b.append("\\node[font=\\scriptsize, anchor=north] at %s {$X(1)$};" % N.pt(P(optimal(1.0)[0]) + [0.05, -0.08]))
    b.append("\\node[font=\\scriptsize, anchor=north west] at %s {$X(t^\\star)$};" % N.pt(P([0, 0]) + [0.04, -0.03]))
    b.append("\\node[font=\\scriptsize, text=ndBlue] at %s {$\\Reach(1)$};" % N.pt(P([1.2, 0.62])))
    b.append("\\node[font=\\scriptsize] at %s {$\\Reach(2)$};" % N.pt(P([2.55, 1.45])))
    b.append("\\node[note, anchor=east, text=ndGold!55!ndInk] (lam) at %s {adjoint $\\lambda(t)$,\\\\ an outward\\\\ normal at $X(t)$};" % N.pt(P([-0.75, 0.95])))
    b.append("\\draw[pointer] (lam.east) to[bend left=15] %s;" % N.pt(P([0, 0]) + 0.5 * np.array([-1, 1]) / np.sqrt(2) + [0.05, 0.08]))
    b.append("\\draw[pointer] (lam.south) to[bend right=20] %s;" % N.pt(P(optimal(1.0)[0]) + [-0.45, 0.04]))
    b.append("\\node[note, anchor=north east] (sup) at %s {$\\Reach(t^\\star)$ lies\\\\ below this line};" % N.pt(P([-0.35, -1.1])))
    b.append("\\draw[pointer] (sup.north east) to[bend right=10] %s;" % N.pt(P([-0.85, -0.85])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) reachable sets and the adjoint};" % N.pt(P([1.0, -2.25])))

    # ---- (b) the optimal synthesis
    ob = np.array([9.2, 2.6])
    Q = lambda v: ob + S * np.asarray(v)
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$x$};" % (N.pt(Q([-1.9, 0])), N.pt(Q([2.5, 0]))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$p$};" % (N.pt(Q([0, -2.15])), N.pt(Q([0, 2.25]))))
    ps = np.linspace(-1.9, 1.9, 161)
    curve = np.stack([-ps * np.abs(ps) / 2, ps], 1)
    b.append("\\fill[ndRose!55] %s -- %s -- %s -- cycle;" % (" -- ".join(N.pt(Q(q)) for q in curve), N.pt(Q([-1.9, 1.9])), N.pt(Q([-1.9, -1.9]))))
    b.append(N.polyline([Q(q) for q in curve], "line width=1.0pt, ndRed"))
    for x0, p0 in [(1, 0), (-1.5, 0.6), (1.3, 1.3), (-0.6, -1.3), (0.3, 1.5), (-1.2, -0.4)]:
        first, second = synthesis(x0, p0)
        ours = (x0, p0) == (1, 0)
        sty = "line width=1.3pt, ndRed" if ours else "line width=0.7pt, ndBlue"
        b.append(N.polyline([Q(q) for q in first], sty))
        k = len(first) // 2
        dirn = first[k + 1] - first[k - 1]
        b.append("\\draw[->, >=stealth, %s] %s -- %s;" % (sty, N.pt(Q(first[k])), N.pt(Q(first[k]) + 0.01 * dirn / np.linalg.norm(dirn))))
        b.append("\\node[dot, minimum size=2.6pt, fill=%s] at %s {};" % ("ndRed" if ours else "ndBlue", N.pt(Q([x0, p0]))))
        b.append("\\node[dot, minimum size=2.6pt] at %s {};" % N.pt(Q(first[-1])))
    b.append("\\node[dot] at %s {};" % N.pt(Q([0, 0])))
    b.append("\\node[font=\\scriptsize] at %s {$f=-1$};" % N.pt(Q([1.8, 1.9])))
    b.append("\\node[font=\\scriptsize] at %s {$f=+1$};" % N.pt(Q([-1.35, -1.75])))
    b.append("\\node[note, anchor=west, text=ndRed] at %s {switching curve\\\\ $x=-\\tfrac12 p\\abs{p}$};" % N.pt(Q([1.62, -1.62])))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the optimal synthesis};" % N.pt(Q([0.3, -2.25])))

    # ---- (c) the switching function
    pl = N.Plot((0, 2.15), (-1.25, 1.25), 3.4, 2.6, origin=(14.2, 1.3))
    b += pl.axes(xlabel="{\\scriptsize $t$}", xticks=[(1, "$1$"), (2, "$t^\\star$")], yticks=[(-1, "$-1$"), (1, "$1$")], yzero=0)
    b.append(pl.curve([0, 2], [-1, 1], "line width=1.0pt, ndGold!85!ndInk"))
    b.append(pl.curve([0, 1, 1, 2], [-1, -1, 1, 1], "line width=1.3pt, ndRed"))
    b.append("\\node[font=\\scriptsize, anchor=west, text=ndGold!55!ndInk] at %s {$\\lambda_p(t)$};" % pl.pt(0.12, -0.3))
    b.append("\\node[font=\\scriptsize, anchor=south, text=ndRed] at %s {$f(t)$};" % pl.pt(1.5, 1.02))
    b.append("\\node[note, anchor=north] (zz) at %s {$f$ switches\\\\ where $\\lambda_p$\\\\ changes sign};" % pl.pt(1.62, -0.42))
    b.append("\\draw[pointer] (zz.north west) to[bend left=25] %s;" % pl.pt(1.03, -0.05))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the switching function};" % pl.pt(1.05, -2.0))
    N.write("ch09_reach", b)


if __name__ == "__main__":
    main()
    print("written figures/ch09_reach.tex")
