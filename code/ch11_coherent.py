"""ch11_coherent.py -- coherent against stochastic errors in a train of pulses, certified by state lifting.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi) and cvxpy with
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

A qubit starts in |0> and receives n pulses, each meant to turn it through pi
about x (the X train) or about x, y, x, y in turn (the XY train, whole cycles
of four), but the
k-th turns it through pi + eps_k, with |eps_k| <= delta = 0.1.  The error of
the train is the infidelity 1 - |<ideal|actual>|^2 of the final state, or the
angle phi = 2 arcsin sqrt(infidelity) between the actual and the ideal Bloch
vectors.  Three kinds of error are compared:
  coherent, worst case    the largest infidelity over all eps in [-delta, delta]^n;
  coherent, systematic    eps_k = delta for every k;
  stochastic              eps_k independent and uniform on [-delta, delta].
In the X train all the turns share an axis, the errors simply add, and the
infidelity is sin^2((eps_1 + ... + eps_n)/2): the worst case is the systematic
one, sin^2(n delta/2), the angle n delta, while for random errors
E[infidelity] = (1 - (sin(delta)/delta)^n)/2 ~ n delta^2/12 and the rms angle
is sqrt(n/3) delta.  In the XY train the turns do not commute.  Equal errors
nearly cancel (infidelity 1.2e-7, 5.0e-7, 1.1e-6 at n = 4, 8, 12), but the
worst case does not: it is about half that of the X train (angle n delta/sqrt(2)
to 0.1%; infidelity 0.0198, 0.0778, 0.1692 at n = 4, 8, 12), while random errors
do as in the X train.  Over n = 4 ... 12 the worst cases grow by a factor of 8
to 9, the random errors by 3.

The worst case is certified by state lifting: the Bloch vector after the k-th
pulse becomes a vector of new variables r_k = R(r_{k-1}, c_k, s_k), where
(c_k, s_k) = (cos eps_k, sin eps_k) obey c^2 + s^2 = 1 and c >= cos(delta),
and pysparsepmi.composition_lower_bound (SL-chord, order 2, SCS with eps = 1e-8)
bounds the largest infidelity from above.  The X train needs only (y, z), the
XY train the whole Bloch vector; the largest PSD block has the order 28 (X) and
36 (XY) for every n >= 4, against binom(5n/2, n/2) (53130 at n = 10) for the
dense relaxation of the lowest order, of degree n in the 2n variables.

Per pulse, the infidelity is sin^2(delta/2) = 2.5e-3, but the diamond distance
from the ideal pulse is sin(delta/2) = 0.050 (Chapter 8, by code/ch08_diamond.py),
and it is diamond distances that add along a train: the infidelity after n
pulses is at most n^2 sin^2(delta/2), whatever the input.  The X train comes
within 9% of this bound at n = 10; the XY train stays at about half of it.

(a) The two trains, and the Bloch vector of the X train after every second
    pulse, for equal errors eps_k = delta (it walks away from |0> by 2 delta a
    time, 1.6 rad after 16 pulses) and for one draw of random errors.
(b) The angle phi / delta against n: the certified worst cases (dots; X for
    n = 2, 4, ..., 10, XY for n = 4 and 8), the exact worst cases (X: n; XY: by
    enumeration of the 2^n vertices and local search from 20 random starts,
    n = 4, 8, 12), the systematic error of the XY train, and the rms angle of
    20000 random trains against sqrt(n/3).
(c) The infidelity on logarithmic scales: the worst cases grow as n^2, the
    random errors as n, and the bound n^2 sin^2(delta/2) of Chapter 8.
(d) The solve times with SCS on this machine.
Every certified value agrees with the closed form (X) or with the enumeration
(XY) to 1e-6, and every SCS status is "optimal".  The independent checks
compute the state of the qubit with 2x2 unitaries, not with the Bloch maps
given to pysparsepmi.
Writes figures/ch11_coherent.tex.
"""
import os
import sys
import time
import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import comb as _comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp
import needham as N
import ch08_diamond as D                                  # the diamond distance of an over-rotation

DELTA = 0.1
NS_X = [2, 4, 6, 8, 10]                                   # certified
NS_XY = [4, 8]
NS_ALL = {"X": [2, 4, 6, 8, 10, 12], "XY": [4, 8, 12]}   # checked by enumeration
SAMPLES = 20000
OPTS = dict(solver="SCS", eps=1e-8, max_iters=200000)     # at eps = 1e-7 SCS says "optimal" 1e-4 below the maximum
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)


def comb(n, k):
    return int(_comb(n, k, exact=True))


def train(name, n):
    return "X" * n if name == "X" else "XYXY" * (n // 4)


# ---- the maps given to pysparsepmi: (c, s) = (cos eps, sin eps)
def x_turn(r, c, s):
    """Turn (x, y, z) through pi + eps about x."""
    x, y, z = r
    return [x, -c * y + s * z, -s * y - c * z]


def y_turn(r, c, s):
    """Turn (x, y, z) through pi + eps about y."""
    x, y, z = r
    return [-c * x - s * z, y, s * x - c * z]


def x_chain(n):
    """The X train on (y, z) alone; ideally the vector ends at (-1)^n times the pole."""
    sign = (-1) ** n
    step = lambda r, e: x_turn([0.0, r[0], r[1]], e[0], e[1])[1:]
    last = lambda r, e: 0.5 * (1 - sign * step(r, e)[1])
    return [lambda r, e: [e[1], -e[0]]] + [step] * (n - 2) + [last]


def xy_chain(n):
    """The XY train on (x, y, z); after whole cycles XYXY the vector should end at the pole."""
    turns = [x_turn if k % 2 == 0 else y_turn for k in range(n)]
    maps = [lambda r, e: x_turn([0.0, 0.0, 1.0], e[0], e[1])]
    maps += [lambda r, e, f=f: f(r, e[0], e[1]) for f in turns[1:-1]]
    maps.append(lambda r, e: 0.5 * (1 - turns[-1](r, e[0], e[1])[2]))
    return maps


def certify(name, n):
    maps = x_chain(n) if name == "X" else xy_chain(n)
    t = time.perf_counter()
    res = psp.composition_lower_bound(maps, box=1.0, order=2, method="chord", sense="max", xdims=2,
                                      local_ineqs=lambda e: [e[0] - np.cos(DELTA)],
                                      local_eqs=lambda e: [e[0] ** 2 + e[1] ** 2 - 1], **OPTS)
    return res, time.perf_counter() - t


# ---- independent evaluation with 2x2 unitaries
def pulse(axis, theta):
    """exp(-i theta P/2) for P = sigma_x or sigma_y; theta may be an array."""
    P = SX if axis == "X" else SY
    th = np.asarray(theta, dtype=float)[..., None, None]
    return np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * P


def infidelity(eps, pulses):
    """1 - |<ideal|actual>|^2 after the train, from |0>; eps has the shape (..., n)."""
    eps = np.asarray(eps, dtype=float)
    psi = np.zeros(eps.shape[:-1] + (2,), dtype=complex)
    psi[..., 0] = 1
    ideal = np.array([1, 0], dtype=complex)
    for k, a in enumerate(pulses):
        psi = np.einsum("...ij,...j->...i", pulse(a, np.pi + eps[..., k]), psi)
        ideal = pulse(a, np.pi) @ ideal
    return 1 - np.abs(psi @ ideal.conj()) ** 2


def angle(infid):
    return 2 * np.arcsin(np.sqrt(np.clip(infid, 0, 1)))


def worst(pulses, starts=20, seed=0):
    """The largest infidelity: over the 2^n vertices, then by local search from random starts."""
    n = len(pulses)
    V = DELTA * np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    vals = infidelity(V, pulses)
    best_vertex = vals.max()
    best = best_vertex
    rng = np.random.default_rng(seed)
    for x0 in [V[vals.argmax()]] + [rng.uniform(-DELTA, DELTA, n) for _ in range(starts)]:
        r = minimize(lambda e: -infidelity(e, pulses), x0, bounds=[(-DELTA, DELTA)] * n, method="L-BFGS-B")
        best = max(best, -r.fun)
    return best, best_vertex


def compute():
    out = {"cert": {"X": {}, "XY": {}}, "worst": {}, "sys": {}, "rand": {}}
    for n in NS_X:
        out["cert"]["X"][n] = certify("X", n)
    for n in NS_XY:
        out["cert"]["XY"][n] = certify("XY", n)
    rng = np.random.default_rng(1)
    for name in ("X", "XY"):
        for n in NS_ALL[name]:
            p = train(name, n)
            out["worst"][name, n] = worst(p)
            out["sys"][name, n] = infidelity(np.full(n, DELTA), p)
            inf = infidelity(rng.uniform(-DELTA, DELTA, (SAMPLES, n)), p)
            out["rand"][name, n] = (inf.mean(), np.sqrt(np.mean(angle(inf) ** 2)))
    return out


def chain_bound(n):
    return (n * np.sin(DELTA / 2)) ** 2


def checks(out):
    rng = np.random.default_rng(2)
    # the X train: the errors add
    for n in (3, 8, 12):
        e = rng.uniform(-DELTA, DELTA, (200, n))
        assert np.allclose(infidelity(e, "X" * n), np.sin(e.sum(1) / 2) ** 2, atol=1e-12)
    # one pulse: infidelity sin^2(delta/2), diamond distance sin(delta/2) (Chapter 8)
    assert abs(infidelity([DELTA], "X") - np.sin(DELTA / 2) ** 2) < 1e-14
    dd = D.diamond_brute(D.rotation(DELTA), D.identity)
    assert abs(dd - np.sin(DELTA / 2)) < 1e-6 and abs(np.sin(DELTA / 2) - 0.050) < 1e-3
    for n in NS_ALL["X"]:
        wx, vx = out["worst"]["X", n]
        assert abs(wx - np.sin(n * DELTA / 2) ** 2) < 1e-10 and abs(vx - wx) < 1e-10
        assert abs(out["sys"]["X", n] - wx) < 1e-12                       # X: the worst case is systematic
        assert wx <= chain_bound(n)
    for n in NS_ALL["XY"]:
        wx, wy, vy = out["worst"]["X", n][0], *out["worst"]["XY", n]
        assert abs(vy - wy) < 1e-9                                        # XY: attained at a vertex
        assert out["sys"]["XY", n] < 1.2e-6                               # XY cancels equal errors ...
        assert 0.5 < wy / wx < 0.54                                       # ... but only halves the worst case
        assert abs(angle(wy) / (n * DELTA / np.sqrt(2)) - 1) < 0.001
    for name in ("X", "XY"):                                              # random errors: ~ n delta^2/12
        for n in NS_ALL[name]:
            m, rms = out["rand"][name, n]
            assert abs(m / ((1 - (np.sin(DELTA) / DELTA) ** n) / 2) - 1) < 0.03
            assert abs(rms / (np.sqrt(n / 3) * DELTA) - 1) < 0.03
        assert 8 < out["worst"][name, 12][0] / out["worst"][name, 4][0] < 9
        assert 2.8 < out["rand"][name, 12][0] / out["rand"][name, 4][0] < 3.1
    assert all(abs(out["sys"]["XY", n] - v) < 0.05e-7 * n for n, v in [(4, 1.24e-7), (8, 4.97e-7), (12, 1.12e-6)])
    assert abs(out["worst"]["X", 10][0] / chain_bound(10) - 1) < 0.09
    assert all(0.45 < out["worst"]["XY", n][0] / chain_bound(n) < 0.5 for n in NS_ALL["XY"])
    for name, n in (("X", 6), ("XY", 8)):                                 # composed, the infidelity has degree n
        e = psp.polyvar(2 * n)
        r = []
        for k, f in enumerate(x_chain(n) if name == "X" else xy_chain(n)):
            r = f(r, e[2 * k:2 * k + 2])
        assert r.degree() == n
    assert all(abs(out["worst"]["XY", n][0] - v) < 1e-4 for n, v in [(4, 0.0198), (8, 0.0778), (12, 0.1692)])
    # the certificates: exact to 1e-6, statuses optimal, block sizes constant
    for name, ns in (("X", NS_X), ("XY", NS_XY)):
        for n in ns:
            res, _ = out["cert"][name][n]
            assert res.status == "optimal", (name, n, res.status)
            assert abs(res.value - out["worst"][name, n][0]) < 1e-6, (name, n, res.value)
        assert {max(out["cert"][name][n][0].block_sizes) for n in ns if n >= 4} == {28 if name == "X" else 36}
    assert comb(25, 5) == 53130


def marks(pl, xs, ys, style, shape="dot"):
    out = []
    for x, y in zip(xs, ys):
        q = pl.P(x, y)
        if shape == "dot":
            out.append("\\fill[%s] (%.3f,%.3f) circle (1.6pt);" % (style, q[0], q[1]))
        elif shape == "square":
            out.append("\\draw[%s, line width=0.7pt, fill=white] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                       % (style, q[0] - 0.05, q[1] - 0.05, q[0] + 0.05, q[1] + 0.05))
        else:
            out.append("\\draw[%s, line width=0.8pt] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
                       % (style, q[0] - 0.06, q[1] - 0.06, q[0] + 0.06, q[1] + 0.06, q[0] - 0.06, q[1] + 0.06, q[0] + 0.06, q[1] - 0.06))
    return out


def log_ticks(pl, values, labels, x=True):
    out = []
    for v, lab in zip(values, labels):
        if x:
            p = pl.P(np.log10(v), pl.ylim[0])
            out.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[below, font=\\scriptsize, text=ndInk] {%s};"
                       % (p[0], p[1] + 0.06, p[0], p[1] - 0.06, lab))
        else:
            p = pl.P(pl.xlim[0], np.log10(v))
            out.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[left, font=\\scriptsize, text=ndInk] {%s};"
                       % (p[0] + 0.06, p[1], p[0] - 0.06, p[1], lab))
    return out


def trains(x0, y0):
    """(a), left: the two trains of pulses."""
    b = []
    w, gap = 0.46, 0.12
    for row, (name, lab) in enumerate([("X" * 8, "X train"), ("XY" * 4, "XY train")]):
        y = y0 - 0.95 * row
        b.append("\\node[font=\\small, anchor=east] at (%.2f,%.2f) {%s};" % (x0 - 0.1, y + 0.25, lab))
        b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f);" % (x0 - 0.05, y, x0 + 8 * (w + gap) + 0.45, y))
        for k, a in enumerate(name):
            xa = x0 + k * (w + gap)
            col = "ndSky" if a == "X" else "ndGold!45"
            b.append("\\draw[ink, fill=%s] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (col, xa, y, xa + w, y + 0.5))
            b.append("\\node[font=\\scriptsize] at (%.2f,%.2f) {$%s$};" % (xa + w / 2, y + 0.25, a))
        b.append("\\node[font=\\small] at (%.2f,%.2f) {$\\cdots$};" % (x0 + 8 * (w + gap) + 0.2, y + 0.25))
    for k in (0, 7):
        xa = x0 + k * (w + gap) + w / 2
        b.append("\\node[font=\\tiny] at (%.2f,%.2f) {$\\pi{+}\\varepsilon_{%d}$};" % (xa, y0 + 0.68, k + 1))
    b.append("\\node[font=\\tiny] at (%.2f,%.2f) {$\\cdots$};" % (x0 + 3.5 * (w + gap) + w / 2, y0 + 0.68))
    b.append("\\node[note, anchor=north west] at (%.2f,%.2f) {every pulse turns the qubit through $\\pi+\\varepsilon_k$,\\\\ "
             "about $x$ or $y$, with $|\\varepsilon_k|\\le\\delta=0.1$ unknown};" % (x0 - 1.55, y0 - 1.15))
    return b


def bloch_circle(cx, cy, R, thetas, colour, title, spread=None):
    """(a), right: the (y, z) circle with the Bloch vector of the X train after every second pulse."""
    b = ["\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (cx - R - 0.2, cy, cx + R + 0.2, cy),
         "\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (cx, cy - R - 0.2, cx, cy + R + 0.25),
         "\\draw[ink] (%.3f,%.3f) circle (%.3f);" % (cx, cy, R),
         "\\node[font=\\scriptsize, anchor=south west] at (%.3f,%.3f) {$|0\\rangle$};" % (cx + 0.02, cy + R + 0.02),
         "\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$|1\\rangle$};" % (cx, cy - R - 0.2)]
    if spread is not None:
        ts = np.linspace(-spread, spread, 40)
        pts = [(cx, cy)] + [(cx - R * np.sin(t), cy + R * np.cos(t)) for t in ts]
        b.append("\\fill[%s!25] %s -- cycle;" % (colour, " -- ".join(N.pt(p) for p in pts)))
    for j, th in enumerate(thetas):
        p = (cx - R * np.sin(th), cy + R * np.cos(th))
        b.append("\\fill[%s] %s circle (%.1fpt);" % (colour, N.pt(p), 1.5 if j < len(thetas) - 1 else 2.0))
    last = thetas[-1]
    b.append("\\draw[vec, %s] (%.3f,%.3f) -- %s;" % (colour, cx, cy, N.pt((cx - 0.93 * R * np.sin(last), cy + 0.93 * R * np.cos(last)))))
    b.append("\\node[note, anchor=north, text=%s] at (%.3f,%.3f) {%s};" % (colour, cx, cy - R - 0.5, title))
    return b


def key(x, y, items, tag):
    """A row naming lines and marks of (b) and (c); each entry starts where the last label ends."""
    b = ["\\coordinate (%s0) at (%.3f,%.3f);" % (tag, x, y)]
    for j, (kind, col, lab) in enumerate(items):
        at = "($(%s%d.east)+(%.2f,0)$)" % (tag, j, 0.0 if j == 0 else 0.3)
        if kind in ("line", "dash"):
            dash = ", dash pattern=on 3pt off 2pt" if kind == "dash" else ""
            b.append("\\draw[line width=0.9pt, %s%s] %s -- ++(0.4,0) coordinate (%sm%d);" % (col, dash, at, tag, j))
        else:
            b.append("\\path %s ++(0.2,0) coordinate (%sm%d);" % (at, tag, j))
            if kind == "dot":
                b.append("\\fill[%s] ($(%sm%d)+(-0.08,0)$) circle (1.6pt);" % (col, tag, j))
            elif kind == "square":
                b.append("\\draw[%s, line width=0.7pt, fill=white] ($(%sm%d)+(-0.13,-0.05)$) rectangle ++(0.1,0.1);" % (col, tag, j))
            else:
                b.append("\\draw[%s, line width=0.8pt] ($(%sm%d)+(-0.14,-0.06)$) -- ++(0.12,0.12) ($(%sm%d)+(-0.14,0.06)$) -- ++(0.12,-0.12);"
                         % (col, tag, j, tag, j))
        b.append("\\node[font=\\scriptsize, anchor=west, inner sep=1.5pt] (%s%d) at (%sm%d) {%s};" % (tag, j + 1, tag, j, lab))
    return b


def main():
    out = compute()
    checks(out)
    b = trains(1.7, 8.2)

    # ---- (a) the X train after every second pulse, from |0>: equal errors and one draw of random ones
    th_sys = [2 * k * DELTA for k in range(1, 9)]
    eps = np.random.default_rng(5).uniform(-DELTA, DELTA, 16)
    th_rand = list(np.cumsum(eps)[1::2])
    cx1, cx2 = 9.6, 14.3
    b += bloch_circle(cx1, 7.45, 1.1, th_sys, "ndBlue", "equal errors add up:\\\\ $1.6$ rad after $16$ pulses")
    arc = [(cx1 - 0.42 * np.sin(t), 7.45 + 0.42 * np.cos(t)) for t in np.linspace(0, 1.6, 30)]
    b.append(N.polyline(arc, "ndBlue, line width=0.5pt, ->, >=stealth"))
    b.append("\\node[font=\\scriptsize, text=ndBlue] at (%.2f,%.2f) {$16\\delta$};" % (cx1 - 0.68 * np.sin(0.8), 7.45 + 0.68 * np.cos(0.8)))
    b += bloch_circle(cx2, 7.45, 1.1, th_rand, "ndGreen", "random errors partly cancel:\\\\ $0.23$ rad (rms) after $16$ pulses",
                      spread=np.sqrt(16 / 3) * DELTA)
    b += key(2.4, 4.27, [("line", "ndBlue", "X, worst case"), ("line", "ndRed", "XY, worst case"),
                         ("line", "ndGreen", "random errors"), ("dash", "ndInk", "the bound of Chapter 8")], "ka")
    b += key(2.4, 3.95, [("dot", "ndInk", "certified by state lifting"), ("dot", "ndGreen", "X, random"),
                         ("square", "ndGreen", "XY, random"), ("cross", "ndRed", "XY, equal errors")], "kb")
    b.append("\\node[font=\\small, anchor=north] at (8.4,4.85) {(a) two trains of $\\pi$ pulses, and the Bloch vector of the X train after every second pulse};")

    nn = np.arange(1, 12.6, 0.1)
    # ---- (b) the angle between the actual and the ideal Bloch vectors, in units of delta
    pb = N.Plot((0, 12.5), (-2.2, 13), 4.3, 3.2, origin=(0.6, 0.0))
    b += pb.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize angle $\\phi/\\delta$}",
                 xticks=[(2, "$2$"), (6, "$6$"), (10, "$10$")], yticks=[(0, "$0$"), (4, "$4$"), (8, "$8$"), (12, "$12$")])
    b.append(pb.curve(nn, nn, "line width=0.9pt, ndBlue"))
    nx, ny = NS_ALL["X"], NS_ALL["XY"]
    b.append(pb.curve(ny, [angle(out["worst"]["XY", n][0]) / DELTA for n in ny], "line width=0.9pt, ndRed"))
    b.append(pb.curve(nn, np.sqrt(nn / 3), "line width=0.9pt, ndGreen"))
    b += marks(pb, nx, [out["rand"]["X", n][1] / DELTA for n in nx], "ndGreen", "dot")
    b += marks(pb, ny, [out["rand"]["XY", n][1] / DELTA for n in ny], "ndGreen", "square")
    b += marks(pb, ny, [angle(out["sys"]["XY", n]) / DELTA for n in ny], "ndRed", "cross")
    b += marks(pb, NS_X, [angle(out["cert"]["X"][n][0].value) / DELTA for n in NS_X], "ndBlue")
    b += marks(pb, NS_XY, [angle(out["cert"]["XY"][n][0].value) / DELTA for n in NS_XY], "ndRed")
    b.append("\\node[note, anchor=east, text=ndBlue] at %s {$n\\delta$};" % pb.pt(9.3, 10.6))
    b.append("\\node[note, anchor=west, text=ndRed] at %s {$\\approx n\\delta/\\sqrt2$};" % pb.pt(9.4, 5.3))
    b.append("\\node[note, anchor=south, text=ndGreen] at %s {$\\approx\\sqrt{n/3}\\,\\delta$ (rms)};" % pb.pt(9.6, 2.25))
    b.append("\\node[font=\\small] at (%.2f,-0.95) {(b) the error as an angle};" % pb.P(6.25, 0)[0])

    # ---- (c) the infidelity, on logarithmic scales
    pc = N.Plot((np.log10(1.6), np.log10(14)), (-3, 0), 4.3, 3.2, origin=(6.3, 0.0))
    b += pc.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize infidelity}")
    b += log_ticks(pc, [2, 4, 8], ["$2$", "$4$", "$8$"], x=True)
    b += log_ticks(pc, [1e-3, 1e-2, 1e-1, 1], ["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"], x=False)
    ln = np.log10(nn[nn >= 1.6])
    b.append(pc.curve(ln, np.log10(chain_bound(10 ** ln)), "line width=0.7pt, ndInk, dash pattern=on 3pt off 2pt"))
    b.append(pc.curve(ln, np.log10(np.sin(10 ** ln * DELTA / 2) ** 2), "line width=0.9pt, ndBlue"))
    b.append(pc.curve(np.log10(ny), [np.log10(out["worst"]["XY", n][0]) for n in ny], "line width=0.9pt, ndRed"))
    b.append(pc.curve(ln, np.log10((1 - (np.sin(DELTA) / DELTA) ** (10 ** ln)) / 2), "line width=0.9pt, ndGreen"))
    b += marks(pc, np.log10(nx), [np.log10(out["rand"]["X", n][0]) for n in nx], "ndGreen", "dot")
    b += marks(pc, np.log10(ny), [np.log10(out["rand"]["XY", n][0]) for n in ny], "ndGreen", "square")
    b += marks(pc, np.log10(NS_X), [np.log10(out["cert"]["X"][n][0].value) for n in NS_X], "ndBlue")
    b += marks(pc, np.log10(NS_XY), [np.log10(out["cert"]["XY"][n][0].value) for n in NS_XY], "ndRed")
    b.append("\\node[note, anchor=north west] (cb) at %s {Chapter 8: $n^2\\sin^2\\frac{\\delta}{2}$};" % pc.pt(np.log10(1.75), -0.1))
    b.append("\\draw[pointer] (cb.south east) to[bend left=10] %s;" % pc.pt(np.log10(6.6), np.log10(chain_bound(6.6)) + 0.04))
    b.append("\\node[note, anchor=south east, text=ndBlue] at %s {slope $2$};" % pc.pt(np.log10(3.4), -1.45))
    b.append("\\node[note, anchor=west, text=ndGreen] at %s {slope $1$};" % pc.pt(np.log10(6.0), -2.75))
    b.append("\\node[font=\\small] at (%.2f,-0.95) {(c) the error as an infidelity};" % pc.P(np.log10(1.6) + 0.5 * np.log10(14 / 1.6), 0)[0])

    # ---- (d) the solve times (log scale)
    pd = N.Plot((0, 11), (-1, 2), 4.3, 3.2, origin=(12.0, 0.0))
    b += pd.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize seconds (SCS)}",
                 xticks=[(2, "$2$"), (6, "$6$"), (10, "$10$")])
    b += log_ticks(pd, [0.1, 1, 10, 100], ["$0.1$", "$1$", "$10$", "$100$"], x=False)
    for name, ns, col in (("X", NS_X, "ndBlue"), ("XY", NS_XY, "ndRed")):
        ts = [np.log10(out["cert"][name][n][1]) for n in ns]
        b.append(pd.curve(ns, ts, "line width=0.6pt, %s" % col))
        b += marks(pd, ns, ts, col)
    b.append("\\node[note, anchor=north west] at %s {largest PSD block:\\\\ $28$ (X), $36$ (XY)\\\\ for every $n\\ge4$};" % pd.pt(4.2, 0.0))
    b.append("\\node[font=\\tiny, text=ndBlue, anchor=north] at %s {X};" % pd.pt(10, np.log10(out["cert"]["X"][10][1]) - 0.12))
    b.append("\\node[font=\\tiny, text=ndRed, anchor=south] at %s {XY};" % pd.pt(8, np.log10(out["cert"]["XY"][8][1]) + 0.1))
    b.append("\\node[font=\\small] at (%.2f,-0.95) {(d) the cost of a certificate};" % pd.P(5.5, 0)[0])
    N.write("ch11_coherent", b)
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    out = main()
    for name in ("X", "XY"):
        for n, (res, t) in out["cert"][name].items():
            print("%-2s n=%2d  certified %.7f  exact %.7f  status %s  block %d  clique %d  %.1f s"
                  % (name, n, res.value, out["worst"][name, n][0], res.status, max(res.block_sizes),
                     max(len(c) for c in res.cliques), t))
        for n in NS_ALL[name]:
            m, rms = out["rand"][name, n]
            print("%-2s n=%2d  worst %.5f  systematic %.2e  random mean %.2e  rms angle/delta %.3f  chain %.5f"
                  % (name, n, out["worst"][name, n][0], out["sys"][name, n], m, rms / DELTA, chain_bound(n)))
    print("total %.0f s; written figures/ch11_coherent.tex" % (time.perf_counter() - t0))
