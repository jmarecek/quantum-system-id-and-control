"""ch09_traps.py -- traps disappear as the family of controls grows.

The driven qubit of Chapter 4 (omega = 1, Delta = -1), to be taken from |0>
to |1>, with a piecewise-constant control of N steps over the total time T,
amplitudes unbounded.  For each (N, T), L-BFGS climbs F = |<1|U(T)|0>|^2 from
100 starting controls with amplitudes drawn uniformly from [-3, 3]; the
gradient is computed with the Bloch costate p(T) = (0, 0, -1/2) carried
backwards (checked against finite differences).  A run succeeds if it ends
with F > 0.999.
(a) Where the runs end for T = 3.2, the time of fig:ch09:landscape, and N = 1, ..., 6.
    With one step no run succeeds: a constant pulse never reaches |1> exactly,
    since F <= 1 - omega^2/Omega^2, and the runs end at the local maxima
    0.313, 0.885 and 0.979.  With two steps 85 runs succeed, 13 end at 0.885,
    the trap of fig:ch09:landscape, the best constant pulse u1 = u2 = 2.788, and two
    stop at 0.972 and 0.995.
    With three steps 99 succeed, and from four steps on every run does.
(b) The fraction of successful runs against N for T = 2, 3.2, 6, 12 and 20.
    It is zero for N = 1, rises with N, not quite monotonically, and from
    N = 6 on every run succeeds for every T here.  For N = 2 it falls as T
    grows, from 98% at T = 2 to 32% at T = 20: the longer the time, the
    more steps are needed.
Writes figures/ch09_traps.tex.
"""
import numpy as np
from scipy.optimize import minimize
import needham as N_

DELTA = -1.0
M = 100
AMP = 3.0
THR = 0.999
TIMES = (2.0, 3.2, 6.0, 12.0, 20.0)
NS = tuple(range(1, 11))


def rotations(us, dt):
    """Rotations of the Bloch vector about (u, 0, Delta) by |(u, 0, Delta)| dt, for an array of u."""
    w = np.sqrt(us * us + DELTA * DELTA)
    n = np.stack([us / w, 0 * us, DELTA / w], -1)
    K = np.zeros(us.shape + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -n[..., 2], n[..., 1]
    K[..., 1, 0], K[..., 1, 2] = n[..., 2], -n[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -n[..., 1], n[..., 0]
    th = (w * dt)[..., None, None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def fidelity_and_gradient(us, T, h=1e-6):
    dt = T / len(us)
    R = rotations(us, dt)
    dR = (rotations(us + h, dt) - rotations(us - h, dt)) / (2 * h)
    r = [np.array([0, 0, 1.0])]
    for Rk in R:
        r.append(Rk @ r[-1])
    p = np.array([0, 0, -0.5])                                     # grad F at T, F = (1 - z)/2
    g = np.zeros(len(us))
    for k in range(len(us) - 1, -1, -1):
        g[k] = p @ (dR[k] @ r[k])
        p = R[k].T @ p
    return (1 - r[-1][2]) / 2, g


def runs(n, T, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(M):
        res = minimize(lambda x: tuple(-v for v in fidelity_and_gradient(x, T)), rng.uniform(-AMP, AMP, n), jac=True,
                       method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-10})
        out.append(-res.fun)
    return np.array(out)


def checks():
    us = np.random.default_rng(1).uniform(-2, 2, 7)
    F, g = fidelity_and_gradient(us, 3.2)
    for k in range(7):
        e = np.zeros(7)
        e[k] = 1e-6
        fd = (fidelity_and_gradient(us + e, 3.2)[0] - fidelity_and_gradient(us - e, 3.2)[0]) / 2e-6
        assert abs(fd - g[k]) < 1e-7
    # a constant pulse never reaches |1>: F <= u^2/(u^2 + Delta^2) < 1
    for u in np.linspace(-30, 30, 601):
        assert fidelity_and_gradient(np.array([u]), 3.2)[0] <= u * u / (u * u + DELTA ** 2) + 1e-12
    ends = {n: runs(n, 3.2) for n in range(1, 7)}
    frac = {T: [np.mean(runs(n, T) > THR) for n in NS] for T in TIMES}
    assert np.allclose(sorted(set(np.round(ends[1], 3))), [0.313, 0.885, 0.979]) and np.mean(ends[1] > THR) == 0
    assert np.sum(ends[2] > THR) == 85 and np.sum(np.abs(ends[2] - 0.8854) < 1e-3) == 13
    assert np.sum(ends[3] > THR) == 99 and all(np.all(ends[n] > THR) for n in (4, 5, 6))
    assert np.allclose(frac[3.2][:6], [np.mean(ends[n] > THR) for n in range(1, 7)])
    for T in TIMES:
        assert frac[T][0] == 0 and all(f == 1 for f in frac[T][5:])
    assert frac[20.0][1] < frac[12.0][1] < frac[6.0][1] < frac[3.2][1] < frac[2.0][1]
    return ends, frac


def main():
    ends, frac = checks()
    b = []
    # ---- (a) where the runs end, T = 3.2
    pa = N_.Plot((0.4, 6.6), (0.0, 1.08), 6.4, 4.2, origin=(1.0, 0.9))
    b += pa.axes(xlabel="{\\scriptsize $N$}", ylabel="{\\scriptsize final $F$}",
                 xticks=[(n, "$%d$" % n) for n in range(1, 7)], yticks=[(0.5, "$0.5$"), (1, "$1$")])
    b.append("\\draw[dashedink] %s -- %s;" % (pa.pt(0.4, 0.8854), pa.pt(6.6, 0.8854)))
    rng = np.random.default_rng(3)
    for n in range(1, 7):
        v = ends[n]
        jit = rng.uniform(-0.28, 0.28, len(v))
        for x, y in zip(n + jit, v):
            col = "ndBlue" if y > THR else "ndRed"
            b.append("\\fill[%s, fill opacity=0.6] %s circle (1.1pt);" % (col, pa.pt(x, y)))
        b.append("\\node[font=\\scriptsize, anchor=south] at %s {%d\\%%};" % (pa.pt(n, 1.02), round(100 * np.mean(v > THR))))
    b.append("\\node[note, anchor=north, text=ndRed] (tr) at %s {$0.885$: the trap of\\\\ the two-step landscape,\\\\ the best constant pulse};" % pa.pt(3.6, 0.72))
    b.append("\\draw[pointer] (tr.north west) to[bend left=15] %s;" % pa.pt(2.25, 0.875))
    b.append("\\node[note, anchor=north west] at %s {one step: never};" % pa.pt(0.45, 0.25))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) where $100$ runs end, $T=3.2$};" % pa.pt(3.5, -0.2))

    # ---- (b) the fraction of successes against N
    pb = N_.Plot((0.5, 10.5), (0.0, 1.05), 6.4, 4.2, origin=(9.5, 0.9))
    b += pb.axes(xlabel="{\\scriptsize $N$}", ylabel="{\\scriptsize runs reaching $F>0.999$}",
                 xticks=[(n, "$%d$" % n) for n in (1, 2, 4, 6, 8, 10)], yticks=[(0.5, "$50\\%$"), (1, "$100\\%$")])
    cols = {2.0: "ndGreen", 3.2: "ndBlue", 6.0: "ndGold!85!ndInk", 12.0: "ndRed", 20.0: "ndInk!70"}
    for T in TIMES:
        b.append(pb.curve(NS, frac[T], "line width=1.0pt, %s" % cols[T]))
        for n, f in zip(NS, frac[T]):
            b.append("\\fill[%s] %s circle (1.4pt);" % (cols[T], pb.pt(n, f)))
    ky = 0.62
    for k, T in enumerate(TIMES):
        y = ky - 0.11 * k
        b.append("\\draw[line width=1.0pt, %s] %s -- %s;" % (cols[T], pb.pt(6.3, y), pb.pt(7.1, y)))
        b.append("\\node[font=\\scriptsize, anchor=west, text=%s] at %s {$T=%g$};" % (cols[T], pb.pt(7.25, y), T))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) more steps, fewer traps};" % pb.pt(5.5, -0.2))
    N_.write("ch09_traps", b)
    return ends, frac


if __name__ == "__main__":
    ends, frac = main()
    for T in TIMES:
        print("T = %4.1f:" % T, " ".join("%3d" % round(100 * f) for f in frac[T]))
    print("written figures/ch09_traps.tex")
