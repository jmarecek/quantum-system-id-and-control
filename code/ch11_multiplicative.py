"""ch11_multiplicative.py -- multiplicative noise in the control, seen by a Markovian fit.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi), with cvxpy and
the SCS solver, through ch11_markovfit.py.  If the package is not installed,
set PYSPARSEPMI_PATH to a clone of the repository.

The driven qubit of Chapter 4, H = -(omega/2) sigma_z + ((1 + xi) u/2) sigma_x
with omega = 1 and u = 1.5, whose drive amplitude carries a relative error xi,
Gaussian with the spread sigma: static (the same in every shot of a run), or
drifting as a Kubo-Anderson process (xi redrawn from the same Gaussian at the
rate 1/tau, so that <xi(0) xi(t)> = sigma^2 exp(-t/tau)).  Without the error
the Bloch vector turns at Omega = (omega^2 + u^2)^(1/2) = 1.803 about a tilted
axis, the Rabi period being 3.485; the error changes the frequency by
Omega' xi, Omega' = u^2/Omega = 1.248, and the average over the shots is
computed exactly (60-point Gauss-Hermite quadrature in xi; for the drift, the
joint linear equation of the qubit and the quadrature node).

Data: the six cardinal states, the coherence vector at 16 times per Rabi
period.  The best Markovian model is found by ch11_markovfit.fit: the GKSL
generator, gamma >= 0 imposed as a polynomial matrix inequality, that best
satisfies the integral form of xdot = A x + beta over sliding windows of half
a Rabi period (a full period would hide the damping, for over a period the
oscillating part integrates to nothing); the certified mean-square misfit is
checked against a direct cvxpy solve and against the misfit of the minimiser
made exactly feasible.  The constraint binds: without it the fit is not
completely positive.

Checked here: a static error damps the oscillation with a Gaussian envelope,
exp(-a t^2/2), a = sigma^2 Omega'^2, not an exponential one: the least-squares
fit of ln|envelope| on [0, 10 periods] for sigma = 0.05 has the t^2
coefficient -0.001944 against -a/2 = -0.001947 and a linear term 8.6e-5.  The
drifting error, by contrast, gives a semigroup at long times, damping at the
rate a tau (the slowest mode of the joint equation, to 1%).

(a) The population signal z(t) from |0>, sigma = 0.05 (static), over ten Rabi
    periods, with its Gaussian envelope and the exponential envelope
    exp(-0.037 t) of the best Markovian model (certified misfit 9.2e-5), too
    steep at first and too shallow later.
(b) The certified misfit against sigma for records of 2, 5 and 10 Rabi periods
    (pulse areas u T = 10.5, 26.1 and 52.3): it grows as sigma^4 (fitted
    slopes 4.00, 3.99, 3.94).  Dashed, the level at which 10^4 shots per
    setting and time would detect it (half of the noisy records then fail the
    test that 95% of Markovian records pass; the null distribution from 30
    simulated records of the error-free qubit).  The thresholds,
    sigma* = 0.063, 0.037 and 0.024, are marked and checked by simulation: at
    sigma*/2 one of 16 noisy records is flagged, at 2 sigma* all of them.
(c) The damping rate of the best Markovian model against the length of the
    record, for sigma = 0.03.  For the static error it grows with the record,
    as a T/2 to within 5% for records of up to two periods (the time-local
    rate a t at the middle of the record), and levels off only once the
    oscillation has died.  For the drifting error it approaches the long-time
    rate of the average (dashed), which is a tau to 1% for tau = 3; for
    tau = 10, where sigma Omega' tau = 0.37 is not small, the narrowing is
    incomplete and that rate is 0.0127 against a tau = 0.0140.  At 32 periods
    the fitted rates have reached 98% and 85% of these.  A Markovian process
    would give a horizontal line.
Writes figures/ch11_multiplicative.tex.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm

import needham as N
import ch11_markovfit as MF

OMEGA, U = 1.0, 1.5                       # the driven qubit of Chapter 4
OM = np.hypot(OMEGA, U)
PERIOD = 2 * np.pi / OM
DOM = U ** 2 / OM                         # dOmega/dxi at xi = 0
STEPS, WIN = 16, 4                        # samples per Rabi period; window of 4 Simpson pairs = half a period
SHOTS = 10 ** 4
RECORDS = (2, 5, 10)                      # record lengths, in Rabi periods
SIG_C = 0.03                              # the spread in panel (c)
TAUS = (np.inf, 10.0, 3.0)


def cross(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def nodes(sig, n=60):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return sig * x, w / w.sum()


def record(sig, periods, tau=np.inf, n=60):
    """Times and the averaged records y = (1, x) of the six cardinal states."""
    K = int(round(STEPS * periods / 2))
    ts = np.linspace(0, 2 * K * PERIOD / STEPS, 2 * K + 1)
    xi, w = nodes(sig, n) if sig > 0 else (np.zeros(1), np.ones(1))
    m = len(xi)
    big = np.zeros((3 * m, 3 * m))
    for k in range(m):
        big[3 * k:3 * k + 3, 3 * k:3 * k + 3] = cross([U * (1 + xi[k]), 0, -OMEGA])
    if np.isfinite(tau) and m > 1:
        big += np.kron(np.outer(w, np.ones(m)) - np.eye(m), np.eye(3)) / tau
    E = expm(big * ts[1])
    Ys = []
    for s in MF.CARDINAL:
        y, out = np.kron(w, s), []
        for _ in ts:
            out.append(y)
            y = E @ y
        X = np.array(out).reshape(len(ts), m, 3).sum(1)
        Ys.append(np.hstack([np.ones((len(ts), 1)), X]))
    return ts, Ys


def markov(ts, Ys, check=False):
    return MF.fit(*MF.design(Ys, ts[1], window=WIN), check=check)


def rate(out):
    lam = np.linalg.eigvals(out["G"][1:, 1:])
    return -lam[np.argmax(np.abs(lam.imag))].real


def envelope(sig, ts):
    """|average of (1 - n_z^2) exp(i Omega(xi) t)| / average of (1 - n_z^2): the envelope of z(t) from |0>."""
    xi, w = nodes(sig, 80)
    om = np.hypot(OMEGA, U * (1 + xi))
    wt = w * (1 - (OMEGA / om) ** 2)
    return np.abs(np.exp(1j * np.outer(ts, om)) @ wt) / wt.sum()


def checks():
    # the static error: a Gaussian, not an exponential, envelope
    sig = 0.05
    ts = np.linspace(0, 10 * PERIOD, 400)
    c2, c1, c0 = np.polyfit(ts, np.log(envelope(sig, ts)), 2)
    a = sig ** 2 * DOM ** 2
    assert abs(c2 + a / 2) < 0.01 * a / 2 and abs(c1) * ts[-1] < 0.002 * abs(c2) * ts[-1] ** 2, (c2, c1)
    # the averaged z(t) is the record's z for |0>
    _, Ys = record(sig, 10)
    nz2 = np.sum(nodes(sig, 80)[1] * (OMEGA / np.hypot(OMEGA, U * (1 + nodes(sig, 80)[0]))) ** 2)
    assert abs(Ys[4][0, 3] - 1) < 1e-12 and abs(np.mean(Ys[4][-STEPS:, 3]) - nz2) < 0.01
    # quadrature: 60 nodes agree with 100
    for tau in (np.inf, 3.0):
        _, Y60 = record(SIG_C, 10, tau)
        _, Y100 = record(SIG_C, 10, tau, n=100)
        assert max(np.abs(a_ - b_).max() for a_, b_ in zip(Y60, Y100)) < 1e-10
    # the drifting error: its slowest oscillating mode decays at a tau when sigma Omega' tau << 1
    slow = {}
    for tau in (10.0, 3.0):
        xi, w = nodes(SIG_C)
        m = len(xi)
        big = np.zeros((3 * m, 3 * m))
        for k in range(m):
            big[3 * k:3 * k + 3, 3 * k:3 * k + 3] = cross([U * (1 + xi[k]), 0, -OMEGA])
        big += np.kron(np.outer(w, np.ones(m)) - np.eye(m), np.eye(3)) / tau
        ev = np.linalg.eigvals(big)
        osc = ev[np.abs(np.abs(ev.imag) - OM) < 0.2]
        slow[tau] = -osc.real.max()
    assert abs(slow[3.0] / (SIG_C ** 2 * DOM ** 2 * 3.0) - 1) < 0.01
    # with no error the record is a semigroup, fitted exactly
    ts, Ys = record(0.0, 2)
    out = markov(ts, Ys, check=2)
    assert out["value"] < 1e-12 and abs(rate(out)) < 1e-6
    return slow


def main():
    slow = checks()
    rng = np.random.default_rng(11)
    res = {}
    # ---- (b) misfit against sigma, the noise floor, the thresholds
    sigs = np.geomspace(0.003, 0.1, 12)
    for P in RECORDS:
        ts, _ = record(0.0, P)
        mis, exc = [], []
        for s in sigs:
            out = markov(*record(s, P), check=2 if abs(s - 0.03) < 0.003 else True)
            mis.append(out["value"])
            exc.append(out["excess"] / out["value"])
        mis = np.array(mis)
        slope = np.polyfit(np.log(sigs[:5]), np.log(mis[:5]), 1)[0]
        assert abs(slope - 4) < 0.1, slope
        _, Y0 = record(0.0, P)
        null = np.array([markov(ts, MF.shots(Y0, SHOTS, rng))["value"] for _ in range(30)])
        q95, med = np.quantile(null, 0.95), np.median(null)
        star = float(np.exp(np.interp(np.log(q95 - med), np.log(mis), np.log(sigs))))
        flagged = []
        for fac in (0.5, 2.0):
            _, Ys = record(fac * star, P)
            vals = [markov(ts, MF.shots(Ys, SHOTS, rng))["value"] for _ in range(16)]
            flagged.append(np.mean(np.array(vals) > q95))
        assert flagged[0] <= 0.25 and flagged[1] >= 0.85, (P, flagged)
        res[P] = dict(mis=mis, exc=np.array(exc), q95=q95, med=med, star=star, flagged=flagged, slope=slope)
    stars = [res[P]["star"] for P in RECORDS]
    assert stars[0] > stars[1] > stars[2]                 # longer records see smaller errors
    # ---- (c) the fitted rate against the length of the record
    pers = np.array([1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    a = SIG_C ** 2 * DOM ** 2
    rates = {tau: np.array([rate(markov(*record(SIG_C, P, tau))) for P in pers]) for tau in TAUS}
    short = pers <= 2
    assert np.all(np.abs(rates[np.inf][short] / (a * pers[short] * PERIOD / 2) - 1) < 0.05)
    assert np.all(np.diff(rates[np.inf]) > -1e-6)
    assert abs(rates[3.0][-1] / (a * 3.0) - 1) < 0.05 and rates[10.0][-1] / (a * 10.0) > 0.7
    assert rates[3.0][-1] / slow[3.0] > 0.97 and 0.8 < rates[10.0][-1] / slow[10.0] < 0.9
    assert rates[np.inf][-1] > 4 * a * 3.0
    # ---- (a) the signal and the best Markovian model
    sa = 0.05
    ts, Ys = record(sa, 10)
    fa = markov(ts, Ys, check=2)
    ga = rate(fa)
    ze = Ys[4][:, 3]
    xi, w = nodes(sa, 80)
    nz2 = np.sum(w * (OMEGA / np.hypot(OMEGA, U * (1 + xi))) ** 2)
    amp = 1 - nz2

    b = []
    lg = np.log10
    ycap = 0.05
    # (a)
    T = ts[-1]
    pa = N.Plot((0, T), (-0.5, 1.05), 4.1, 3.6, origin=(0.75, 0.85))
    b += pa.axes(xlabel="{\\scriptsize $t$}", ylabel="{\\scriptsize $\\langle\\sigma_z\\rangle$ from $\\ket0$}",
                 xticks=[(10, "$10$"), (20, "$20$"), (30, "$30$")], yticks=[(0, "$0$"), (1, "$1$")], yzero=None)
    b.append("\\draw[faint] %s -- %s;" % (pa.pt(0, 0), pa.pt(T, 0)))
    tt = np.linspace(0, T, 300)
    for sgn in (1, -1):
        b.append(pa.curve(tt, nz2 + sgn * amp * envelope(sa, tt), "line width=0.9pt, ndBlue!70"))
        b.append(pa.curve(tt, nz2 + sgn * amp * np.exp(-ga * tt), "line width=0.9pt, ndRed, dash pattern=on 2.5pt off 1.5pt"))
    b.append(pa.curve(ts, ze, "line width=0.5pt, ndInk!80"))
    b.append("\\node[note, anchor=south, text=ndBlue] (ga) at %s {Gaussian envelope};" % pa.pt(0.66 * T, 0.93))
    b.append("\\draw[pointer] (ga.south) to[bend left=15] %s;" % pa.pt(0.55 * T, nz2 + amp * envelope(sa, np.array([0.55 * T]))[0] + 0.02))
    b.append("\\node[note, anchor=north, text=ndRed] (ea) at %s {best Markovian\\\\ model: $e^{-%.3ft}$};" % (pa.pt(0.74 * T, -0.15), ga))
    b.append("\\draw[pointer] (ea.north west) to[bend left=20] %s;" % pa.pt(0.55 * T, nz2 - amp * np.exp(-ga * 0.55 * T) - 0.01))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) $\\sigma=%.2f$, ten Rabi periods};" % (N.pt(pa.P(T / 2, -0.5) + np.array([0, -0.55])), sa))

    # (b)
    xl, yl = (lg(0.003), lg(0.1)), (-10.2, -2.8)
    pb = N.Plot(xl, yl, 4.1, 3.6, origin=(6.3, 0.85))
    b += pb.axes(xlabel="{\\scriptsize $\\sigma$}", ylabel="{\\scriptsize certified misfit}",
                 xticks=[(lg(0.01), "$0.01$"), (lg(0.03), "$0.03$"), (lg(0.1), "$0.1$")],
                 yticks=[(-10, "$10^{-10}$"), (-8, "$10^{-8}$"), (-6, "$10^{-6}$"), (-4, "$10^{-4}$")])
    cols = {2: "ndBlue!40", 5: "ndBlue!70", 10: "ndBlue!80!ndInk"}
    for P in RECORDS:
        r = res[P]
        b.append(pb.curve(lg(sigs), lg(r["mis"]), "line width=1.1pt, %s" % cols[P]))
        for s, m in zip(sigs, r["mis"]):
            b.append("\\fill[%s] %s circle (1.0pt);" % (cols[P], pb.pt(lg(s), lg(m))))
        lev = lg(r["q95"] - r["med"])
        b.append("\\draw[%s, line width=0.6pt, dash pattern=on 2pt off 1.5pt] %s -- %s;" % (cols[P], pb.pt(xl[0], lev), pb.pt(xl[1], lev)))
        b.append("\\draw[%s, line width=0.5pt] %s circle (2.0pt);" % (cols[P], pb.pt(lg(r["star"]), lev)))
    b.append("\\node[note, anchor=south east, align=right] at %s {records of {\\color{ndBlue!40}$2$}, {\\color{ndBlue!70}$5$},\\\\ {\\color{ndBlue!80!ndInk}$10$} Rabi periods};" % pb.pt(xl[1], -10.1))
    b.append("\\node[note, anchor=north west] at %s {detected by $10^4$ shots\\\\ above the dashes};" % pb.pt(xl[0] + 0.03, -2.85))
    b.append("\\node[note, anchor=north west] at %s {$\\propto\\sigma^4$};" % pb.pt(lg(0.0072), -8.6))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) certified misfit against $\\sigma$};" % N.pt(pb.P(np.mean(xl), yl[0]) + np.array([0, -0.55])))

    # (c)
    xc, yc = (0.0, lg(32)), (-3.1, -1.4)
    pc = N.Plot(xc, yc, 4.1, 3.6, origin=(11.7, 0.85))
    b += pc.axes(xlabel="{\\scriptsize periods}", ylabel="{\\scriptsize fitted rate}",
                 xticks=[(0, "$1$"), (lg(4), "$4$"), (lg(16), "$16$")],
                 yticks=[(-3, "$10^{-3}$"), (-2, "$10^{-2}$")])
    styles = {np.inf: "line width=1.1pt, ndRed", 10.0: "line width=1.1pt, ndGold!85!ndInk", 3.0: "line width=1.1pt, ndGreen"}
    for tau in TAUS:
        b.append(pc.curve(lg(pers), lg(rates[tau]), styles[tau]))
        for p_, r_ in zip(pers, rates[tau]):
            b.append("\\fill[%s] %s circle (1.0pt);" % (styles[tau].split(", ")[1], pc.pt(lg(p_), lg(r_))))
        if np.isfinite(tau):
            b.append("\\draw[dashedink] %s -- %s;" % (pc.pt(xc[0], lg(slow[tau])), pc.pt(xc[1], lg(slow[tau]))))
            b.append("\\node[font=\\scriptsize, anchor=south east] at %s {$\\tau=%d$};" % (pc.pt(xc[1], lg(slow[tau])), tau))
    b.append(pc.curve(lg(pers[pers <= 6]), lg(a * pers[pers <= 6] * PERIOD / 2), "line width=0.5pt, ndInk!60, dash pattern=on 1pt off 1.2pt"))
    b.append("\\node[note, anchor=north west, text=ndRed] at %s {static error:\\\\ $\\approx aT/2$ at first};" % pc.pt(0.03, -1.42))
    b.append("\\node[note, anchor=north, align=center] at %s {drifting, with\\\\ correlation time $\\tau$};" % pc.pt(lg(6), -2.72))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) the fitted rate, $\\sigma=%.2f$};" % (N.pt(pc.P(np.mean(xc), yc[0]) + np.array([0, -0.55])), SIG_C))
    N.write("ch11_multiplicative", b)
    return res, rates, pers, fa, ga, slow


if __name__ == "__main__":
    res, rates, pers, fa, ga, slow = main()
    print("slowest mode of the drifting average: %s (a tau = %s)" % (slow, {t: SIG_C ** 2 * DOM ** 2 * t for t in slow}))
    a = SIG_C ** 2 * DOM ** 2
    print("Omega = %.4f, Rabi period %.4f, Omega' = %.4f" % (OM, PERIOD, DOM))
    for P in RECORDS:
        r = res[P]
        print("record %2d periods (area %.1f): slope %.2f, null median %.2e, q95 %.2e, sigma* = %.4f, flagged at sigma*/2, 2 sigma*: %s; excess share %.0f-%.0f%%"
              % (P, U * P * PERIOD, r["slope"], r["med"], r["q95"], r["star"], r["flagged"], 100 * r["exc"].min(), 100 * r["exc"].max()))
    for tau in TAUS:
        print("tau = %s: rate / (a T/2) at 1 period %.3f; rates %s; a tau = %s" % (tau, rates[tau][0] / (a * PERIOD / 2),
              " ".join("%.4f" % r for r in rates[tau]), "-" if not np.isfinite(tau) else "%.4f" % (a * tau)))
    print("at 32 periods, fitted / long-time rate: tau = 3: %.3f, tau = 10: %.3f" % (rates[3.0][-1] / slow[3.0], rates[10.0][-1] / slow[10.0]))
    print("(a): misfit %.3e (direct %.3e, factored %.3e), excess %.3e, rate %.4f" % (fa["value"], fa["direct"], fa["factored"], fa["excess"], ga))
    print("written figures/ch11_multiplicative.tex")
