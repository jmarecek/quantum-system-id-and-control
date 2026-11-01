"""ch11_band.py -- a certified prediction band for the identified qubit of Chapter 5.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi) and cvxpy with
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

The qubit of Chapter 5 (omega = 3, gamma1 = 0.2, gphi = 0.4, Gamma = 0.5),
prepared in the state X0 at the polar angle 2 pi/3, has
    <sigma_x>(k tau) = a Re(lambda^k),   a = sin(2 pi/3),   lambda = exp((-Gamma + i omega) tau),
with tau = 0.25, so that lambda = 0.646 + 0.602i.  A short record, k = 0, ..., 8
(t <= 2) with 200 shots at each time, is identified by Ho--Kalman
(code/ch05_spectrum.py) with two states; the pole of positive imaginary part
is lambda^ = 0.638 + 0.612i (Gamma^ = 0.493, omega^ = 3.057).  A parametric
bootstrap (400 records simulated from the identified model and identified
again) gives the covariance S of (Re lambda^, Im lambda^) (standard deviations
0.076 in Gamma and 0.139 in omega), and the stated confidence set is the
ellipse
    E = {lambda : (lambda - lambda^)' S^{-1} (lambda - lambda^) <= chi2_2(0.95) = 5.99},
which contains the true pole (Mahalanobis distance squared 0.18) and 93% of the
bootstrap poles.  The preparation, and with it a, is taken as known.

For every model in E the prediction a Re(lambda^k) at the time k tau is a
polynomial of degree k in the two coordinates of lambda; writing
lambda = lambda^ (1 + w(u)) with u in the unit disc keeps the coefficients of
order one.  The largest and the smallest prediction over E, for k = 1, ..., 20
(t <= 5), are bounded by pysparsepmi.sos_lower_bound at the order ceil(k/2)
(the lowest possible; PSD blocks of order binom(ceil(k/2) + 2, 2), 66 at
k = 20), and the bounds are exact: as Re(lambda^k) is harmonic, its extremes
over E lie on the boundary ellipse, and a search over that ellipse agrees with
every certified bound to 1e-6 of the envelope a|lambda^|^k.  All SCS statuses
are "optimal".

The certified band contains the prediction of the true model at every k.  It
is wider than the pointwise 95% band of the bootstrap predictions, by a factor
between 1.2 and 1.9: the bootstrap band describes the scatter of the estimate
one time at a time, the certified band the worst model of the set at each time.
The two answer different questions, and the narrower is not inside the wider:
at k = 4 and 8, where the prediction turns, the bootstrap band leaves the
certified one by up to 1.9% of the envelope, the predictions that leave it
(4% of them at k = 4) all coming from bootstrap poles outside E.  The envelope
of 4000 models drawn uniformly from E lies inside the certified band, short of
its edges by between 0.2% and 0.9% of its width: sampling approaches the band
from inside, the certificate bounds it from outside.

(a) The plane of the pole: the 400 bootstrap poles, the ellipse E, the
    identified and the true pole.
(b) The record (dots, with one standard deviation of the shot noise), the
    prediction of the identified model, that of the true model (dashed), the
    certified band (pale, with the bounds at t = k tau marked) and the 95%
    bootstrap band (darker bars), for t <= 5.
(c) The same multiplied by exp(Gamma^ t), so that the decay no longer hides
    the widening of the band as the uncertainty in omega becomes one of phase.
(d) The widths of the two bands, relative to the envelope a|lambda^|^k, and
    the total time of the forty certificates (about 40 s on this machine).
Writes figures/ch11_band.tex.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp
import needham as N
import ch05_spectrum as Q                      # OMEGA = 3, GAMMA1 = 0.2, GPHI = 0.4, TAU = 0.25, X0, ho_kalman

AMP = np.sin(Q.THETA0)
TAU = Q.TAU
LAM0 = np.exp((-Q.GAMMA + 1j * Q.OMEGA) * TAU)
J, SHOTS, BOOT, SEED = 9, 200, 400, 0
K = 20
CHI = chi2.ppf(0.95, 2)
OPTS = dict(solver="SCS", eps=1e-9, max_iters=200000)


def record(lam, rng):
    """<sigma_x> at k tau, k < J, estimated from SHOTS shots each."""
    x = AMP * np.real(lam ** np.arange(J))
    return 2 * rng.binomial(SHOTS, (1 + x) / 2) / SHOTS - 1


def identify(y):
    """The pole of positive imaginary part of a two-state Ho--Kalman realisation."""
    Ad, _, _ = Q.ho_kalman(y[:, None], 2)
    ev = np.linalg.eigvals(Ad)
    return ev[np.argmax(ev.imag)]


def confidence_set():
    rng = np.random.default_rng(SEED)
    y = record(LAM0, rng)
    lh = identify(y)
    boots = np.array([identify(record(lh, rng)) for _ in range(BOOT)])
    S = np.cov(np.stack([boots.real, boots.imag]))
    return y, lh, boots, S


def mahalanobis2(lam, lh, S):
    d = np.stack([np.real(lam) - lh.real, np.imag(lam) - lh.imag])
    return np.einsum("i...,ij,j...->...", d, np.linalg.inv(S), d)


def certify(lh, S):
    """Certified max and min of a Re(lambda^k) over E, k = 1..K, with the times."""
    L = np.linalg.cholesky(CHI * S)                     # lambda = lh + L u (as a complex number), |u| <= 1
    m = [(L[0, j] + 1j * L[1, j]) / lh for j in range(2)]
    u = psp.polyvar(2)
    one = psp.Polynomial.constant(2, 1.0)
    base = (one + u[0] * m[0].real + u[1] * m[1].real, u[0] * m[0].imag + u[1] * m[1].imag)   # 1 + w
    disc = 1 - u[0] ** 2 - u[1] ** 2
    P = (one, psp.Polynomial.zero(2))
    out = []
    for k in range(1, K + 1):
        P = (P[0] * base[0] - P[1] * base[1], P[0] * base[1] + P[1] * base[0])
        th = k * np.angle(lh)
        F = P[0] * np.cos(th) - P[1] * np.sin(th)           # Re(e^{ik arg lh} (1 + w)^k)
        scale = AMP * abs(lh) ** k
        t = time.perf_counter()
        lo = psp.sos_lower_bound(F, ineqs=[disc], order=(k + 1) // 2, **OPTS)
        hi = psp.sos_lower_bound(-F, ineqs=[disc], order=(k + 1) // 2, **OPTS)
        out.append(dict(k=k, lo=scale * lo.value, hi=-scale * hi.value, scale=scale, status=(lo.status, hi.status),
                        block=max(lo.block_sizes), secs=time.perf_counter() - t))
    return out


def exact(lh, S, k):
    """Max and min of a Re(lambda^k) over the boundary of E, by a search over its angle."""
    L = np.linalg.cholesky(CHI * S)
    f = lambda p: AMP * np.real((lh + (L @ np.array([np.cos(p), np.sin(p)])).T @ np.array([1, 1j])) ** k)
    ps = np.linspace(0, 2 * np.pi, 20001)
    v = f(ps)
    out = []
    for sgn in (-1, 1):                                  # the minimum, then the maximum
        p0 = ps[np.argmax(sgn * v)]
        r = minimize_scalar(lambda p: -sgn * f(p), bounds=(p0 - 1e-3, p0 + 1e-3), method="bounded",
                            options=dict(xatol=1e-12))
        out.append(sgn * max(np.max(sgn * v), -r.fun))
    return out


def compute():
    y, lh, boots, S = confidence_set()
    cert = certify(lh, S)
    ks = np.arange(1, K + 1)
    ex = np.array([exact(lh, S, k) for k in ks])
    pred = AMP * np.real(boots[:, None] ** ks[None, :])
    bands = np.percentile(pred, [2.5, 97.5], axis=0)
    rng = np.random.default_rng(7)
    r, p = np.sqrt(rng.uniform(0, 1, 4000)), rng.uniform(0, 2 * np.pi, 4000)
    L = np.linalg.cholesky(CHI * S)
    samples = lh + (L @ np.stack([r * np.cos(p), r * np.sin(p)])).T @ np.array([1, 1j])
    sp = AMP * np.real(samples[:, None] ** ks[None, :])
    return dict(y=y, lh=lh, boots=boots, S=S, cert=cert, exact=ex, boot_band=bands,
                samp=(sp.min(0), sp.max(0)), ks=ks)


def checks(out):
    lh, S, cert, ks = out["lh"], out["S"], out["cert"], out["ks"]
    assert abs(LAM0 - (0.6457 + 0.6015j)) < 1e-4 and abs(AMP - np.sqrt(3) / 2) < 1e-12
    Gh, wh = -np.log(abs(lh)) / TAU, np.angle(lh) / TAU
    assert abs(Gh - 0.493) < 5e-4 and abs(wh - 3.057) < 5e-4 and abs(lh - (0.638 + 0.612j)) < 1e-3
    bG, bw = -np.log(abs(out["boots"])) / TAU, np.angle(out["boots"]) / TAU
    assert abs(bG.std() - 0.076) < 5e-4 and abs(bw.std() - 0.139) < 5e-4
    assert abs(CHI - 5.99) < 5e-3
    assert abs(mahalanobis2(LAM0, lh, S) - 0.18) < 5e-3                    # the truth lies in E
    assert abs(np.mean(mahalanobis2(out["boots"], lh, S) <= CHI) - 0.93) < 5e-3
    lo, hi = np.array([c["lo"] for c in cert]), np.array([c["hi"] for c in cert])
    scale = np.array([c["scale"] for c in cert])
    for c in cert:
        assert c["status"] == ("optimal", "optimal"), c
    for c in cert:                                       # the Gram block of the lowest order in two variables
        d = (c["k"] + 1) // 2
        assert c["block"] == (d + 2) * (d + 1) // 2
    assert cert[-1]["block"] == 66
    assert np.all(np.abs(lo - out["exact"][:, 0]) / scale < 1e-6) and np.all(np.abs(hi - out["exact"][:, 1]) / scale < 1e-6)
    truth = AMP * np.real(LAM0 ** ks)
    assert np.all(lo <= truth) and np.all(truth <= hi)
    # harmonic: no point inside E beats the boundary
    smin, smax = out["samp"]
    assert np.all(smin >= lo - 1e-9) and np.all(smax <= hi + 1e-9)
    gap = np.maximum(hi - smax, smin - lo) / (hi - lo)
    assert gap.max() < 0.01 and gap.min() > 1.5e-3
    bl, bh = out["boot_band"]
    ratio = (hi - lo) / (bh - bl)
    assert 1.2 < ratio.min() and ratio.max() < 1.9
    poke = (bl < lo) | (bh > hi)                         # the bootstrap band leaves the certified one ...
    assert list(ks[poke]) == [4, 8] and np.max(np.maximum(lo - bl, bh - hi)[poke] / scale[poke]) < 0.019
    pred = AMP * np.real(out["boots"][:, None] ** ks[None, :])
    assert abs(np.mean(pred[:, 3] > hi[3]) - 0.04) < 1e-9
    inside = mahalanobis2(out["boots"], lh, S) <= CHI     # ... only through the bootstrap poles outside E
    assert np.all(pred[inside].max(0) <= hi + 1e-9) and np.all(pred[inside].min(0) >= lo - 1e-9)


def main():
    out = compute()
    checks(out)
    lh, S, cert, ks = out["lh"], out["S"], out["cert"], out["ks"]
    lo, hi = np.array([c["lo"] for c in cert]), np.array([c["hi"] for c in cert])
    bl, bh = out["boot_band"]
    tk = ks * TAU
    b = []

    # ---- (a) the plane of the pole
    R = 1.4 * np.sqrt(CHI * np.diag(S)).max()
    pa = N.Plot((lh.real - R, lh.real + R), (lh.imag - R, lh.imag + R), 3.4, 3.4, origin=(0.9, 4.6))
    nice = lambda c: [v for v in np.arange(np.ceil((c - R) / 0.05) * 0.05, c + R, 0.05)]
    b += pa.axes(xlabel="{\\scriptsize $\\mathrm{Re}\\,\\lambda$}", ylabel="{\\scriptsize $\\mathrm{Im}\\,\\lambda$}",
                 xticks=[(v, "$%.2f$" % v) for v in nice(lh.real)], yticks=[(v, "$%.2f$" % v) for v in nice(lh.imag)])
    for z in out["boots"]:
        if abs(z.real - lh.real) < R and abs(z.imag - lh.imag) < R:
            b.append("\\fill[ndInk!45] %s circle (0.5pt);" % pa.pt(z.real, z.imag))
    L = np.linalg.cholesky(CHI * S)
    ell = [lh + (L @ [np.cos(p), np.sin(p)]) @ [1, 1j] for p in np.linspace(0, 2 * np.pi, 121)]
    b.append("\\filldraw[ndBlue, fill=ndSky, fill opacity=0.3, line width=0.9pt] %s -- cycle;"
             % " -- ".join(pa.pt(z.real, z.imag) for z in ell))
    b.append("\\fill[ndBlue] %s circle (1.7pt);" % pa.pt(lh.real, lh.imag))
    q = pa.P(LAM0.real, LAM0.imag)
    b.append("\\draw[ndRed, line width=0.9pt] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
             % (q[0] - 0.07, q[1] - 0.07, q[0] + 0.07, q[1] + 0.07, q[0] - 0.07, q[1] + 0.07, q[0] + 0.07, q[1] - 0.07))
    zE = max(ell, key=lambda z: z.real + z.imag)
    b.append("\\node[font=\\small, text=ndBlue, anchor=south west] at %s {$E$};" % pa.pt(zE.real, zE.imag))
    b.append("\\node[note, anchor=south, text=ndBlue] (lh) at %s {identified};" % pa.pt(lh.real + 0.45 * R, lh.imag + 0.72 * R))
    b.append("\\draw[pointer] (lh.south) to[bend left=20] %s;" % pa.pt(lh.real + 0.004, lh.imag + 0.005))
    b.append("\\node[note, anchor=north east, text=ndRed] (l0) at %s {true};" % pa.pt(lh.real - 0.55 * R, lh.imag - 0.7 * R))
    b.append("\\draw[pointer] (l0.east) to[bend right=20] %s;" % pa.pt(LAM0.real - 0.005, LAM0.imag - 0.005))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,3.95) {(a) the pole, and its set $E$};" % pa.P(lh.real, 0)[0])

    # ---- (b) the band
    tt = np.linspace(0, 5.1, 400)
    nominal = lambda t: AMP * np.exp(np.log(abs(lh)) * t / TAU) * np.cos(np.angle(lh) * t / TAU)
    truth = lambda t: AMP * np.exp(-Q.GAMMA * t) * np.cos(Q.OMEGA * t)
    pb = N.Plot((0, 5.2), (-0.75, 0.95), 10.2, 3.4, origin=(5.9, 4.6))
    b += pb.axes(xlabel="{\\scriptsize $t$}", ylabel="{\\scriptsize $\\langle\\sigma_x\\rangle$}",
                 xticks=[(1, "$1$"), (2, "$2$"), (3, "$3$"), (4, "$4$"), (5, "$5$")],
                 yticks=[(-0.5, "$-0.5$"), (0, "$0$"), (0.5, "$0.5$")])
    b.append(pb.curve([0, 5.2], [0, 0], "faint"))
    b += band(pb, tk, lo, hi, bl, bh)
    b.append(pb.curve(tt, nominal(tt), "line width=0.8pt, ndBlue"))
    b.append(pb.curve(tt, truth(tt), "line width=0.8pt, ndRed, dash pattern=on 3pt off 2pt"))
    sig = np.sqrt((1 - out["y"] ** 2) / SHOTS)
    for k in range(J):
        p0, p1 = pb.P(k * TAU, out["y"][k] - sig[k]), pb.P(k * TAU, out["y"][k] + sig[k])
        b.append("\\draw[ndInk, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (p0[0], p0[1], p1[0], p1[1]))
        b.append("\\fill[ndInk] %s circle (1.5pt);" % pb.pt(k * TAU, out["y"][k]))
    q0, q1 = pb.P(2.0 + TAU / 2, -0.75), pb.P(2.0 + TAU / 2, 0.95)
    b.append("\\draw[dashedink] (%.3f,%.3f) -- (%.3f,%.3f);" % (q0[0], q0[1], q1[0], q1[1]))
    b.append("\\node[note, anchor=north east] at %s {the record:\\\\ $200$ shots a point};" % pb.pt(2.05, 0.95))
    b.append("\\node[note, anchor=north west] at %s {prediction};" % pb.pt(2.2, 0.95))
    b += key(pb, 3.4, 0.88)
    b.append("\\node[font=\\small, anchor=north] at (%.3f,3.95) {(b) the record and the predictions};" % pb.P(2.6, 0)[0])

    # ---- (c) the same, times exp(Gamma^ t)
    gh = -np.log(abs(lh)) / TAU
    pc = N.Plot((0, 5.2), (-2.3, 2.1), 10.2, 3.4, origin=(0.9, -0.6))
    b += pc.axes(xlabel="{\\scriptsize $t$}", ylabel="{\\scriptsize $e^{\\hat\\Gamma t}\\langle\\sigma_x\\rangle$}",
                 xticks=[(1, "$1$"), (2, "$2$"), (3, "$3$"), (4, "$4$"), (5, "$5$")],
                 yticks=[(-2, "$-2$"), (-1, "$-1$"), (0, "$0$"), (1, "$1$"), (2, "$2$")])
    b.append(pc.curve([0, 5.2], [0, 0], "faint"))
    e = np.exp(gh * tk)
    b += band(pc, tk, lo * e, hi * e, bl * e, bh * e)
    b.append(pc.curve(tt, nominal(tt) * np.exp(gh * tt), "line width=0.8pt, ndBlue"))
    b.append(pc.curve(tt, truth(tt) * np.exp(gh * tt), "line width=0.8pt, ndRed, dash pattern=on 3pt off 2pt"))
    q0, q1 = pc.P(2.0 + TAU / 2, -2.3), pc.P(2.0 + TAU / 2, 2.1)
    b.append("\\draw[dashedink] (%.3f,%.3f) -- (%.3f,%.3f);" % (q0[0], q0[1], q1[0], q1[1]))
    b.append("\\node[note, anchor=north west] at %s {the uncertainty in $\\omega$\\\\ becomes one of phase};" % pc.pt(0.15, 2.1))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,-1.35) {(c) the same, with the decay $e^{-\\hat\\Gamma t}$ divided out};"
             % pc.P(2.6, 0)[0])

    # ---- (d) the widths relative to the envelope, and the times
    scale = np.array([c["scale"] for c in cert])
    pd = N.Plot((0, 5.2), (0, 3.6), 3.5, 3.4, origin=(12.6, -0.6))
    b += pd.axes(xlabel="{\\scriptsize $t$}", ylabel="{\\scriptsize width / envelope}",
                 xticks=[(1, "$1$"), (3, "$3$"), (5, "$5$")], yticks=[(1, "$1$"), (2, "$2$"), (3, "$3$")])
    b.append(pd.curve(np.r_[0, tk], np.r_[0, (hi - lo) / scale], "line width=0.9pt, ndBlue"))
    b.append(pd.curve(np.r_[0, tk], np.r_[0, (bh - bl) / scale], "line width=0.9pt, ndGold"))
    b.append("\\node[note, anchor=south east, text=ndBlue] at %s {certified};" % pd.pt(2.1, 1.9))
    b.append("\\node[note, anchor=north west, text=ndGold!80!ndInk] at %s {bootstrap $95\\%%$};" % pd.pt(2.6, 0.55))
    secs = sum(c["secs"] for c in cert)
    b.append("\\node[font=\\tiny, anchor=north west, align=left] at %s {order $\\lceil k/2\\rceil$ for $k=t/\\tau$:\\\\ blocks up to $66$,\\\\ %.0f s in all (SCS)};"
             % (pd.pt(0.2, 3.55), secs))
    b.append("\\node[font=\\small, anchor=north] at (%.3f,-1.35) {(d) the widths};" % pd.P(2.6, 0)[0])
    N.write("ch11_band", b)
    return out


def band(pl, tk, lo, hi, bl, bh):
    """The certified band (pale polygon, ticks at the bounds) and the bootstrap band (bars)."""
    b = []
    poly = [pl.P(0, AMP)] + [pl.P(t, v) for t, v in zip(tk, hi)] + [pl.P(t, v) for t, v in zip(tk[::-1], lo[::-1])]
    b.append("\\fill[ndSky, fill opacity=0.8] %s -- cycle;" % " -- ".join(N.pt(p) for p in poly))
    for t, l, h, cl, ch in zip(tk, lo, hi, bl, bh):
        p0, p1 = pl.P(t, cl), pl.P(t, ch)
        b.append("\\draw[ndGold, line width=2.2pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (p0[0], p0[1], p1[0], p1[1]))
        for v in (l, h):
            q = pl.P(t, v)
            b.append("\\draw[ndBlue, line width=0.7pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (q[0] - 0.07, q[1], q[0] + 0.07, q[1]))
    return b


def key(pl, x, y):
    b = []
    rows = [("\\draw[ndBlue, line width=0.8pt] (%.3f,%.3f) -- (%.3f,%.3f);", "identified model"),
            ("\\draw[ndRed, line width=0.8pt, dash pattern=on 3pt off 2pt] (%.3f,%.3f) -- (%.3f,%.3f);", "true model"),
            ("\\draw[ndSky, line width=5pt] (%.3f,%.3f) -- (%.3f,%.3f);", "certified band over $E$"),
            ("\\draw[ndGold, line width=2.2pt] (%.3f,%.3f) -- (%.3f,%.3f);", "bootstrap, $95\\%$ pointwise")]
    for j, (fmt, lab) in enumerate(rows):
        p = pl.P(x, y - 0.17 * j)
        if j == 3:
            b.append(fmt % (p[0] + 0.15, p[1] - 0.09, p[0] + 0.15, p[1] + 0.09))
        else:
            b.append(fmt % (p[0], p[1], p[0] + 0.3, p[1]))
        b.append("\\node[font=\\scriptsize, anchor=west] at (%.3f,%.3f) {%s};" % (p[0] + 0.38, p[1], lab))
    return b


if __name__ == "__main__":
    t0 = time.perf_counter()
    out = main()
    lh = out["lh"]
    print("lambda^ = %.4f%+.4fi  Gamma^ %.4f  omega^ %.4f  truth m2 %.3f" % (lh.real, lh.imag, -np.log(abs(lh)) / TAU,
                                                                         np.angle(lh) / TAU, mahalanobis2(LAM0, lh, out["S"])))
    for c, ex, bl, bh, smin, smax in zip(out["cert"], out["exact"], *out["boot_band"], *out["samp"]):
        print("k=%2d t=%.2f  certified [%.5f, %.5f]  exact [%.5f, %.5f]  boot [%.5f, %.5f]  sampled [%.5f, %.5f]  %s block %d  %.1f s"
              % (c["k"], c["k"] * TAU, c["lo"], c["hi"], ex[0], ex[1], bl, bh, smin, smax, c["status"], c["block"], c["secs"]))
    print("total %.0f s; written figures/ch11_band.tex" % (time.perf_counter() - t0))
