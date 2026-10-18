"""ch05_beats.py -- resolution: two close frequencies need a long record.

The signal y(t) = cos(3.0 t) + cos(3.4 t) beats: it equals
2 cos(0.2 t) cos(3.2 t), with an envelope of period 2 pi / 0.4 ~ 15.7.
It is sampled every tau = 0.25 for a short record (K = 32 samples, K tau = 8)
and a long one (K = 160, K tau = 40).  The spectrum of K samples,
|tau sum_k y_k e^{-i nu k tau}|^2, cannot separate frequencies closer than
about 2 pi / (K tau); the DFT samples it at exactly that spacing.
  (a) the signal, its envelope and the two records;
  (b) the spectra: a single broad peak for the short record, two for the long one.
Writes figures/ch05_beats.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import TAU

W1, W2 = 3.0, 3.4
KS, KL = 32, 160


def y(t):
    return np.cos(W1 * t) + np.cos(W2 * t)


def spectrum(K, nu):
    t = TAU * np.arange(K)
    return np.abs(TAU * np.exp(-1j * np.outer(nu, t)) @ y(t)) ** 2


def peaks(p, nu, lo=2.5, hi=4.0):
    i = np.where((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]) & (nu[1:-1] > lo) & (nu[1:-1] < hi)
                 & (p[1:-1] > 0.3 * p.max()))[0] + 1
    return nu[i]


def main():
    t = np.linspace(0, KL * TAU, 1200)
    assert np.allclose(y(t), 2 * np.cos(0.5 * (W2 - W1) * t) * np.cos(0.5 * (W1 + W2) * t))
    nu = np.linspace(1.8, 4.8, 751)
    ps, pl = spectrum(KS, nu), spectrum(KL, nu)
    res_s, res_l = 2 * np.pi / (KS * TAU), 2 * np.pi / (KL * TAU)
    assert res_s > W2 - W1 > res_l
    pk_s, pk_l = peaks(ps, nu), peaks(pl, nu)
    assert len(pk_s) == 1 and len(pk_l) == 2
    assert np.allclose(pk_l, [W1, W2], atol=0.03)
    # the DFT values are samples of the same curve, every 2 pi / (K tau)
    for K, p in [(KS, ps), (KL, pl)]:
        dft = np.abs(TAU * np.fft.fft(y(TAU * np.arange(K)))) ** 2
        nb = 2 * np.pi * np.arange(K) / (K * TAU)
        sel = (nb > nu[0]) & (nb < nu[-1])
        assert np.allclose(dft[sel], spectrum(K, nb[sel]))
    print("resolution: short %.3f, long %.3f; peaks short %s, long %s" % (res_s, res_l, np.round(pk_s, 3), np.round(pk_l, 3)))

    b = []
    # ---- (a) the signal
    T = KL * TAU
    Pl = N.Plot((0, T), (-2.3, 2.6), 8.4, 4.2, origin=(0.0, 0.7))
    b.append("\\fill[ndRose, fill opacity=0.55] %s rectangle %s;" % (Pl.pt(0, -2.15), Pl.pt(KS * TAU, 2.15)))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk] {$t$};" % (Pl.pt(0, 0), Pl.pt(T + 1.2, 0)))
    env = 2 * np.abs(np.cos(0.5 * (W2 - W1) * t))
    b.append(Pl.curve(t, env, "dashedink"))
    b.append(Pl.curve(t, -env, "dashedink"))
    b.append(Pl.curve(t, y(t), "line width=0.6pt, ndInk"))
    # one beat period, between two zeros of the envelope
    z0, z1 = np.pi / (W2 - W1), 3 * np.pi / (W2 - W1)
    yb = 2.45
    b.append("\\draw[ink, |-|] %s -- %s node[midway, above, font=\\scriptsize] {beat period $2\\pi/(\\omega_2-\\omega_1)\\approx%.0f$};" % (Pl.pt(z0, yb), Pl.pt(z1, yb), 2 * np.pi / (W2 - W1)))
    for z in (z0, z1):
        b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(z, 0), Pl.pt(z, yb)))
    for K, yr, col, name in [(KS, -2.45, "ndRed", "short"), (KL, -2.95, "ndBlue", "long")]:
        b.append("\\draw[line width=1.1pt, %s, |-|] %s -- %s;" % (col, Pl.pt(0, yr), Pl.pt(K * TAU, yr)))
        where = ("anchor=west", Pl.pt(K * TAU + 0.4, yr)) if K == KS else ("anchor=north", Pl.pt(K * TAU / 2, yr - 0.05))
        b.append("\\node[font=\\scriptsize, text=%s, %s] at %s {%s record, $K=%d$ samples, $K\\tau=%g$};" % (col, where[0], where[1], name, K, K * TAU))
    b.append("\\node[font=\\small] at %s {(a) $y(t)=\\cos\\omega_1t+\\cos\\omega_2t$, $\\omega_1=%g$, $\\omega_2=%g$};" % (Pl.pt(T / 2, -4.0), W1, W2))
    # ---- (b) the spectra, each normalised to height 1
    Q = N.Plot((1.8, 4.8), (0, 1.15), 6.2, 4.2, origin=(9.9, 0.7))
    b += Q.axes("$\\nu$", "", xticks=[(W1, "$\\omega_1$"), (W2, "$\\omega_2$")])
    for w in (W1, W2):
        b.append("\\draw[dashedink] %s -- %s;" % (Q.pt(w, 0), Q.pt(w, 1.08)))
    for K, p, col in [(KS, ps, "ndRed"), (KL, pl, "ndBlue")]:
        s = p.max()
        b.append(Q.curve(nu, p / s, "line width=1.3pt, %s" % col))
        nb = 2 * np.pi * np.arange(K) / (K * TAU)
        nb = nb[(nb > 1.85) & (nb < 4.75)]
        b += ["\\fill[%s] %s circle (1.5pt);" % (col, Q.pt(v, spectrum(K, [v])[0] / s)) for v in nb]
    # resolution brackets
    ys_ = 0.5 * ps.max()
    above = nu[(ps >= ys_) & (abs(nu - 3.2) < 1)]
    b.append("\\draw[line width=0.7pt, ndRed, |-|] %s -- %s;" % (Q.pt(3.2 - res_s / 2, 0.5), Q.pt(3.2 + res_s / 2, 0.5)))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=south west] at %s {$2\\pi/(K\\tau)=%.2f$};" % (Q.pt(3.2 + res_s / 2 + 0.02, 0.52), res_s))
    assert abs((above.max() - above.min()) - res_s) < 0.25 * res_s        # the blur is about one resolution cell
    b.append("\\node[note, anchor=east, text=ndRed] (s) at %s {short record:\\\\ a single broad peak};" % Q.pt(2.7, 0.9))
    b.append("\\draw[pointer] (s.east) to[bend left=15] %s;" % Q.pt(3.0, ps[np.argmin(abs(nu - 3.0))] / ps.max() + 0.02))
    b.append("\\node[note, anchor=west, text=ndBlue] (l) at %s {long record:\\\\ two peaks};" % Q.pt(3.8, 0.9))
    b.append("\\draw[pointer] (l.west) to[bend right=15] %s;" % Q.pt(W2 + 0.04, 0.97))
    b.append("\\node[font=\\small, align=center] at (%.3f,%.3f) {(b) $\\abs{\\hat y(\\nu)}^2$ of the two records;\\\\ dots: the DFT, at intervals $2\\pi/(K\\tau)$};" % (Q.P(3.3, 0)[0], Pl.P(0, -4.0)[1] - 0.2))
    N.write("ch05_beats", b)
    print("written figures/ch05_beats.tex")


if __name__ == "__main__":
    main()
