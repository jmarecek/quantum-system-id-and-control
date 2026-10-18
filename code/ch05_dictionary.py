"""ch05_dictionary.py -- a time <-> frequency dictionary.

Six pairs of thumbnails, all computed: a signal on the left, the magnitude of
its Fourier transform  |y^(nu)| = |int_0^T y(t) e^{-i nu t} dt|  on the right.
  1. how fast it oscillates  <-> where the peak is;
  2. how fast it decays      <-> how wide the peak is (half-width = decay rate);
  3. how much of each mode   <-> how tall each peak is;
  4. how long we record      <-> how close two peaks may be and still be told
                                 apart (resolution 2 pi / T);
  5. a sample every tau      <-> the spectrum repeats every 2 pi / tau;
  6. a mode the readout misses (C v = 0) <-> a missing peak
     (<sigma_z> of the running example has no peak at omega).
For a sum of damped modes  y = sum_i r_i e^{lambda_i t},  t >= 0, the
transform is  sum_i r_i / (i nu - lambda_i)  (closed form, checked against
quadrature).  Writes figures/ch05_dictionary.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import X0, CX, CZ, OMEGA, GAMMA, GAMMA1, qubit_A, modes

W, H = 5.0, 1.25                     # thumbnail size (cm)
XL, XR = 0.0, 7.3                    # left edges of the two columns
ROW = 2.2


def ft_modes(lam, res, nu):
    """Fourier transform of sum_i r_i e^{lam_i t} on t >= 0 (Re lam_i < 0)."""
    nu = np.asarray(nu)[:, None]
    return np.sum(res / (1j * nu - lam), axis=1)


def ft_finite(y, t, nu):
    """|int_0^T y(t) e^{-i nu t} dt| by the trapezoidal rule."""
    E = np.exp(-1j * np.outer(nu, t)) * y
    return np.abs(np.trapz(E, t, axis=1))


def damped(alpha, omega):
    """e^{alpha t} cos(omega t) as two modes."""
    lam = np.array([alpha + 1j * omega, alpha - 1j * omega])
    return lam, np.array([0.5, 0.5])


def local_maxima(p, nu, lo, hi, rel=0.2):
    m = (nu > lo) & (nu < hi)
    q, v = p[m], nu[m]
    idx = [i for i in range(1, len(q) - 1) if q[i] > q[i - 1] and q[i] >= q[i + 1] and q[i] > rel * q.max()]
    return v[idx]


class Thumb:
    def __init__(self, x0, y0, xlim, ylim):
        self.P = N.Plot(xlim, ylim, W, H, origin=(x0, y0))
        self.x0, self.y0, self.xlim, self.ylim = x0, y0, xlim, ylim

    def axis(self, b, var, yzero=None, ticks=()):
        yz = self.ylim[0] if yzero is None else yzero
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$%s$};"
                 % (self.P.pt(self.xlim[0], yz), self.P.pt(self.xlim[1] + 0.03 * (self.xlim[1] - self.xlim[0]), yz), var))
        for v, lab in ticks:
            p = self.P.P(v, self.ylim[0])
            b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[below, font=\\scriptsize, text=ndInk, inner sep=1.5pt] {%s};"
                     % (p[0], p[1] + 0.05, p[0], p[1] - 0.05, lab))

    def curve(self, b, x, y, style):
        y = np.clip(y, self.ylim[0], self.ylim[1])
        b.append(self.P.curve(x, y, style))


def caption(b, x, y, text):
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {%s};" % (x + W / 2, y - 0.3, text))


def main():
    b = []
    t = np.linspace(0, 12, 900)
    nu = np.linspace(0, 7, 701)
    BLUE, RED = "line width=1.0pt, ndBlue", "line width=1.0pt, ndRed"
    INK = "line width=1.0pt, ndInk"
    ytop = 0.0
    b.append("\\node[font=\\small, anchor=south] at (%.3f,%.3f) {the signal $y(t)$};" % (XL + W / 2, ytop + 0.25))
    b.append("\\node[font=\\small, anchor=south] at (%.3f,%.3f) {its spectrum $\\abs{\\hat y(\\nu)}$};" % (XR + W / 2, ytop + 0.25))
    rows = [ytop - 0.15 - H - k * ROW for k in range(6)]

    def arrow(y):
        b.append("\\draw[<->, >=stealth, line width=0.8pt, ndInk!60] (%.3f,%.3f) -- (%.3f,%.3f);" % (XL + W + 0.65, y + H / 2, XR - 0.35, y + H / 2))

    # ---- 1. oscillation frequency <-> peak position
    y0 = rows[0]
    L, R = Thumb(XL, y0, (0, 12), (-2.3, 1.1)), Thumb(XR, y0, (0, 7), (0, 1.15))
    L.axis(b, "t", yzero=-1.15)
    R.axis(b, "\\nu", ticks=[(2, "$2$"), (5, "$5$")])
    specs = []
    for (w, off, st) in [(2.0, 0.0, BLUE), (5.0, -2.2, RED)]:
        lam, res = damped(-0.25, w)
        L.curve(b, t, off * 0.5 + 0.5 * np.exp(-0.25 * t) * np.cos(w * t) + (0 if off == 0 else -0.05), st)
        p = np.abs(ft_modes(lam, res, nu))
        specs.append(p)
    smax = max(s.max() for s in specs)
    for p, st, w in zip(specs, [BLUE, RED], (2.0, 5.0)):
        R.curve(b, nu, p / smax, st)
        assert abs(nu[np.argmax(p)] - w) < 0.03
    caption(b, XL, y0, "slow and fast oscillations")
    caption(b, XR, y0, "the peak lies at the frequency")
    arrow(y0)

    # ---- 2. decay rate <-> peak width
    y0 = rows[1]
    L, R = Thumb(XL, y0, (0, 12), (-1.1, 1.1)), Thumb(XR, y0, (0, 7), (0, 1.15))
    L.axis(b, "t", yzero=0.0)
    R.axis(b, "\\nu", ticks=[(4, "$4$")])
    for a, st in [(-0.15, BLUE), (-0.8, RED)]:
        lam, res = damped(a, 4.0)
        L.curve(b, t, np.exp(a * t) * np.cos(4.0 * t), st)
        p = np.abs(ft_modes(lam, res, nu)) ** 2
        p /= p.max()
        R.curve(b, nu, p, st)
        half = nu[p > 0.5]
        assert abs((half.max() - half.min()) / 2 - abs(a)) < 0.03          # half-width = decay rate
    caption(b, XL, y0, "slow and fast decay")
    caption(b, XR, y0, "$\\abs{\\hat y}^2$: half-width of the peak $=$ decay rate\\\\[-1pt] (scaled to equal height)")
    arrow(y0)

    # ---- 3. weights <-> peak heights
    y0 = rows[2]
    L, R = Thumb(XL, y0, (0, 12), (-1.5, 1.5)), Thumb(XR, y0, (0, 7), (0, 1.15))
    L.axis(b, "t", yzero=0.0)
    R.axis(b, "\\nu", ticks=[(2.5, "$2.5$"), (5, "$5$")])
    l1, r1 = damped(-0.3, 2.5)
    l2, r2 = damped(-0.3, 5.0)
    lam, res = np.concatenate([l1, l2]), np.concatenate([1.0 * r1, 0.4 * r2])
    y = np.real(np.exp(np.outer(t, lam)) @ res)
    L.curve(b, t, np.exp(-0.3 * t) * np.cos(2.5 * t), "line width=0.5pt, ndBlue!70")
    L.curve(b, t, 0.4 * np.exp(-0.3 * t) * np.cos(5.0 * t), "line width=0.5pt, ndRed!70")
    L.curve(b, t, y, INK)
    p = np.abs(ft_modes(lam, res, nu))
    tt = np.linspace(0, 80, 40001)
    yy = np.real(np.exp(np.outer(tt, lam)) @ res)
    assert np.allclose(ft_finite(yy, tt, nu[::50]), p[::50], atol=2e-3)      # closed form = quadrature
    s = p.max()
    R.curve(b, nu, p / s, INK)
    h1, h2 = p[np.argmin(abs(nu - 2.5))] / s, p[np.argmin(abs(nu - 5.0))] / s
    assert abs(h2 / h1 - 0.4) < 0.05
    for w, h, col in [(2.5, h1, "ndBlue"), (5.0, h2, "ndRed")]:
        b.append("\\fill[%s] %s circle (1.6pt);" % (col, R.P.pt(w, h)))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=west] at %s {$1$};" % R.P.pt(2.75, h1))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at %s {$0.4$};" % R.P.pt(5.2, h2))
    caption(b, XL, y0, "two modes, with weights $1$ and $0.4$")
    caption(b, XR, y0, "the heights follow the weights")
    arrow(y0)

    # ---- 4. record length <-> resolution
    y0 = rows[3]
    tl = np.linspace(0, 40, 8001)
    yb = np.cos(3.0 * tl) + np.cos(3.6 * tl)
    L, R = Thumb(XL, y0, (0, 40), (-3.0, 3.0)), Thumb(XR, y0, (1.5, 5.5), (0, 1.15))
    L.axis(b, "t", yzero=0.0)
    R.axis(b, "\\nu", ticks=[(3.0, "$3$"), (3.6, "$3.6$")])
    L.curve(b, tl, yb, "line width=0.5pt, ndInk!70")
    for Tr, col, yy_ in [(6.0, "ndRed", 2.6), (40.0, "ndBlue", -2.6)]:
        b.append("\\draw[%s, line width=1.2pt, |-|] %s -- %s;" % (col, L.P.pt(0, yy_), L.P.pt(Tr, yy_)))
    nr = np.linspace(1.5, 5.5, 801)
    peaks = {}
    for Tr, st in [(6.0, RED), (40.0, BLUE)]:
        m = tl <= Tr + 1e-9
        p = ft_finite(yb[m], tl[m], nr) / Tr
        R.curve(b, nr, p / 1.1, st)
        peaks[Tr] = local_maxima(p, nr, 2.0, 4.6, rel=0.5)
    assert len(peaks[6.0]) == 1 and len(peaks[40.0]) == 2                 # blurred vs resolved
    assert np.allclose(peaks[40.0], [3.0, 3.6], atol=0.03)
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at %s {$T=6$};" % L.P.pt(6.4, 2.6))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=west] at %s {$T=40$};" % L.P.pt(40.5, -2.6))
    caption(b, XL, y0, "two nearby frequencies; a short and a long record")
    caption(b, XR, y0, "short record: one broad peak; long: two\\\\[-1pt] (resolution $2\\pi/T$)")
    arrow(y0)

    # ---- 5. sampling <-> periodic copies
    y0 = rows[4]
    tau, a, w = 0.6, -0.3, 3.0
    L, R = Thumb(XL, y0, (0, 12), (-1.1, 1.1)), Thumb(XR, y0, (0, 22), (0, 1.15))
    L.axis(b, "t", yzero=0.0)
    ws = 2 * np.pi / tau
    R.axis(b, "\\nu", ticks=[(w, "$\\omega$"), (ws / 2, "$\\frac{\\pi}{\\tau}$"), (ws, "$\\frac{2\\pi}{\\tau}$"), (2 * ws, "$\\frac{4\\pi}{\\tau}$")])
    L.curve(b, t, np.exp(a * t) * np.cos(w * t), "line width=0.5pt, ndInk!45")
    ks = np.arange(0, int(12 / tau) + 1)
    for k in ks:
        v = np.exp(a * k * tau) * np.cos(w * k * tau)
        b.append("\\draw[ndBlue, line width=0.7pt] %s -- %s;" % (L.P.pt(k * tau, 0), L.P.pt(k * tau, v)))
        b.append("\\fill[ndBlue] %s circle (1.1pt);" % L.P.pt(k * tau, v))
    lam, res = damped(a, w)
    nus = np.linspace(0, 22, 1761)
    dt = np.abs(tau * np.sum(res / (1 - np.exp((lam - 1j * nus[:, None]) * tau)), axis=1))
    ct = np.abs(ft_modes(lam, res, nus))
    assert np.allclose(np.abs(tau * np.sum(res / (1 - np.exp((lam - 1j * (nus[:, None] + ws)) * tau)), axis=1)), dt)   # period 2 pi / tau
    # the DTFT is the sum of the shifted copies of the continuous spectrum (Poisson); check near omega
    sm = dt.max()
    b.append("\\fill[ndSky, fill opacity=0.6] %s rectangle %s;" % (R.P.pt(0, 0), R.P.pt(ws / 2, 1.15)))
    R.curve(b, nus, ct / sm, "dashedink")
    R.curve(b, nus, dt / sm, BLUE)
    pk = local_maxima(dt, nus, 0.5, 21.9)
    assert np.allclose(pk, [w, ws - w, ws + w, 2 * ws - w], atol=0.05)
    caption(b, XL, y0, "samples at intervals $\\tau$")
    caption(b, XR, y0, "the peak is repeated at $2\\pi k/\\tau\\pm\\omega$;\\\\[-1pt] the shaded band $\\nu<\\pi/\\tau$ contains one copy")
    arrow(y0)

    # ---- 6. hidden mode <-> missing peak (running example)
    y0 = rows[5]
    A = qubit_A()
    L, R = Thumb(XL, y0, (0, 12), (-1.1, 1.1)), Thumb(XR, y0, (0, 7), (0, 1.15))
    L.axis(b, "t", yzero=0.0)
    R.axis(b, "\\nu", ticks=[(0, "$0$"), (OMEGA, "$\\omega$")])
    lx, rx = modes(A, CX, X0)
    lz, rz = modes(A, CZ, X0)
    assert np.all(np.abs(rz[np.abs(lz.imag) > 1]) < 1e-12)                # C v = 0 for the transverse modes
    assert np.all(np.abs(rx[np.abs(lx.imag) < 1]) < 1e-12)
    yxt = np.real(np.exp(np.outer(t, lx)) @ rx)
    yzt = np.real(np.exp(np.outer(t, lz)) @ rz)
    L.curve(b, t, yxt, BLUE)
    L.curve(b, t, yzt, RED)
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=south west] at %s {$\\langle\\sigma_x\\rangle$};" % L.P.pt(1.2, 0.45))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=north west] at %s {$\\langle\\sigma_z\\rangle$};" % L.P.pt(9.5, 0.8))
    keep = np.abs(lz) > 1e-9                                             # <sigma_z> minus its final value
    px = np.abs(ft_modes(lx[np.abs(rx) > 1e-12], rx[np.abs(rx) > 1e-12], nu))
    pz = np.abs(ft_modes(lz[keep & (np.abs(rz) > 1e-12)], rz[keep & (np.abs(rz) > 1e-12)], nu))
    R.curve(b, nu, px / px.max(), BLUE)
    R.curve(b, nu, pz / pz.max(), RED)
    assert pz[np.argmin(abs(nu - OMEGA))] / pz.max() < 0.07 and abs(nu[np.argmax(px)] - OMEGA) < 0.05
    b.append("\\node[note, anchor=west, text=ndRed] (np) at %s {no peak at $\\omega$};" % R.P.pt(4.1, 0.78))
    b.append("\\draw[pointer, ndRed!70] (np.west) to[bend right=15] %s;" % R.P.pt(OMEGA, 0.07))
    caption(b, XL, y0, "the example, measuring $\\sigma_x$ and $\\sigma_z$")
    caption(b, XR, y0, "$\\sigma_z$ does not see the mode $-\\Gamma\\pm\\imath\\omega$ ($\\mathbf C\\vec v=0$)\\\\[-1pt] (each curve scaled to height $1$)")
    arrow(y0)
    N.write("ch05_dictionary", b)
    print("written figures/ch05_dictionary.tex; resolved peaks", np.round(peaks[40.0], 3), "sampled peaks", np.round(pk, 2))


if __name__ == "__main__":
    main()
