"""ch05_expmap.py -- sampling as the exponential map  z = e^{lambda tau}.

Sampling every tau replaces a pole lambda of the continuous-time generator by
the eigenvalue z = e^{lambda tau} of G_tau = e^{A tau}.  The map wraps the
lambda-plane onto the z-plane:
  * a vertical line Re lambda = alpha (fixed decay rate) becomes the circle
    |z| = e^{alpha tau}; the imaginary axis becomes the unit circle;
  * a horizontal line Im lambda = omega (fixed frequency) becomes the ray at
    angle omega tau;
  * each horizontal strip of height 2 pi / tau covers the punctured z-plane
    exactly once, so lambda and its aliases lambda + 2 pi i k / tau land on
    the same z.
(a) The lambda-plane of the running example (poles 0, -gamma_1,
    -Gamma +- i omega) cut into strips, with the aliases of -Gamma + i omega.
(b) The z-plane of the samples.  tau = 1.6, so that omega = 3 lies outside
    the Nyquist band |Im lambda| < pi / tau = 1.96 and is folded to -0.93.
Writes figures/ch05_expmap.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import OMEGA, GAMMA, GAMMA1, POLES

TAU = 1.6
H = 2 * np.pi / TAU                     # strip height
NYQ = np.pi / TAU
# panel (a): lambda-plane, horizontal axis stretched
SR, SI, AX = 2.0, 0.52, 3.0
RE_MIN, RE_MAX = -1.0, 0.45
# panel (b): z-plane
SZ, BX = 2.35, 11.0


def Pa(lam):
    return np.array([AX + SR * np.real(lam), SI * np.imag(lam)])


def Pb(z):
    return np.array([BX + SZ * np.real(z), SZ * np.imag(z)])


def main():
    lam = -GAMMA + 1j * OMEGA
    aliases = [lam + 2j * np.pi * k / TAU for k in (-2, -1, 0)]
    z1 = np.exp(lam * TAU)
    for a in aliases:
        assert abs(np.exp(a * TAU) - z1) < 1e-12               # all aliases land on one z
    principal = np.log(z1) / TAU
    assert abs(principal - aliases[1]) < 1e-12 and abs(principal.imag) < NYQ   # folded to -0.93 i
    assert abs(abs(z1) - np.exp(-GAMMA * TAU)) < 1e-12
    assert abs(np.angle(z1) - (OMEGA * TAU - 2 * np.pi)) < 1e-12
    assert all(abs(a.imag) <= 3 * NYQ for a in aliases)          # all inside the drawn strips

    b = []
    # ---------------- (a) strips
    for k, col in [(-1, "ndSky"), (0, "ndGold!35"), (1, "ndSky")]:
        lo, hi = (2 * k - 1) * NYQ, (2 * k + 1) * NYQ
        b.append("\\fill[%s, fill opacity=0.55] %s rectangle %s;" % (col, N.pt(Pa(RE_MIN + 1j * lo)), N.pt(Pa(RE_MAX + 1j * hi))))
    for k in (-1, 0, 1, 2):
        y = (2 * k - 1) * NYQ
        b.append("\\draw[dashedink] %s -- %s;" % (N.pt(Pa(RE_MIN + 1j * y)), N.pt(Pa(RE_MAX + 1j * y))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk, font=\\small] {$\\mathrm{Re}\\,\\lambda$};"
             % (N.pt(Pa(RE_MIN)), N.pt(Pa(RE_MAX + 0.12))))
    b.append("\\draw[ink, ->, >=stealth] %s -- %s node[above, text=ndInk, font=\\small] {$\\mathrm{Im}\\,\\lambda$};"
             % (N.pt(Pa(-3.15j * NYQ)), N.pt(Pa(3.35j * NYQ))))
    # vertical line of fixed decay rate -Gamma
    b.append("\\draw[line width=1.0pt, ndBlue] %s -- %s;" % (N.pt(Pa(-GAMMA - 3j * NYQ)), N.pt(Pa(-GAMMA + 3j * NYQ))))
    # horizontal lines of fixed frequency: omega and its aliases (red), 0 (gold)
    for k, a in enumerate(aliases):
        b.append("\\draw[line width=%s, ndRed] %s -- %s;" % ("1.0pt" if k == 2 else "0.6pt, dash pattern=on 3pt off 1.5pt",
                                                            N.pt(Pa(RE_MIN + 1j * a.imag)), N.pt(Pa(RE_MAX + 1j * a.imag))))
    b.append("\\draw[line width=1.0pt, ndGold] %s -- %s;" % (N.pt(Pa(RE_MIN)), N.pt(Pa(RE_MAX))))
    # ticks
    for y, lab in [(NYQ, "$\\pi/\\tau$"), (-NYQ, "$-\\pi/\\tau$"), (OMEGA, "$\\omega$")]:
        p = Pa(1j * y)
        b.append("\\node[font=\\scriptsize, anchor=west, fill=white, fill opacity=0.7, text opacity=1, inner sep=1pt] at (%.3f,%.3f) {%s};" % (p[0] + 0.05, p[1] + 0.13, lab))
    pg = Pa(-GAMMA - 3j * NYQ)
    b.append("\\node[font=\\scriptsize, text=ndBlue, below] at %s {$-\\Gamma$};" % N.pt(pg))
    # strip height bracket on the left
    xl = Pa(RE_MAX)[0] + 0.2
    b.append("\\draw[ink, |-|] (%.3f,%.3f) -- (%.3f,%.3f);" % (xl, Pa(1j * NYQ)[1], xl, Pa(3j * NYQ)[1]))
    b.append("\\node[font=\\scriptsize, anchor=west] at (%.3f,%.3f) {$2\\pi/\\tau$};" % (xl + 0.03, Pa(2j * NYQ)[1]))
    # poles
    for p in POLES:
        if abs(p - lam) < 1e-9:
            continue
        b.append("\\node[dot, minimum size=4.4pt] at %s {};" % N.pt(Pa(p)))
    for k, a in enumerate(aliases):
        if k == 2:
            b.append("\\node[dot, fill=ndRed, minimum size=5pt] at %s {};" % N.pt(Pa(a)))
        else:
            b.append("\\draw[ndRed, line width=1pt, fill=white] %s circle (2.4pt);" % N.pt(Pa(a)))
    # notes (a)
    xn = Pa(RE_MIN)[0] - 0.75
    b.append("\\node[note, anchor=east] (ny) at (%.3f,%.3f) {Nyquist strip\\\\ $\\abs{\\mathrm{Im}\\,\\lambda}<\\pi/\\tau$};" % (xn, -0.35))
    b.append("\\draw[pointer] (ny.east) -- (%.3f,%.3f);" % (Pa(RE_MIN + 0.08)[0], -0.35))
    b.append("\\node[note, anchor=east, text=ndRed] (al) at (%.3f,%.3f) {aliases\\\\ $\\lambda+2\\pi\\imath k/\\tau$:\\\\ identical samples};" % (xn, -1.9))
    b.append("\\draw[pointer, ndRed!70] (al.east) to[bend right=12] %s;" % N.pt(Pa(aliases[0]) + np.array([-0.1, 0.03])))
    b.append("\\draw[pointer, ndRed!70] (al.east) to[bend left=12] %s;" % N.pt(Pa(aliases[1]) + np.array([-0.1, -0.03])))
    b.append("\\node[note, anchor=east, text=ndRed] (pl) at (%.3f,%.3f) {pole\\\\ $-\\Gamma+\\imath\\omega$};" % (xn, 1.85))
    b.append("\\draw[pointer, ndRed!70] (pl.east) -- %s;" % N.pt(Pa(aliases[2]) + np.array([-0.12, 0.0])))
    # ---------------- arrow between the panels
    b.append("\\draw[vec, ndInk!70] (5.35,0.95) to[bend left=18] node[above, font=\\small, text=ndInk] {$z=\\ee^{\\lambda\\tau}$} (7.75,0.95);")
    b.append("\\node[note] at (6.55,0.3) {wraps each strip\\\\ once round $0$};")
    # ---------------- (b) z-plane
    b.append("\\fill[ndSky, fill opacity=0.35] %s circle (%.3f);" % (N.pt(Pb(0)), SZ))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk, font=\\small] {$\\mathrm{Re}\\,z$};" % (N.pt(Pb(-1.12)), N.pt(Pb(1.35))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, text=ndInk, font=\\small] {$\\mathrm{Im}\\,z$};" % (N.pt(Pb(-1.25j)), N.pt(Pb(1.3j))))
    b.append("\\draw[ink] %s circle (%.3f);" % (N.pt(Pb(0)), SZ))
    r = np.exp(-GAMMA * TAU)
    b.append("\\draw[line width=1.0pt, ndBlue] %s circle (%.3f);" % (N.pt(Pb(0)), SZ * r))
    b.append("\\draw[line width=1.0pt, ndGold] %s -- %s;" % (N.pt(Pb(0)), N.pt(Pb(1.25))))
    b.append("\\draw[line width=1.0pt, ndRed] %s -- %s;" % (N.pt(Pb(0)), N.pt(Pb(1.25 * np.exp(1j * OMEGA * TAU)))))
    b.append("\\draw[dashedink, line width=0.8pt] %s -- %s;" % (N.pt(Pb(0)), N.pt(Pb(-1.1))))
    # angle omega tau
    th = np.linspace(0, OMEGA * TAU, 80)
    b.append(N.polyline([Pb(0.2 * np.exp(1j * t)) for t in th], "line width=0.5pt, ndRed, ->, >=stealth"))
    b.append("\\node[font=\\scriptsize, text=ndRed] at %s {$\\omega\\tau$};" % N.pt(Pb(0.33 * np.exp(1j * 2.3))))
    zp = np.exp(POLES * TAU)
    for p, z in zip(POLES, zp):
        if abs(p - lam) < 1e-9:
            b.append("\\node[dot, fill=ndRed, minimum size=5pt] at %s {};" % N.pt(Pb(z)))
        else:
            b.append("\\node[dot, minimum size=4.4pt] at %s {};" % N.pt(Pb(z)))
    assert abs(zp[0] - 1) < 1e-12 and abs(zp[1] - np.exp(-GAMMA1 * TAU)) < 1e-12
    # notes (b)
    xr = BX + SZ + 0.45
    b.append("\\node[note, anchor=west] (uc) at (%.3f,%.3f) {unit circle:\\\\ no decay, $\\mathrm{Re}\\,\\lambda=0$};" % (xr, 2.05))
    b.append("\\draw[pointer] (uc.west) to[bend right=10] %s;" % N.pt(Pb(np.exp(0.75j))))
    b.append("\\node[note, anchor=west, text=ndBlue] (bc) at (%.3f,%.3f) {circle of radius $\\ee^{-\\Gamma\\tau}$:\\\\ all poles with\\\\ decay rate $\\Gamma$};" % (xr, 0.75))
    b.append("\\draw[pointer, ndBlue!70] (bc.west) to[bend right=8] %s;" % N.pt(Pb(r * np.exp(0.35j))))
    b.append("\\node[note, anchor=west, text=ndRed] (rz) at (%.3f,%.3f) {the three red dots\\\\ of (a) all map here,\\\\ on the ray at angle $\\omega\\tau$};" % (xr, -1.15))
    b.append("\\draw[pointer, ndRed!70] (rz.west) to[bend left=10] %s;" % N.pt(Pb(z1) + np.array([0.1, -0.02])))
    b.append("\\node[note, anchor=east] (ng) at (%.3f,%.3f) {strip edges $\\pm\\pi/\\tau$\\\\ map to the negative\\\\ real axis};" % (BX - SZ - 0.05, -1.05))
    b.append("\\draw[pointer] (ng.north) to[bend left=25] %s;" % N.pt(Pb(-0.7 - 0.02j)))
    b.append("\\node[font=\\scriptsize, anchor=north west] at %s {$1$};" % N.pt(Pb(1)))
    b.append("\\node[font=\\scriptsize, text=ndInk!80, anchor=south west] at %s {$\\lambda=0$};" % N.pt(Pb(1) + np.array([0.02, 0.02])))
    # panel labels
    yb = -3.75
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) the $\\lambda$-plane in strips of height $2\\pi/\\tau$};" % (2.0, yb))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) the $z$-plane of the samples};" % (BX + 0.6, yb))
    N.write("ch05_expmap", b)
    print("written figures/ch05_expmap.tex; z = %.4f e^{%.4f i}, folded pole %s" % (abs(z1), np.angle(z1), np.round(principal, 4)))


if __name__ == "__main__":
    main()
