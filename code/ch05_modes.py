"""ch05_modes.py -- the data are a weighted sum of modes.

The running example of the chapter (ch05_spectrum.py), read out along the axis
n = (x + z)/sqrt 2.  By the modal expansion
    <sigma_n>(t) = sum_i r_i e^{lambda_i t},
the signal is the sum of three ingredients, one per pole (or conjugate pair):
    a constant (lambda = 0, the steady state),
    a decay (lambda = -gamma_1, relaxation of <Z>),
    a damped oscillation (lambda = -Gamma +- i omega, precession of <X>),
each already multiplied by its residue r_i.  Identification undoes the sum.
Writes figures/ch05_modes.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import qubit_A, modes, outputs, X0, CN, OMEGA, GAMMA, GAMMA1, POLES

TMAX = 12.0
SY = 0.85                       # page cm per unit of signal in the three ingredients


def icon(cx, cy, active):
    """A tiny lambda-plane with all poles in grey and the active ones in red."""
    sx, sy = 1.3, 0.16            # cm per unit of Re, Im
    out = ["\\draw[faint, ->, >=stealth] (%.3f,%.3f) -- (%.3f,%.3f);" % (cx - 0.85, cy, cx + 0.3, cy),
           "\\draw[faint, ->, >=stealth] (%.3f,%.3f) -- (%.3f,%.3f);" % (cx, cy - 0.62, cx, cy + 0.62)]
    for p in POLES:
        on = any(abs(p - a) < 1e-9 for a in active)
        out.append("\\fill[%s] (%.3f,%.3f) circle (%s);"
                   % ("ndRed" if on else "ndInk!30", cx + sx * p.real, cy + sy * p.imag, "1.5pt" if on else "1.1pt"))
    return out


def main():
    A = qubit_A()
    lam, res = modes(A, CN, X0)
    t = np.linspace(0, TMAX, 400)
    y = outputs(A, CN, X0, t)[:, 0]
    groups = [("constant", [0.0], "ndGreen"),
              ("decay", [-GAMMA1], "ndGold"),
              ("oscillation", [-GAMMA + 1j * OMEGA, -GAMMA - 1j * OMEGA], "ndBlue")]
    parts = []
    for _, poles, _ in groups:
        idx = [int(np.argmin(abs(lam - p))) for p in poles]
        parts.append(np.real(sum(res[i] * np.exp(lam[i] * t) for i in idx)))
    assert np.allclose(sum(parts), y, atol=1e-10)
    # the three residues, as printed in the figure
    c0 = parts[0][0]
    c1 = parts[1][0] - 0.0
    amp = 2 * abs(res[int(np.argmin(abs(lam - (-GAMMA + 1j * OMEGA))))])
    assert np.isclose(c0, 1 / np.sqrt(2)) and np.isclose(c1, (X0[2] - 1) / np.sqrt(2))
    assert np.isclose(amp, X0[0] / np.sqrt(2))
    assert np.allclose(parts[1], c1 * np.exp(-GAMMA1 * t)) and np.allclose(parts[2], amp * np.exp(-GAMMA * t) * np.cos(OMEGA * t))

    b = []
    # ---- the three ingredients, stacked
    x0p, wp = 2.0, 4.6
    ylims = [(-0.25, 0.95), (-1.2, 0.2), (-0.72, 0.72)]
    tops = [6.0, 3.95, 1.75]
    labels = ["$%.2f$" % c0,
              "$%.2f\\,\\ee^{-%.1ft}$" % (c1, GAMMA1),
              "$%.2f\\,\\ee^{-%.1ft}\\cos %gt$" % (amp, GAMMA, OMEGA)]
    names = ["$\\lambda=0$", "$\\lambda=-\\gamma_1$", "$\\lambda=-\\Gamma\\pm\\imath\\omega$"]
    words = ["steady state", "relaxation", "precession"]
    plots = []
    for k, ((name, poles, col), part, (lo, hi)) in enumerate(zip(groups, parts, ylims)):
        h = SY * (hi - lo)
        Pl = N.Plot((0, TMAX), (lo, hi), wp, h, origin=(x0p, tops[k] - h))
        plots.append(Pl)
        b.append("\\draw[faint] %s -- %s;" % (Pl.pt(0, 0), Pl.pt(TMAX, 0)))
        b.append("\\draw[faint] %s -- %s;" % (Pl.pt(0, lo), Pl.pt(0, hi)))
        b.append(Pl.curve(t, part, "line width=1.3pt, %s" % col))
        b += icon(0.95, tops[k] - h / 2 + 0.25, poles)
        b.append("\\node[font=\\scriptsize, text=ndRed, anchor=north, align=center] at (0.75,%.3f) {%s\\\\[-1pt] \\textit{\\color{ndInk!85}%s}};" % (tops[k] - h / 2 - 0.4, names[k], words[k]))
    # formulas next to the curves
    b.append("\\node[font=\\small, text=ndGreen, anchor=south] at %s {%s};" % (plots[0].pt(TMAX * 0.5, c0 + 0.02), labels[0]))
    b.append("\\node[font=\\small, text=ndGold, anchor=north west] at %s {%s};" % (plots[1].pt(TMAX * 0.55, -0.62), labels[1]))
    b.append("\\node[font=\\small, text=ndBlue, anchor=south west] at %s {%s};" % (plots[2].pt(TMAX * 0.42, 0.12), labels[2]))
    # plus signs between the ingredients
    for k in (0, 1):
        ymid = 0.5 * (plots[k].P(0, ylims[k][0])[1] + plots[k + 1].P(0, ylims[k + 1][1])[1])
        b.append("\\node[font=\\Large] at (%.3f,%.3f) {$+$};" % (x0p + wp / 2, ymid))
    # ---- the total
    lo, hi = -0.7, 0.9
    Q = N.Plot((0, TMAX), (lo, hi), 6.0, 3.6, origin=(9.0, 0.95))
    b.append("\\draw[faint] %s -- %s;" % (Q.pt(0, 0), Q.pt(TMAX, 0)))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk] {$t$};" % (Q.pt(0, 0), Q.pt(TMAX + 0.5, 0)))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (Q.pt(0, lo), Q.pt(0, hi + 0.08)))
    b.append("\\draw[dashedink] %s -- %s;" % (Q.pt(0, c0), Q.pt(TMAX, c0)))
    b.append("\\node[font=\\scriptsize, anchor=east] at %s {$%.2f$};" % (Q.pt(0, c0), c0))
    b.append(Q.curve(t, y, "line width=1.0pt, ndInk"))
    ts = np.arange(0, TMAX + 1e-9, 0.5)
    for s in ts:
        b.append("\\fill[ndRed] %s circle (1.6pt);" % Q.pt(s, outputs(A, CN, X0, [s])[0, 0]))
    b.append("\\node[font=\\small, anchor=south] at %s {$y(t)=\\langle\\sigma_n\\rangle(t)$};" % Q.pt(TMAX / 2, hi - 0.05))
    b.append("\\node[note, anchor=north] (d) at %s {the data: the samples\\\\ recorded (red)};" % Q.pt(TMAX * 0.62, -0.32))
    b.append("\\node[font=\\Large] at (7.75,%.3f) {$=$};" % Q.P(0, 0.1)[1])
    # ---- mixing and un-mixing
    b.append("\\draw[pointer, line width=0.7pt, ndInk!75] (6.9,5.6) to[bend left=18] node[note, above, sloped] {the terms are added} %s;" % Q.pt(1.0, hi + 0.55))
    b.append("\\draw[pointer, line width=0.7pt, ndRed] %s to[bend left=22] node[note, below, text=ndRed] {identification: recover\\\\ each $\\lambda_i$ and $r_i$ from the sum} (6.9,0.5);"
             % Q.pt(TMAX * 0.15, lo - 0.1))
    N.write("ch05_modes", b)
    print("written figures/ch05_modes.tex; weights: constant %.4f, decay %.4f, oscillation %.4f" % (c0, c1, amp))


if __name__ == "__main__":
    main()
