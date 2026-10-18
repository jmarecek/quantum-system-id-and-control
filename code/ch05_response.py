"""ch05_response.py -- frequency response: sine in, sine out.

A weak transverse drive u(t) sigma_x / 2 on the damped qubit of the chapter,
around its ground state x_ss = (0, 0, 1, 1), enters linearly:
    d/du [A(u) x_ss] = b = (0, -1, 0, 0),
so that  dx' = A dx + b u,  y = <sigma_x> = C_x dx,  with the transfer function
    G(s) = C_x (s - A)^{-1} b = omega / ((s + Gamma)^2 + omega^2).
A sinusoidal input u = eps sin(nu t) gives, after the transient, the output
    y(t) = eps |G(i nu)| sin(nu t + arg G(i nu)):
the same frequency, scaled by the gain and shifted by the phase.
  (a) input and output at nu = 2.6, gain and lag marked;
  (b) the gain |G(i nu)| against nu: a resonance at omega, whose half-power
      points (|G|^2 at half its peak) are about 2 Gamma apart.
Writes figures/ch05_response.tex.
"""
import numpy as np
from scipy.integrate import solve_ivp
import needham as N
from ch05_spectrum import qubit_A, CX, OMEGA, GAMMA

B = np.array([0.0, -1.0, 0.0, 0.0])
XSS = np.array([0.0, 0.0, 1.0, 1.0])
NU = 2.6


def G(s):
    """C_x (s - A)^{-1} b on the Bloch components (the affine constant is not driven)."""
    A = qubit_A()[:3, :3]
    return (CX[:, :3] @ np.linalg.solve(s * np.eye(3) - A, B[:3])).item()


def checks():
    A = qubit_A()
    assert np.allclose(A @ XSS, 0)                                   # ground state is stationary
    h = 1e-6
    assert np.allclose((qubit_A(u=h) @ XSS - qubit_A(u=-h) @ XSS) / (2 * h), B)
    for s in [0.3j, 2.0 + 1j, 3j]:
        assert np.isclose(G(s), OMEGA / ((s + GAMMA) ** 2 + OMEGA ** 2))
    # a weak drive: the full (bilinear) qubit against the linear prediction
    eps = 0.01
    f = lambda t, x: qubit_A(u=eps * np.sin(NU * t)) @ x
    t = np.linspace(30, 36, 200)                                     # after the transient
    sol = solve_ivp(f, (0, 36), XSS, t_eval=t, rtol=1e-10, atol=1e-12)
    g = G(1j * NU)
    lin = eps * abs(g) * np.sin(NU * t + np.angle(g))
    assert np.max(abs(sol.y[0] - lin)) < 1e-2 * eps
    return g


def main():
    g = checks()
    gain, phase = abs(g), np.angle(g)
    nus = np.linspace(0, 6.2, 1241)
    mag = np.array([abs(G(1j * v)) for v in nus])
    peak = mag.max()
    # half-power points of |G|^2
    half = mag ** 2 >= peak ** 2 / 2
    lo_nu, hi_nu = nus[half][0], nus[half][-1]
    assert abs(nus[np.argmax(mag)] - OMEGA) < 0.05
    assert abs((hi_nu - lo_nu) - 2 * GAMMA) < 0.05 and lo_nu < OMEGA < hi_nu    # full width ~ 2 Gamma
    print("gain %.3f, phase %.1f deg, peak %.3f at %.2f, half-power %.2f..%.2f"
          % (gain, np.degrees(phase), peak, nus[np.argmax(mag)], lo_nu, hi_nu))

    b = []
    # ---- (a) input and output, after the transient
    # an input peak and the next output peak, in the third period
    tu = (4 * np.pi + np.pi / 2) / NU
    ty = (4 * np.pi + np.pi / 2 - phase) / NU
    assert np.isclose(np.sin(NU * ty + phase), 1) and ty > tu
    tend = ty + 0.12
    tmax = tend + 2.3
    Pl = N.Plot((0, tmax), (-1.25, 1.25), 7.4, 4.2, origin=(0.0, 0.6))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk] {$t$};" % (Pl.pt(0, 0), Pl.pt(tmax + 0.3, 0)))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (Pl.pt(0, -1.25), Pl.pt(0, 1.4)))
    for v, lab in [(1, "$\\epsilon$"), (-1, "$-\\epsilon$")]:
        b.append("\\draw[faint] %s -- ++(-0.08,0) node[left, font=\\scriptsize, text=ndInk] {%s};" % (Pl.pt(0, v), lab))
    t = np.linspace(0, tend, 400)
    u = np.sin(NU * t)
    y = gain * np.sin(NU * t + phase)
    b.append(Pl.curve(t, u, "line width=0.9pt, ndGold"))
    b.append(Pl.curve(t, y, "line width=1.5pt, ndBlue"))
    b.append("\\draw[dashedink] %s -- %s;" % (Pl.pt(tu, 0), Pl.pt(tu, 1.0)))
    b.append("\\draw[dashedink] %s -- %s;" % (Pl.pt(ty, 0), Pl.pt(ty, gain)))
    # lag bracket (horizontal, above the peaks)
    yb = 1.13
    b.append("\\draw[line width=0.7pt, ndRed, |-|] %s -- %s;" % (Pl.pt(tu, yb), Pl.pt(ty, yb)))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=south west, align=left] at %s {phase lag $%.0f^\\circ$};" % (Pl.pt(tu - 0.05, yb + 0.02), -np.degrees(phase)))
    # gain bracket (vertical, beside the output peak), label in the free space to the right
    xb = ty + 0.42
    b.append("\\draw[line width=0.7pt, ndRed, |-|] %s -- %s;" % (Pl.pt(xb, 0), Pl.pt(xb, gain)))
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(ty, gain), Pl.pt(xb, gain)))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west, align=left] at %s {gain\\\\ $\\abs{\\TF(\\imath\\nu)}=%.2f$:\\\\ ratio of output\\\\ to input\\\\ amplitude};" % (Pl.pt(xb + 0.08, 0.5 * gain), gain))
    b.append("\\node[font=\\small, text=ndGold, anchor=north] at %s {input $u=\\epsilon\\sin\\nu t$};" % Pl.pt(tend + 1.05, -0.35))
    b.append("\\node[font=\\small, text=ndBlue, anchor=north west] at %s {output $\\langle\\sigma_x\\rangle$};" % Pl.pt(0.15, -0.95))
    b.append("\\node[note, anchor=north] at %s {the same frequency $\\nu$; only the\\\\ amplitude and phase change};" % Pl.pt(tmax / 2, -1.45))
    b.append("\\node[font=\\small] at %s {(a) a weak drive at $\\nu=%.1f$, after the transient};" % (Pl.pt(tmax / 2, -2.3), NU))
    # ---- (b) the gain against nu
    Q = N.Plot((0, 6.2), (0, 1.15), 6.6, 4.2, origin=(9.0, 0.6))
    b += Q.axes("$\\nu$", "$\\abs{\\TF(\\imath\\nu)}$",
                xticks=[(OMEGA - GAMMA, ""), (OMEGA, "$\\omega$"), (OMEGA + GAMMA, ""), (NU, "")],
                yticks=[(peak, "$%.1f$" % peak)])
    b.append(Q.curve(nus, mag, "line width=1.5pt, ndBlue"))
    hp = peak / np.sqrt(2)
    b.append("\\draw[line width=0.7pt, ndRed, |-|] %s -- %s;" % (Q.pt(lo_nu, hp), Q.pt(hi_nu, hp)))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=west] at %s {$\\approx2\\Gamma$};" % Q.pt(hi_nu + 0.05, hp))
    b.append("\\draw[dashedink] %s -- %s;" % (Q.pt(OMEGA, 0), Q.pt(OMEGA, peak)))
    b.append("\\draw[dashedink] %s -- %s;" % (Q.pt(NU, 0), Q.pt(NU, gain)))
    b.append("\\node[dot, fill=ndRed] at %s {};" % Q.pt(NU, gain))
    b.append("\\node[font=\\scriptsize, anchor=north] at %s {$\\nu$};" % Q.pt(NU - 0.05, -0.02))
    b.append("\\node[note, anchor=east] (r) at %s {resonance at the\\\\ frequency $\\omega$\\\\ of the pole};" % Q.pt(1.9, 0.82))
    b.append("\\draw[pointer] (r.east) to[bend left=15] %s;" % Q.pt(OMEGA - 0.08, peak + 0.02))
    b.append("\\node[note, anchor=west] (w) at %s {width governed by\\\\ the decay rate $\\Gamma$:\\\\ $\\abs{\\TF}^2$ falls to half\\\\ about $\\Gamma$ from the peak};" % Q.pt(4.0, 0.62))
    b.append("\\node[note, anchor=west] (d) at %s {the frequency of (a)};" % Q.pt(0.35, 0.14))
    b.append("\\draw[pointer] (d.east) to[bend right=25] %s;" % Q.pt(NU - 0.04, gain - 0.04))
    b.append("\\node[font=\\small] at %s {(b) the gain against the driving frequency};" % Q.pt(3.1, -0.33))
    N.write("ch05_response", b)
    print("written figures/ch05_response.tex")


if __name__ == "__main__":
    main()
