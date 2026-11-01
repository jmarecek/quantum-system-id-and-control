"""ch11_aliases.py -- aliases as the atoms of a moment relaxation.

Requires pysparsepmi (https://github.com/jmarecek/pysparsepmi), with cvxpy and
the SCS solver.  If the package is not installed, set PYSPARSEPMI_PATH to a
clone of the repository.

The damped qubit of Chapter 5 (omega = 3, Gamma = 1/T2 = 0.5): its complex
coherence c = <X> + i<Y> decays as e^{lambda t}, lambda = -Gamma + i omega, so
samples taken every tau determine only Lambda = e^{lambda tau}, the ratio of two
successive samples.  With the step h = 0.4 and the unknown z = e^{lambda h} =
u + iv, the samples at the rate tau_1 = 4h = 1.6 of Chapter 5 say z^4 = Lambda_1,
and those at tau_2 = 5h = 2.0 say z^5 = Lambda_2.  The only prior is that the
mode decays (|z| <= 1) and that |omega| < pi/h = 7.85, the band in which z
determines lambda.  We minimise the polynomial misfit

    f(u, v) = |z^4 - Lambda_1|^2,   |z^5 - Lambda_2|^2,   or their sum,

over the unit disc with pysparsepmi.sos_lower_bound (relaxation order 4, 5, 5),
and read off the global minimisers as the atoms of the flat pseudo-moment
matrix (Henrion and Lasserre), each atom giving lambda = log(z)/h.  All three
bounds vanish (|value| < 1e-8, SCS status "optimal").

  tau_1 = 1.6: the moment matrix has rank 4 and four atoms,
               omega = -4.85, -0.93, 3, 6.93 (omega + 2 pi k/1.6), all Gamma = 0.5;
  tau_2 = 2.0: rank 5, five atoms, omega = -6.42, -3.28, -0.14, 3, 6.14;
  both:        rank 1, one atom, omega = 3.000, Gamma = 0.500.

Every atom is checked against the closed form Lambda^{1/n} e^{2 pi i k/n} and
against numpy.roots to 1e-6, the objective is checked to vanish there, and the
ratio Lambda is checked against the simulated samples of the qubit.  The two
combs of aliases, of spacings 2 pi/1.6 and 2 pi/2.0, meet again only after
2 pi/h = 15.7, the width of the band, so that two rates in the ratio 5:4
leave a single atom in it.
(a) The z-plane: the four atoms of tau_1 (blue circles), the five of tau_2
    (gold squares), all on the circle |z| = e^{-Gamma h} = 0.819, and the one
    atom of the two rates together (red).
(b) The same atoms in the (omega, Gamma) plane: aliasing moves the frequency
    and leaves the decay rate alone.
(c) The singular values of the three moment matrices: the drop after the 4th,
    5th and 1st value is the certificate of flatness and counts the atoms.
Writes figures/ch11_aliases.tex.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm

try:
    import pysparsepmi as psp
except ImportError:
    sys.path.insert(0, os.environ.get("PYSPARSEPMI_PATH", ""))
    import pysparsepmi as psp

import needham as N
import ch05_spectrum as S                    # OMEGA = 3, GAMMA = 0.5, qubit_A, X0

H = 0.4
RATES = {4: 1.6, 5: 2.0}                     # n: tau = n h
LAM = -S.GAMMA + 1j * S.OMEGA
U, V = psp.polyvar(2)


def ratio(n):
    """Lambda = c(tau)/c(0) from two simulated samples of the qubit, tau = n h."""
    A = S.qubit_A()
    x0, x1 = S.X0, expm(A * n * H) @ S.X0
    return (x1[0] + 1j * x1[1]) / (x0[0] + 1j * x0[1])


def power(n):
    """Real and imaginary parts of z^n, z = u + iv, as polynomials."""
    re, im = U, V
    for _ in range(n - 1):
        re, im = re * U - im * V, re * V + im * U
    return re, im


def misfit(ns):
    f = psp.Polynomial.zero(2)
    for n in ns:
        re, im = power(n)
        lam = ratio(n)
        f = f + (re - lam.real) ** 2 + (im - lam.imag) ** 2
    return f


def solve(ns):
    """Certified bound, atoms (as complex z) and singular values of the moment matrix."""
    order = max(ns)
    res = psp.sos_lower_bound(misfit(ns), ineqs=[1 - U * U - V * V], order=order,
                              solver="SCS", eps=1e-9, max_iters=200000)
    at = res.atoms()
    z = at[:, 0] + 1j * at[:, 1]
    z = z[np.argsort(np.angle(z))]
    sv = np.linalg.svd(res.moment_matrix(order=order), compute_uv=False)
    return res, z, sv


def closed_form(ns):
    """The roots common to z^n = Lambda_n, sorted by angle."""
    sets = []
    for n in ns:
        lam = ratio(n)
        sets.append(np.abs(lam) ** (1 / n) * np.exp(1j * (np.angle(lam) + 2 * np.pi * np.arange(n)) / n))
    common = [z for z in sets[0] if all(np.min(np.abs(s - z)) < 1e-9 for s in sets[1:])]
    return np.array(sorted(common, key=np.angle))


def to_lambda(z):
    return np.log(z) / H                     # principal branch: |omega| < pi/h


def checks(sol):
    for n, tau in RATES.items():
        assert abs(n * H - tau) < 1e-12
        assert abs(ratio(n) - np.exp(LAM * tau)) < 1e-12           # the samples see e^{lambda tau}
        assert np.min(np.abs(np.linalg.eigvals(expm(S.qubit_A() * tau)) - np.exp(LAM * tau))) < 1e-12
    assert S.OMEGA < np.pi / H
    s1, s2 = 2 * np.pi / RATES[4], 2 * np.pi / RATES[5]                # the combs meet first at 2 pi/h
    assert abs(4 * s1 - 2 * np.pi / H) < 1e-12 and abs(5 * s2 - 2 * np.pi / H) < 1e-12
    assert all(abs(a * s1 - c * s2) > 0.1 for a in range(1, 4) for c in range(1, 5))
    for ns, (res, z, sv) in sol.items():
        cf = closed_form(ns)
        assert res.status == "optimal" and abs(res.value) < 1e-8, (ns, res.status, res.value)
        assert len(z) == len(cf) and np.abs(z - cf).max() < 1e-6, (ns, z, cf)
        assert np.sum(sv > 1e-6 * sv[0]) == len(z)                  # the rank counts the atoms
        f = misfit(ns)
        assert max(abs(f((w.real, w.imag))) for w in z) < 1e-10      # each atom fits the data
        lam = to_lambda(z)
        assert np.allclose(-lam.real, S.GAMMA, atol=1e-6)            # one decay rate for all
        k = (lam.imag - S.OMEGA) * RATES[ns[0]] / (2 * np.pi)
        assert np.allclose(k, np.round(k), atol=1e-6)                # omega + 2 pi k / tau
        if len(ns) == 1:
            roots = np.roots([1] + [0] * (ns[0] - 1) + [-ratio(ns[0])])
            assert np.abs(np.sort_complex(roots) - np.sort_complex(z)).max() < 1e-6
    lam = to_lambda(sol[(4, 5)][1])
    assert len(lam) == 1 and abs(lam[0] - LAM) < 1e-6


def main():
    sol = {ns: solve(ns) for ns in [(4,), (5,), (4, 5)]}
    checks(sol)
    b = []
    rho = np.exp(-S.GAMMA * H)
    sty = {(4,): ("ndBlue", "circle"), (5,): ("ndGold!85!ndInk", "square")}

    def marker(p, ns, size=1.0):
        col, shape = sty[ns]
        if shape == "circle":
            return "\\draw[%s, line width=0.8pt, fill=white] %s circle (%.2fpt);" % (col, N.pt(p), 2.6 * size)
        s = 2.3 * size / 28.45
        return "\\draw[%s, line width=0.8pt, fill=white] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (
            col, p[0] - s, p[1] - s, p[0] + s, p[1] + s)

    # ---- (a) the z-plane
    c, R = np.array([2.3, 2.75]), 1.95
    Z = lambda z: c + R * np.array([z.real, z.imag])
    b.append("\\fill[ndSky!55] %s circle (%.3f);" % (N.pt(c), R))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk] {\\scriptsize $\\mathrm{Re}\\,z$};" % (N.pt(c - [R + 0.15, 0]), N.pt(c + [R + 0.25, 0])))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, text=ndInk] {\\scriptsize $\\mathrm{Im}\\,z$};" % (N.pt(c - [0, R + 0.15]), N.pt(c + [0, R + 0.25])))
    b.append("\\draw[ink] %s circle (%.3f);" % (N.pt(c), R))
    b.append("\\draw[dashedink] %s circle (%.3f);" % (N.pt(c), R * rho))
    for ns in [(4,), (5,)]:
        for z in sol[ns][1]:
            b.append(marker(Z(z), ns))
    zt = sol[(4, 5)][1][0]
    b.append("\\fill[ndRed] %s circle (1.5pt);" % N.pt(Z(zt)))
    b.append("\\node[note, anchor=north west] (ra) at %s {$\\abs z=\\ee^{-\\Gamma h}$};" % N.pt(c + [1.2, -1.93]))
    b.append("\\draw[pointer] (ra.north) to[bend right=25] %s;" % N.pt(Z(rho * np.exp(-0.62j))))
    b.append("\\node[note, anchor=south, text=ndRed] (tr) at %s {both rates:\\\\ one atom};" % N.pt(c + [-1.75, 1.9]))
    b.append("\\draw[pointer] (tr.east) to[bend right=15] %s;" % N.pt(Z(zt) + np.array([-0.1, 0.05])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) atoms in the $z$-plane};" % N.pt(c - [0, R + 0.4]))

    # ---- (b) the (omega, Gamma) plane
    wmax = np.pi / H
    pb = N.Plot((-wmax, wmax), (0, 1.0), 6.0, 3.2, origin=(5.55, 0.85))
    b.append("\\fill[ndSky!55] %s rectangle %s;" % (pb.pt(-wmax, 0), pb.pt(wmax, 1.0)))
    b += pb.axes(xlabel="{\\scriptsize $\\omega$}", ylabel="{\\scriptsize $\\Gamma$}",
                 xticks=[(-wmax, "$-\\pi/h$"), (0, "$0$"), (S.OMEGA, "$3$"), (wmax, "$\\pi/h$")],
                 yticks=[(S.GAMMA, "$0.5$"), (1.0, "$1$")])
    b.append("\\draw[faint] %s -- %s;" % (pb.pt(-wmax, S.GAMMA), pb.pt(wmax, S.GAMMA)))
    for ns in [(4,), (5,)]:
        for l in to_lambda(sol[ns][1]):
            b.append(marker(pb.P(l.imag, -l.real), ns))
    l = to_lambda(zt)
    b.append("\\fill[ndRed] %s circle (1.5pt);" % pb.pt(l.imag, -l.real))
    lx = pb.P(-wmax, 0.32)
    b.append(marker(lx + np.array([0.35, 0]), (4,)))
    b.append("\\node[font=\\scriptsize, anchor=west] at %s {$z^4=\\Lambda_1$, $\\tau_1=1.6$: four atoms};" % N.pt(lx + np.array([0.5, 0])))
    b.append(marker(lx + np.array([0.35, -0.38]), (5,)))
    b.append("\\node[font=\\scriptsize, anchor=west] at %s {$z^5=\\Lambda_2$, $\\tau_2=2.0$: five atoms};" % N.pt(lx + np.array([0.5, -0.38])))
    b.append("\\fill[ndRed] %s circle (1.5pt);" % N.pt(lx + np.array([0.35, -0.76])))
    b.append("\\node[font=\\scriptsize, anchor=west] at %s {both: one atom};" % N.pt(lx + np.array([0.5, -0.76])))
    w1 = to_lambda(sol[(4,)][1]).imag
    i0 = int(np.argmin(np.abs(w1 - S.OMEGA)))
    ya = pb.P(0, S.GAMMA)[1] + 0.45
    b.append("\\draw[<->, >=stealth, line width=0.4pt, ndBlue] (%.3f,%.3f) -- (%.3f,%.3f) node[midway, above, font=\\scriptsize] {$2\\pi/\\tau_1$};"
             % (pb.P(w1[i0 - 1], 0)[0], ya, pb.P(w1[i0], 0)[0], ya))
    b.append("\\node[note, anchor=north] at %s {the rate of decay\\\\ is never in doubt};" % N.pt(pb.P(5.3, 0.97)))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the same atoms as $(\\omega,\\Gamma)$};" % N.pt(pb.P(0, 0) + np.array([0, -0.55])))

    # ---- (c) the singular values of the moment matrices
    kmax = 9
    pc = N.Plot((0.5, kmax + 0.5), (-14, 1), 3.6, 3.6, origin=(13.0, 0.85))
    b += pc.axes(xlabel="{\\scriptsize $i$}", ylabel="{\\scriptsize $\\log_{10}\\sigma_i$}",
                 xticks=[(1, "$1$"), (4, "$4$"), (5, "$5$"), (9, "$9$")], yticks=[(0, "$0$"), (-6, "$-6$"), (-12, "$-12$")])
    for ns, col in [((4,), "ndBlue"), ((5,), "ndGold!85!ndInk"), ((4, 5), "ndRed")]:
        sv = np.log10(np.maximum(sol[ns][2][:kmax], 1e-14))
        b.append(pc.curve(np.arange(1, kmax + 1), sv, "line width=0.7pt, %s" % col))
        for i, s in enumerate(sv):
            b.append("\\fill[%s] %s circle (1.1pt);" % (col, pc.pt(i + 1, s)))
    b.append("\\node[note, anchor=west] at %s {rank $=$\\\\ number\\\\ of atoms};" % pc.pt(5.6, -4.0))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) flat moment matrices};" % N.pt(pc.P(5, -14) + np.array([0, -0.55])))
    N.write("ch11_aliases", b)
    return sol


if __name__ == "__main__":
    import time
    t0 = time.time()
    sol = main()
    for ns, (res, z, sv) in sol.items():
        lam = to_lambda(z)
        print("rates %s: %s, bound %.1e, rank %d, omega = %s, Gamma = %s"
              % ([RATES[n] for n in ns], res.status, res.value, len(z),
                 np.array2string(lam.imag, precision=3), np.array2string(-lam.real, precision=3)))
    print("written figures/ch11_aliases.tex (%.1f s)" % (time.time() - t0))
