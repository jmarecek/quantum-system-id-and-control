"""ch06_summary.py -- the chapter in nine pictures.

A dictionary of computed thumbnails, one for each law of the chapter, with the
law in words above and its formula below:
  1. a shot is a coin toss: running means of outcomes +-1 with mu = 0.4
     (ch06_coins), inside the band mu +- 2 sqrt(1 - mu^2)/sqrt(N);
  2. many observables at once: Hoeffding's inequality with the union bound,
     N = (2/eps^2) ln(2M/delta) shots for each of M expectation values
     (eps = 0.1, delta = 0.05), which grows like log M;
  3. when to measure: the Fisher information about omega of one shot of
     sigma_x of the qubit of Chapter 5 (ch06_fisher), against t, under the
     envelope a^2 t^2 e^{-2 Gamma t}, which is greatest at t = 1/Gamma;
  4. weak modes cost more: the shots per time needed for a standard deviation
     of 0.01 in omega (ch06_cost) against the angle phi of the preparation,
     and the curve c / sin^2 phi;
  5. time as a resource: the error in omega against the total time T, 1/T for
     coherent evolution, and the Cramer--Rao bound of the open system
     (ch06_heisenberg), which returns to T^{-1/2} beyond T = 1/(2 Gamma);
  6. classical shadows: the copies needed for all Pauli observables of weight
     at most 2 on q qubits (ch06_shadows), with shadows and with each
     observable measured on copies of its own;
  7. Gibbs states: the copies needed to learn omega to 0.01 from copies of the
     Gibbs state of H = -(omega/2) sigma_z, omega = 3, against the inverse
     temperature beta, from the Fisher information
         F = beta^2 Var(dE/d omega) = (beta^2/4) sech^2(beta omega/2)
     of the populations; it grows like 1/beta^2 when hot and like
     e^{beta Delta}/beta^2 when cold, Delta = omega the gap;
  8. Ho--Kalman on shots: the singular values of the Hankel matrix of the
     example (ch06_hankel_noise) against the noise floor c/sqrt(N), estimated
     here by Monte Carlo; the weakest mode is seen once the floor falls below
     sigma_4, at N* = (c/sigma_4)^2;
  9. the whole budget: the error of each stage of the pipeline (ch06_budget),
     for N = 10^4, multiplied by sqrt(RSp), 1/sigma_4, kappa(g)/tau and
     kappa(M) in turn.
Writes figures/ch06_summary.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N
from ch05_spectrum import qubit_A, hankel, X0, CN, TAU, GAMMA, OMEGA
from ch06_coins import running_means, MU, SIG, NMAX
from ch06_cost import shots_needed
from ch06_heisenberg import open_bound
from ch06_shadows import n_obs, EPS as EPS_SH, DELTA as DELTA_SH, WMAX
from ch06_hankel_noise import data, svals, floor
from ch06_budget import reconstruction_map, K, R, S, P

W, H = 4.25, 1.3                     # thumbnail size (cm)
COLW, ROWH = 5.7, 3.1               # cell spacing (cm)
BLUE, RED, GOLD, GREEN = ("line width=1.0pt, %s" % c for c in ("ndBlue", "ndRed", "ndGold", "ndGreen"))


class Thumb:
    def __init__(self, cell, xlim, ylim):
        i, j = cell
        self.x0, self.y0 = j * COLW, -i * ROWH
        self.P = N.Plot(xlim, ylim, W, H, origin=(self.x0 + 0.3, self.y0 - 0.35 - H))
        self.xlim, self.ylim = xlim, ylim

    def frame(self, b, n, law, formula, xlab, ylab, xticks=(), yticks=()):
        P, (x0, x1), (y0, y1) = self.P, self.xlim, self.ylim
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk, inner sep=1.5pt] {%s};"
                 % (P.pt(x0, y0), P.pt(x1 + 0.04 * (x1 - x0), y0), xlab))
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above right, font=\\scriptsize, text=ndInk, inner sep=1pt] {%s};"
                 % (P.pt(x0, y0), P.pt(x0, y1 + 0.08 * (y1 - y0)), ylab))
        for v, lab in xticks:
            p = P.P(v, y0)
            b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[below, font=\\scriptsize, text=ndInk, inner sep=1pt] {%s};"
                     % (p[0], p[1] + 0.05, p[0], p[1] - 0.05, lab))
        for v, lab in yticks:
            p = P.P(x0, v)
            b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[left, font=\\scriptsize, text=ndInk, inner sep=1pt] {%s};"
                     % (p[0] + 0.05, p[1], p[0] - 0.05, p[1], lab))
        yl = self.y0 - 0.35 - H - 0.4
        b.append("\\node[font=\\small, anchor=north] at (%.3f,%.3f) {\\textbf{%d}\\enspace %s};" % (self.x0 + 0.3 + W / 2, yl, n, law))
        b.append("\\node[font=\\footnotesize, anchor=north] at (%.3f,%.3f) {%s};" % (self.x0 + 0.3 + W / 2, yl - 0.42, formula))

    def curve(self, b, x, y, style):
        y = np.clip(y, self.ylim[0], self.ylim[1])
        b.append(self.P.curve(x, y, style))

    def note(self, b, x, y, text, anchor="west", col="ndInk"):
        b.append("\\node[font=\\scriptsize, text=%s, anchor=%s, inner sep=1pt] at %s {%s};" % (col, anchor, self.P.pt(x, y), text))


def main():
    b = []
    rng = np.random.default_rng(7)

    # ---- 1. a shot is a coin toss
    T = Thumb((0, 0), (0, np.log10(NMAX)), (-1.05, 1.05))
    rm = running_means(rng, paths=3)
    ns = np.unique(np.round(np.logspace(0, np.log10(NMAX), 200)).astype(int))
    lx = np.log10(ns)
    band = 2 * SIG / np.sqrt(ns)
    lo, hi = np.clip(MU - band, -1.05, 1.05), np.clip(MU + band, -1.05, 1.05)
    b.append("\\fill[ndSky, fill opacity=0.7] %s;" % " -- ".join([T.P.pt(x, v) for x, v in zip(lx, hi)] + [T.P.pt(x, v) for x, v in zip(lx[::-1], lo[::-1])]) + "")
    T.curve(b, [0, lx[-1]], [MU, MU], "dashedink")
    for path, st in zip(rm, (BLUE, RED, GOLD)):
        T.curve(b, lx, path[ns - 1], st.replace("1.0pt", "0.7pt"))
    assert np.all(np.abs(rm[:, -1] - MU) < 3 * SIG / np.sqrt(NMAX))
    T.frame(b, 1, "a shot is a coin toss", "error $\\sqrt{(1-\\mu^2)/N}$", "$N$", "mean",
            xticks=[(0, "$1$"), (1, "$10$"), (2, "$10^2$"), (3, "$10^3$")])

    # ---- 2. many observables at once
    eps, delta = 0.1, 0.05
    lm = np.linspace(0, 6, 61)
    nM = 2 / eps ** 2 * np.log(2 * 10 ** lm / delta)
    T = Thumb((0, 1), (0, 6), (-700, 4000))
    T.curve(b, lm, nM, BLUE)
    assert abs(nM[0] - 738) < 1 and abs(nM[-1] - 3501) < 1
    for x, v in [(0, nM[0]), (6, nM[-1])]:
        b.append("\\node[dot, fill=ndBlue] at %s {};" % T.P.pt(x, v))
    T.note(b, 0.2, nM[0] - 80, "$%d$" % round(nM[0]), anchor="north west", col="ndBlue")
    T.note(b, 5.8, nM[-1] + 80, "$%d$" % round(nM[-1]), anchor="south east", col="ndBlue")
    T.note(b, 2.6, 700, "$\\eps=0.1$, $\\delta=0.05$", anchor="west")
    T.frame(b, 2, "many observables at once", "$N=\\frac{2}{\\eps^2}\\ln\\frac{2M}{\\delta}$ for each", "$M$", "$N$",
            xticks=[(0, "$1$"), (3, "$10^3$"), (6, "$10^6$")])

    # ---- 3. when to measure
    a = X0[0]
    t = np.linspace(1e-3, 8, 1601)
    y = a * np.exp(-GAMMA * t) * np.cos(OMEGA * t)
    I_w = (a * t * np.exp(-GAMMA * t) * np.sin(OMEGA * t)) ** 2 / (1 - y ** 2)
    env = a ** 2 * t ** 2 * np.exp(-2 * GAMMA * t)
    s = I_w.max()
    assert abs(t[np.argmax(env)] - 1 / GAMMA) < 0.01
    T = Thumb((0, 2), (0, 8), (0, 1.1))
    T.curve(b, t, env / s, "dashedink")
    T.curve(b, t, I_w / s, BLUE)
    T.note(b, 4.6, 0.75, "information about $\\omega$", col="ndBlue")
    T.frame(b, 3, "when to measure", "$(\\partial_\\theta\\mu)^2/(1-\\mu^2)$ per shot", "$t$", "",
            xticks=[(1 / GAMMA, "$1/\\Gamma$"), (8, "$8$")])

    # ---- 4. weak modes cost more
    degs = np.array([3, 5, 8, 12, 18, 27, 40, 60, 90.0])
    need = np.array([shots_needed(np.radians(d)) for d in degs])
    c = np.median(need * np.sin(np.radians(degs)) ** 2)
    assert np.all(np.abs(np.log(need * np.sin(np.radians(degs)) ** 2 / c)) < 0.35)
    T = Thumb((1, 0), (0, 90), (3, 7))
    dd = np.linspace(2.0, 90, 300)
    T.curve(b, dd, np.log10(c / np.sin(np.radians(dd)) ** 2), "dashedink")
    b += ["\\node[dot, fill=ndRed] at %s {};" % T.P.pt(d, np.log10(v)) for d, v in zip(degs, need)]
    T.note(b, 30, 5.6, "not excited at $\\phi=0$", col="ndRed")
    T.frame(b, 4, "weak modes cost more", "shots $\\propto1/\\abs{r_i}^2\\propto1/\\sin^2\\phi$", "$\\phi$", "shots",
            xticks=[(0, "$0^\\circ$"), (45, "$45^\\circ$"), (90, "$90^\\circ$")],
            yticks=[(3, "$10^3$"), (5, "$10^5$"), (7, "$10^7$")])

    # ---- 5. time as a resource
    lT = np.linspace(-1, 4, 200)
    Tt = 10 ** lT
    ob = open_bound(Tt)
    assert np.allclose(ob[Tt < 0.3] * Tt[Tt < 0.3], 1, atol=0.2)           # 1/T while short
    late = Tt > 100
    assert np.allclose(ob[late], np.sqrt(2 * GAMMA * np.e / Tt[late]), rtol=1e-6)
    T = Thumb((1, 1), (-1, 4), (-4, 1))
    T.curve(b, lT, -lT, "line width=0.8pt, ndBlue, dash pattern=on 2.5pt off 2pt")
    T.curve(b, lT, np.log10(ob), RED)
    b.append("\\draw[faint] %s -- %s;" % (T.P.pt(np.log10(1 / (2 * GAMMA)), -4), T.P.pt(np.log10(1 / (2 * GAMMA)), 1)))
    T.note(b, -0.9, -3.35, "coherent: $1/T$", anchor="west", col="ndBlue")
    T.note(b, 2.0, -0.25, "with decoherence", col="ndRed")
    T.frame(b, 5, "time as a resource", "error $\\propto1/T$ while $T<1/(2\\Gamma)$", "$T$", "error",
            xticks=[(np.log10(1 / (2 * GAMMA)), "$1/(2\\Gamma)$"), (4, "$10^4$")])

    # ---- 6. classical shadows
    qs = np.arange(2, 51)
    M = np.array([n_obs(q) for q in qs], dtype=float)
    shadows = 3.0 ** WMAX * 2 * np.log(2 * M / DELTA_SH) / EPS_SH ** 2
    separate = M * 2 * np.log(2 * M / DELTA_SH) / EPS_SH ** 2
    assert shadows[-1] / shadows[0] < 2.5 and separate[-1] / separate[0] > 500
    T = Thumb((1, 2), (0, 52), (4, 9))
    T.curve(b, qs, np.log10(shadows), GREEN)
    T.curve(b, qs, np.log10(separate), RED)
    T.note(b, 20, np.log10(shadows[18]) + 0.45, "shadows", col="ndGreen")
    T.note(b, 3, 8.75, "each on copies of its own", anchor="west", col="ndRed")
    T.frame(b, 6, "classical shadows", "variance $3^w$, whatever $q$", "$q$", "copies",
            xticks=[(2, "$2$"), (50, "$50$")], yticks=[(5, "$10^5$"), (8, "$10^8$")])

    # ---- 7. Gibbs states
    eps_g = 0.01
    lb = np.linspace(-1.5, np.log10(4.0), 300)
    beta = 10 ** lb
    E = np.array([-OMEGA / 2, OMEGA / 2])
    dE = np.array([-0.5, 0.5])
    p = np.exp(-np.outer(beta, E))
    p /= p.sum(axis=1, keepdims=True)
    F = beta ** 2 * (p @ dE ** 2 - (p @ dE) ** 2)
    assert np.allclose(F, beta ** 2 / 4 / np.cosh(beta * OMEGA / 2) ** 2)
    copies = 1 / (F * eps_g ** 2)
    hot, cold = 4 / (beta * eps_g) ** 2, np.exp(beta * OMEGA) / (beta * eps_g) ** 2
    assert abs(copies[0] / hot[0] - 1) < 0.01 and abs(copies[-1] / cold[-1] - 1) < 0.01
    bmin = beta[np.argmin(copies)]
    assert abs(bmin * OMEGA / 2 * np.tanh(bmin * OMEGA / 2) - 1) < 0.02     # x tanh x = 1 at the minimum
    T = Thumb((2, 0), (-1.5, np.log10(4.0)), (5, 10))
    T.curve(b, lb, np.log10(hot), "dashedink")
    T.curve(b, lb, np.log10(cold), "dashedink")
    T.curve(b, lb, np.log10(copies), BLUE)
    b.append("\\node[dot, fill=ndBlue] at %s {};" % T.P.pt(np.log10(bmin), np.log10(copies.min())))
    T.note(b, -1.45, 9.4, "hot: $\\beta^{-2}$", col="ndInk")
    T.note(b, 0.58, 9.4, "cold: $\\ee^{\\beta\\Delta}$", anchor="east", col="ndInk")
    T.frame(b, 7, "learning from a Gibbs state", "copies $\\propto\\beta^{-2}$ hot, $\\ee^{\\beta\\Delta}$ cold", "$\\beta$", "copies",
            xticks=[(-1, "$0.1$"), (0, "$1$")], yticks=[(5, "$10^5$"), (10, "$10^{10}$")])

    # ---- 8. Ho--Kalman on shots
    y0 = data(X0)
    sv = svals(y0)[:4]
    Ns = np.logspace(1, 6, 11)                                           # as in ch06_hankel_noise
    rng6 = np.random.default_rng(6)
    fl = np.array([floor(y0, int(n), rng6) for n in Ns])
    slope, logc = np.polyfit(np.log10(Ns), np.log10(fl), 1)
    assert abs(slope + 0.5) < 0.05                                         # the floor falls like N^{-1/2}
    cfl = 10 ** logc
    nstar = (cfl / sv[3]) ** 2
    T = Thumb((2, 1), (1, 5), (-2.2, 1.6))
    keep = Ns <= 1e5
    Ns, fl = Ns[keep], fl[keep]
    ln = np.linspace(1, 5, 50)
    T.curve(b, ln, np.log10(cfl / np.sqrt(10 ** ln)), "line width=1.0pt, ndGold")
    b += ["\\node[dot, fill=ndGold] at %s {};" % T.P.pt(x, np.log10(v)) for x, v in zip(np.log10(Ns), fl)]
    for k, v in enumerate(sv):
        T.curve(b, [1, 5], [np.log10(v)] * 2, "line width=%s, ndBlue%s" % (("1.0pt", "") if k == 3 else ("0.5pt", "!55")))
    b.append("\\draw[ndRed, line width=0.6pt] %s -- %s;" % (T.P.pt(np.log10(nstar), -2.2), T.P.pt(np.log10(nstar), np.log10(sv[3]))))
    b.append("\\fill[ndRed] %s circle (1.6pt);" % T.P.pt(np.log10(nstar), np.log10(sv[3])))
    T.note(b, 1.05, np.log10(sv[3]) - 0.05, "$\\sigma_4$", anchor="north west", col="ndBlue")
    T.note(b, 2.3, -1.9, "floor $c/\\sqrt N$", col="ndGold")
    T.frame(b, 8, "Ho--Kalman on shots", "seen once $\\sigma_n>c/\\sqrt N$: shots $\\propto\\sigma_n^{-2}$", "$N$", "$\\sigma_i$",
            xticks=[(np.log10(nstar), "$N^*$"), (5, "$10^5$")])

    # ---- 9. the whole budget
    Mmap, _ = reconstruction_map()
    _, V = np.linalg.eig(qubit_A())
    kg = np.linalg.cond(V / np.linalg.norm(V, axis=0))
    G = expm(qubit_A() * TAU)
    yb = np.array([(CN @ np.linalg.matrix_power(G, k) @ X0).item() for k in range(K)])
    s4 = np.linalg.svd(hankel(yb[:, None], R, S), compute_uv=False)[3]
    factors = [np.sqrt(R * S * P), 1 / s4, kg / TAU, np.linalg.cond(Mmap)]
    assert abs(factors[0] - 20) < 1e-9 and abs(factors[3] - 2) < 1e-9
    errs = 1e-2 * np.cumprod(np.r_[1.0, factors])
    T = Thumb((2, 2), (0, 5), (-2.2, 1.2))
    names = ["$\\eps_0$", "$\\Hankel$", "$\\mathbf G_\\tau$", "$\\mathbf A$", "$\\vec\\theta$"]
    for k, e in enumerate(errs):
        b.append("\\fill[%s] %s rectangle %s;" % ("ndRose" if k == 4 else "ndSky", T.P.pt(k + 0.12, -2.2), T.P.pt(k + 0.88, np.log10(e))))
        b.append("\\draw[ink, line width=0.5pt] %s rectangle %s;" % (T.P.pt(k + 0.12, -2.2), T.P.pt(k + 0.88, np.log10(e))))
        T.note(b, k + 0.5, -2.2, names[k], anchor="north")
    for k, f in enumerate(factors):
        T.note(b, k + 0.88, np.log10(errs[k + 1]) + 0.12, "$\\times%s$" % (("%.0f" % f) if f >= 10 or abs(f - round(f)) < 1e-6 else "%.2f" % f if f < 9.995 else "%.1f" % f),
               anchor="south", col="ndRed")
    T.frame(b, 9, "the whole budget", "$N\\propto\\kappa^2/\\eps^2$, $\\kappa$ the product", "", "error")
    N.write("ch06_summary", b)
    print("written figures/ch06_summary.tex; Hoeffding N(1) = %.0f, N(1e6) = %.0f; cost c = %.0f; Gibbs minimum at beta = %.2f (%.2e copies);"
          " floor c = %.2f, N* = %.0f; budget factors %s, total %.0f" % (nM[0], nM[-1], c, bmin, copies.min(), cfl, nstar, np.round(factors, 2), np.prod(factors)))


if __name__ == "__main__":
    main()
