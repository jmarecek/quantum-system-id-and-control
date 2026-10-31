"""ch06_costmap.py -- a map of the cost of the experiment.

The qubit of Chapter 5 with the Hankel set-up of ch06_nearly (K = 60 samples,
R = 10, S = 40).  The weakest mode rises above the noise floor of
ch06_hankel_noise, about c / sqrt(N) with c = 6.6, when
    N* = (c / sigma_4)^2,
sigma_4 the smallest non-zero singular value of the exact Hankel matrix H(0).
N* is mapped over the choices of the experimenter:
  (a) the tilt phi of the preparation from |1> against the angle psi of the
      axis of measurement from z (both in the x-z plane), with tau = 0.25;
  (b) phi against the sampling interval tau, measured along n (psi = 45).
The cost is infinite on walls: phi = 0 (the precession is not excited),
phi = 180 (|0>, nothing is excited), psi = 0 and 180 (sigma_z does not see the
precession), psi = 90 (sigma_x does not see the relaxation), tau = pi/omega
and 2 pi/omega (modes merge), and it grows without bound as tau -> 0, where
every e^{lambda tau} tends to 1.  The example of the chapter, (phi, psi, tau)
= (60, 45, 0.25), needs N* = 130; the best point of (a) about 75.  The three
curves of ch06_nearly are cuts through these maps (dotted).
Writes figures/ch06_costmap.tex.
"""
import numpy as np
from scipy.linalg import expm
from skimage import measure
import needham as N
from ch05_spectrum import qubit_A, hankel, X0, CN, TAU, OMEGA
from ch06_shots import tilted
from ch06_nearly import readout, K, R, S
from ch06_hankel_noise import data, floor

C_FLOOR = 6.6                                              # fitted in ch06_hankel_noise
LEVELS = [2, 3, 4, 5, 6]                                   # log10 of N*
BANDS = ["ndGreen!22", "ndSky!75", "ndSky!30", "ndRose!55", "ndRose", "ndRed!28"]


def powers(tau):
    G = expm(qubit_A() * tau)
    out, g = [], np.eye(4)
    for _ in range(K):
        out.append(g)
        g = G @ g
    return out


def sigma4(x0, C, Gp):
    y = np.array([(C @ g @ x0).item() for g in Gp])
    return np.linalg.svd(hankel(y[:, None], R, S), compute_uv=False)[3]


def nstar(s4, c):
    return (c / np.maximum(s4, 1e-300)) ** 2


def bands(Pl, xs, ys, F):
    """Run-length rectangles of the colour bands of F (rows follow ys, columns xs)."""
    out = []
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    idx = np.searchsorted(LEVELS, F)
    for i, y in enumerate(ys):
        j = 0
        while j < len(xs):
            k = j
            while k + 1 < len(xs) and idx[i, k + 1] == idx[i, j]:
                k += 1
            x0, x1 = max(xs[j] - dx / 2, xs[0]), min(xs[k] + dx / 2, xs[-1])
            y0, y1 = max(y - dy / 2, ys[0]), min(y + dy / 2, ys[-1])
            out.append("\\fill[%s] %s rectangle %s;" % (BANDS[idx[i, j]], Pl.pt(x0, y0), Pl.pt(x1, y1)))
            j = k + 1
    return out


def contour_lines(Pl, xs, ys, F, labels):
    """Contours of F at LEVELS; labels maps a level to page-data points near which it is labelled."""
    out, tags = [], []
    for lev in LEVELS:
        paths = []
        for path in measure.find_contours(F, lev):
            pts = np.array([(np.interp(cj, np.arange(len(xs)), xs), np.interp(ci, np.arange(len(ys)), ys)) for ci, cj in path])
            paths.append(pts)
            out.append(N.polyline([Pl.P(a, c) for a, c in pts], "line width=0.45pt, ndInk!50"))
        for target in labels.get(lev, []):
            allp = np.vstack(paths)
            q = allp[np.argmin(np.sum(((allp - target) / np.array([Pl.xlim[1] - Pl.xlim[0], Pl.ylim[1] - Pl.ylim[0]])) ** 2, axis=1))]
            tags.append("\\node[font=\\tiny, fill=white, fill opacity=0.85, text opacity=1, inner sep=0.6pt] at %s {$10^{%d}$};" % (Pl.pt(*q), lev))
    return out + tags


def star(p, r=0.13):
    """A five-pointed star (no TikZ shape library is loaded)."""
    a = np.pi / 2 + np.arange(10) * np.pi / 5
    rr = np.where(np.arange(10) % 2 == 0, r, 0.42 * r)
    pts = [np.asarray(p) + q * np.array([np.cos(t), np.sin(t)]) for q, t in zip(rr, a)]
    return "\\fill[ndGold, draw=ndInk, line width=0.3pt] %s -- cycle;" % " -- ".join(N.pt(q) for q in pts)


def walls(Pl, lines):
    return ["\\draw[line width=1.6pt, ndRed] %s -- %s;" % (Pl.pt(*a), Pl.pt(*b)) for a, b in lines]


def main():
    deg = np.radians
    rng = np.random.default_rng(6)
    c = C_FLOOR
    assert abs(floor(data(X0), 10 ** 4, rng) * 100.0 / c - 1) < 0.06     # the floor is about c / sqrt(N)
    Gp = powers(TAU)
    n_ex = nstar(sigma4(X0, CN, Gp), c)
    assert np.allclose(X0, tilted(deg(60.0))) and np.allclose(CN, readout(deg(45.0)))
    assert 110 < n_ex < 150                                      # N* of ch06_hankel_noise

    # ---- (a) phi against psi
    phis = np.linspace(0.0, 180.0, 91)
    psis = np.linspace(0.0, 180.0, 91)
    Fa = np.array([[np.log10(nstar(sigma4(tilted(deg(p)), readout(deg(q)), Gp), c)) for q in psis] for p in phis])
    Fa = np.minimum(Fa, 9.0)
    for q in (0.0, 90.0, 180.0):                                 # walls of the readout
        assert sigma4(tilted(deg(60.0)), readout(deg(q)), Gp) < 1e-10
    for p in (0.0, 180.0):                                       # walls of the preparation
        assert sigma4(tilted(deg(p)), CN, Gp) < 1e-10
    ia, ja = np.unravel_index(np.argmin(Fa), Fa.shape)
    best_a = (psis[ja], phis[ia], 10 ** Fa[ia, ja])
    ib, jb = np.unravel_index(np.argmin(np.where((phis[:, None] <= 90) & (psis[None, :] <= 90), Fa, 99)), Fa.shape)
    assert best_a[2] < n_ex and 1.5 < n_ex / best_a[2] < 2.0       # a better axis exists in the other quadrant
    assert abs(10 ** Fa[ib, jb] / n_ex - 1) < 0.1                    # in the first quadrant the example is nearly the best

    # ---- (b) phi against tau, measured along n
    taus = np.linspace(0.02, 2.4, 96)
    Fb = []
    for t in taus:
        Gt = powers(t)
        Fb.append([np.log10(nstar(sigma4(tilted(deg(p)), CN, Gt), c)) for p in phis])
    Fb = np.minimum(np.array(Fb).T, 9.0)
    for t in (np.pi / OMEGA, 2 * np.pi / OMEGA):
        assert sigma4(X0, CN, powers(t)) < 1e-8                  # merged
    ic, jc = np.unravel_index(np.argmin(Fb), Fb.shape)
    best_b = (taus[jc], phis[ic], 10 ** Fb[ic, jc])
    assert 0.2 < best_b[0] < 0.45 and best_b[2] < n_ex
    print("floor constant c = %.2f; N* example %.0f; best (psi, phi) = (%.0f, %.0f) N* = %.0f; first quadrant best %.0f at (%.0f, %.0f);"
          " best (tau, phi) = (%.2f, %.0f) N* = %.0f" % (c, n_ex, *best_a, 10 ** Fa[ib, jb], psis[jb], phis[ib], *best_b))

    b = []
    ends = [(0, "$\\ket1$"), (60, "$60^\\circ$"), (90, "$90^\\circ$"), (180, "$\\ket0$")]
    # ---- (a)
    Pa = N.Plot((0.0, 180.0), (0.0, 180.0), 5.0, 4.4, origin=(0.0, 0.0))
    b += bands(Pa, psis, phis, Fa)
    b += contour_lines(Pa, psis, phis, Fa, {2: [(150, 70)], 3: [(105, 22), (165, 125)], 4: [(60, 148)], 5: [(115, 168)]})
    b += walls(Pa, [((0, 0), (180, 0)), ((0, 180), (180, 180)), ((0, 0), (0, 180)), ((90, 0), (90, 180)), ((180, 0), (180, 180))])
    b.append("\\draw[line width=0.8pt, ndInk, dash pattern=on 1pt off 1.5pt] %s -- %s;" % (Pa.pt(45, 0), Pa.pt(45, 90)))
    b.append("\\draw[line width=0.8pt, ndInk, dash pattern=on 1pt off 1.5pt] %s -- %s;" % (Pa.pt(0, 60), Pa.pt(90, 60)))
    b.append("\\node[font=\\tiny, anchor=west] at %s {(a)};" % Pa.pt(46, 84))
    b.append("\\node[font=\\tiny, anchor=south] at %s {(b)};" % Pa.pt(20, 61))
    b.append(star(Pa.P(best_a[0], best_a[1])))
    b.append("\\node[dot, minimum size=4.5pt] at %s {};" % Pa.pt(45, 60))
    b += Pa.axes("$\\psi$", "$\\phi$", xticks=[(0, "$0^\\circ$"), (45, "$45^\\circ$"), (90, "$90^\\circ$"), (135, "$135^\\circ$"), (180, "$180^\\circ$")],
                 yticks=ends)
    top = Pa.P(0, 180)[1] + 0.1
    for q, txt, anc, dx in [(0, "$\\sigma_z$", "south west", 0.12), (90, "$\\sigma_x$", "south", 0.0), (180, "$-\\sigma_z$", "south", 0.0)]:
        b.append("\\node[font=\\scriptsize, text=ndRed, anchor=%s] at (%.3f,%.3f) {%s};" % (anc, Pa.P(q, 0)[0] + dx, top, txt))
    b.append("\\node[font=\\small] at %s {(a) preparation $\\phi$ and axis $\\psi$, $\\tau=0.25$};" % N.pt(Pa.P(90, 0) + np.array([0.0, -0.9])))
    # ---- (b)
    Pb = N.Plot((0.0, 2.4), (0.0, 180.0), 5.0, 4.4, origin=(6.75, 0.0))
    b += bands(Pb, taus, phis, Fb)
    b += contour_lines(Pb, taus, phis, Fb, {3: [(0.08, 110), (1.42, 70)], 4: [(1.7, 125)], 5: [(1.92, 40)]})
    tc = np.pi / OMEGA
    b += walls(Pb, [((0.02, 0), (2.4, 0)), ((0.02, 180), (2.4, 180)), ((tc, 0), (tc, 180)), ((2 * tc, 0), (2 * tc, 180))])
    b.append("\\draw[line width=0.8pt, ndInk, dash pattern=on 1pt off 1.5pt] %s -- %s;" % (Pb.pt(0.75, 60), Pb.pt(1.35, 60)))
    b.append("\\node[font=\\tiny, anchor=south] at %s {(c)};" % Pb.pt(1.25, 61))
    b.append(star(Pb.P(best_b[0], best_b[1])))
    b.append("\\node[dot, minimum size=4.5pt] at %s {};" % Pb.pt(TAU, 60))
    b += Pb.axes("$\\tau$", "$\\phi$", xticks=[(0.5, "$0.5$"), (tc, "$\\frac{\\pi}{\\omega}$"), (2 * tc, "$\\frac{2\\pi}{\\omega}$")],
                 yticks=ends)
    b.append("\\node[font=\\small] at %s {(b) preparation $\\phi$ and interval $\\tau$, axis $\\vec n$};" % N.pt(Pb.P(1.2, 0) + np.array([0.0, -0.9])))
    # ---- notes on the right
    xr = Pb.P(2.4, 0)[0] + 0.4
    notes = [(4.75, "red: walls of infinite cost,\\\\ where a mode is not\\\\ excited, not seen, or\\\\ merged; $\\tau\\to0$ merges all"),
             (3.05, "contours: the shots $N^*$\\\\ at which the weakest\\\\ mode rises above the\\\\ noise floor"),
             (1.35, "dot: the example, $N^*\\approx%d$;\\\\ star: the best point,\\\\ $N^*\\approx%d$ in (a), $%d$ in (b)" % (round(n_ex, -1), round(best_a[2]), round(best_b[2])))]
    for y, txt in notes:
        b.append("\\node[note, anchor=north west, align=left] at (%.2f,%.2f) {%s};" % (xr, y, txt))
    N.write("ch06_costmap", b)
    print("written figures/ch06_costmap.tex")


if __name__ == "__main__":
    main()
