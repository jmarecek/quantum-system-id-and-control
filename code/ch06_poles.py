"""ch06_poles.py -- the poles found by Ho--Kalman, scattered by the shots.

The set-up of ch06_hankel_noise: the qubit of Chapter 5 measured along n,
K = 60 samples at tau = 0.25, Hankel matrix with R = 10 block rows and S = 40
columns, Ho--Kalman of order 4.  Each sample is the mean of N shots, and the
algorithm is run on 150 independent data sets for each of N = 10^2, 10^3 and
10^4 (below and above N* = 130 of ch06_hankel_noise).
  (a) Two windows on the lambda-plane, drawn to the same scale: one about the
      pole -Gamma + i omega, one about the poles 0 and -gamma_1 on the real
      axis.  The clouds of estimates shrink like N^{-1/2}, and that of
      -gamma_1 is the widest.  At N = 10^2 many estimates fall outside the
      windows: the weakest mode is lost in the noise, and the fourth pole is
      put anywhere on the negative real axis, or at Im lambda = pi / tau (a
      negative eigenvalue of G_tau).  The eigenvalues e^{-gamma_1 tau} = 0.95
      and e^{0} = 1 lie close together, nearly merged, and at N = 10^3 the
      noise also pushes the two estimates apart: those of -gamma_1 to about
      -0.35, those of 0 to about +0.04.
  (b) The z-plane of G_tau, to the same scale, with the eigenvalues
      mu = e^{lambda tau} of the same runs and the images of the two windows
      of (a).  The poles are computed as lambda = (log mu) / tau, which
      magnifies distances near mu by 1/(tau |mu|), about four and a half:
      the same clouds are four or five times smaller here.
Writes figures/ch06_poles.tex.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import needham as N
from ch05_spectrum import ho_kalman, X0, TAU, POLES, GAMMA, OMEGA, GAMMA1
from ch06_shots import sample_mean
from ch06_hankel_noise import data, R, S

SHOTS = (100, 1000, 10000)
COLS = {100: "ndRed", 1000: "ndGold!90!ndInk", 10000: "ndBlue"}
REPS = 150
SC = 4.0                                                   # cm per unit, in both planes
W1 = (-0.9, -0.1, 2.6, 3.4)                                # windows (re0, re1, im0, im1)
W2 = (-0.75, 0.25, -0.29, 0.13)
ROWS = {100: -0.21, 1000: -0.14, 10000: -0.07}            # real estimates in W2 are drawn in rows below the axis
WZ = (0.3, 1.1, -0.12, 0.75)


def inside(z, w):
    return (w[0] <= z.real <= w[1]) and (w[2] <= z.imag <= w[3])


def runs(y, shots, rng):
    mus = []
    for _ in range(REPS):
        Ad, _, _ = ho_kalman(sample_mean(y, shots, rng)[:, None], 4, r=R, s=S)
        mus.append(np.linalg.eigvals(Ad).astype(complex))
    return np.array(mus)


def matched(lam):
    """The estimates matched to the true poles, run by run (columns follow POLES)."""
    out = np.empty_like(lam)
    for k, l in enumerate(lam):
        i, j = linear_sum_assignment(np.abs(l[:, None] - POLES[None, :]))
        out[k, j] = l[i]
    return out


def window(b, w, ox, oy, pts, true, label_axes, rows=None):
    """A window of the complex plane with its lower-left corner at (ox, oy) on the page.

    With `rows`, the real points of each colour are drawn at the given height below the axis."""
    P = lambda z: (ox + SC * (z.real - w[0]), oy + SC * (z.imag - w[2]))
    b.append("\\fill[ndPaper] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (*P(w[0] + 1j * w[2]), *P(w[1] + 1j * w[3])))
    if w[2] < 0 < w[3]:
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (*P(w[0]), *P(w[1])))
    if w[0] < 0 < w[1]:
        b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (*P(1j * w[2]), *P(1j * w[3])))
    if rows:
        for z in true:
            b.append("\\draw[faint, densely dotted] (%.3f,%.3f) -- (%.3f,%.3f);" % (*P(z), *P(z + 1j * (min(rows.values()) - 0.04))))
    for col, zs in pts:
        for z in zs:
            if rows and abs(z.imag) < 1e-9:
                z = z.real + 1j * rows[col]
            if inside(z, w):
                b.append("\\fill[%s, fill opacity=0.7] (%.3f,%.3f) circle (0.75pt);" % (col, *P(z)))
    for z in true:
        b.append("\\draw[ndInk, line width=0.5pt] (%.3f,%.3f) circle (3.6pt);" % P(z))
    b.append("\\draw[ink, line width=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (*P(w[0] + 1j * w[2]), *P(w[1] + 1j * w[3])))
    for v in label_axes[0]:
        b.append("\\draw[faint] (%.3f,%.3f) -- ++(0,-0.08) node[below, font=\\scriptsize, text=ndInk] {$%s$};" % (P(v + 1j * w[2]) + (str(v).rstrip("0").rstrip(".") if v else "0",)))
    for v, lab in label_axes[1]:
        b.append("\\draw[faint] (%.3f,%.3f) -- ++(-0.08,0) node[left, font=\\scriptsize, text=ndInk] {%s};" % (P(w[0] + 1j * v) + (lab,)))
    return P


def main():
    rng = np.random.default_rng(8)
    y = data(X0)
    mus = {n: runs(y, n, rng) for n in SHOTS}
    lams = {n: np.log(mus[n]) / TAU for n in SHOTS}
    # how the clouds shrink, and which is widest
    sd = {}
    for n in SHOTS:
        m = matched(lams[n])
        sd[n] = np.sqrt(np.mean(np.abs(m - m.mean(0)) ** 2, axis=0))      # spread of each cloud, columns follow POLES
        print("N = %5d  spread of the clouds (0, -g1, -G+iw):" % n, np.round(sd[n][:3], 4),
              " mean -gamma_1 estimate %.3f" % m[:, 1].real.mean())
    ratio = sd[1000][2] / sd[10000][2]
    assert 2.2 < ratio < 4.5                                                # about sqrt(10)
    assert sd[10000][1] > sd[10000][2] and sd[10000][1] > sd[10000][0]      # -gamma_1 is the widest
    shown = lambda l: l[l.imag >= -1e-9]
    out = {n: sum(1 for z in shown(lams[n].ravel()) if not (inside(z, W1) or inside(z, W2))) for n in SHOTS}
    nyq = int(np.sum(np.abs(lams[100].imag - np.pi / TAU) < 1e-6))
    neg_real = int(np.sum([np.any((np.abs(l.imag) < 1e-9) & (l.real < W2[0])) for l in lams[100]]))
    print("estimates outside the windows:", out, "; at Im = pi/tau (N=100):", nyq, "; runs with a real pole left of the window:", neg_real)
    assert out[10000] == 0 and out[100] > 60
    outz = {n: sum(1 for z in mus[n].ravel() if z.imag >= -1e-9 and not inside(z, WZ)) for n in SHOTS}
    print("eigenvalues outside the z window:", outz)
    m3 = matched(lams[1000])
    assert m3[:, 1].real.mean() < -0.3 and m3[:, 0].real.mean() > 0.0        # the displacement at N = 10^3

    b = []
    pts = [(COLS[n], shown(lams[n].ravel())) for n in SHOTS]
    # ---- (a) two windows on the lambda-plane
    yt = [(2.8, "$2.8$"), (3.0, "$3$"), (3.2, "$3.2$")]
    P1 = window(b, W1, 0.0, 0.0, pts, [POLES[2]], ((-0.8, -0.5, -0.2), yt))
    ox2 = SC * (W1[1] - W1[0]) + 0.75
    oy2 = 0.95
    P2 = window(b, W2, ox2, oy2, pts, [POLES[0], POLES[1]], ((-0.6, -0.2, 0.0, 0.2), [(0.0, "$0$")]),
                rows={COLS[n]: ROWS[n] for n in SHOTS})
    b.append("\\node[font=\\scriptsize, anchor=south west] (tp) at (%.3f,%.3f) {$-\\Gamma+\\imath\\omega$};" % P1(-0.42 + 3.2j))
    b.append("\\draw[pointer] (tp.south west) to[bend right=15] (%.3f,%.3f);" % P1(POLES[2] + 0.025 + 0.025j))
    b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,%.3f) {$-\\gamma_1$};" % (P2(POLES[1])[0], P2(POLES[1])[1] + 0.1))
    b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,%.3f) {$0$};" % (P2(POLES[0])[0], P2(POLES[0])[1] + 0.1))
    top = SC * (W1[3] - W1[2])
    b.append("\\node[note, anchor=south west, text=ndRed] at (%.3f,%.3f) {$N=10^2$: %d of the $%d$ estimates\\\\ lie outside both windows};"
             % (-0.75, top + 0.12, out[100], len(shown(lams[100].ravel()))))
    b.append("\\node[note, anchor=south west] at (%.3f,%.3f) {circles: the true poles};" % (ox2, oy2 + SC * (W2[3] - W2[2]) + 0.1))
    b.append("\\node[note, anchor=north west, align=left] at (%.3f,%.3f) {real estimates are drawn in rows\\\\ below the axis; at $10^3$ those of\\\\ $-\\gamma_1$ are also displaced to the left};" % (ox2 - 0.1, oy2 - 0.42))
    for n, k in zip(SHOTS, range(3)):
        b.append("\\fill[%s] (%.3f,%.3f) circle (1.6pt) node[right, font=\\scriptsize, text=%s] {$N=10^{%d}$};"
                 % (COLS[n], ox2 + 0.05 + 1.35 * k, oy2 + SC * (W2[3] - W2[2]) + 0.8, COLS[n], int(np.log10(n))))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(a) the $\\lambda$-plane near the poles};" % ((ox2 + SC * (W2[1] - W2[0])) / 2, -1.0))
    # ---- (b) the z-plane
    ozx = ox2 + SC * (W2[1] - W2[0]) + 1.25
    zpts = [(COLS[n], mus[n].ravel()) for n in SHOTS]
    Pz = window(b, WZ, ozx, -0.35, zpts, list(np.exp(POLES[:3] * TAU)), ((0.5, 1.0), [(0.0, "$0$"), (0.5, "$0.5$")]))
    th = np.linspace(-0.12, 0.85, 200)
    arc = [Pz(np.exp(1j * t)) for t in th if inside(np.exp(1j * t), WZ)]
    b.append(N.polyline(arc, "ink, line width=0.6pt"))
    for w in (W1, W2):                                                   # the images of the windows of (a)
        edge = np.concatenate([np.linspace(w[0], w[1], 60) + 1j * w[2], w[1] + 1j * np.linspace(w[2], w[3], 60),
                               np.linspace(w[1], w[0], 60) + 1j * w[3], w[0] + 1j * np.linspace(w[3], w[2], 60)])
        b.append(N.polyline([Pz(np.exp(z * TAU)) for z in edge] + [Pz(np.exp(edge[0] * TAU))], "dashedink"))
    b.append("\\node[note, anchor=south, align=center] at (%.3f,%.3f) {dashed: the windows of (a); here the\\\\ clouds are $1/(\\tau\\abs{\\mu})\\approx%.1f$ times smaller};"
             % (Pz(0.7 + 0.75j)[0], Pz(0.7 + 0.75j)[1] + 0.1, 1 / (TAU * abs(np.exp(POLES[2] * TAU)))))
    b.append("\\node[note, anchor=north west, text=ndRed, align=left] at (%.3f,%.3f) {$N=10^2$: all but\\\\ %d in view};" % (*Pz(0.31 + 0.3j), outz[100]))
    b.append("\\node[font=\\scriptsize, anchor=west] at (%.3f,%.3f) {$\\abs{z}=1$};" % Pz(0.82 + 0.6j))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {(b) the $z$-plane, $\\mu=\\ee^{\\lambda\\tau}$};" % (Pz(0.7 - 0.12j)[0], -1.0))
    N.write("ch06_poles", b)
    print("written figures/ch06_poles.tex")


if __name__ == "__main__":
    main()
