"""ch05_spectrum.py -- the running example of Chapter 5 and its spectrum.

The damped qubit of the chapter: Hamiltonian (omega/2) sigma_z, amplitude
damping gamma1 towards |0> (Bloch vector up) and dephasing gphi, written as the
affine Bloch equation  x' = A x  for  x = (<X>, <Y>, <Z>, 1).  Its eigenvalues
(the poles) are
    0,   -gamma1,   -Gamma +- i omega,      Gamma = gamma1/2 + gphi.
The qubit starts from a pure state below the equator, X0, so that every mode is
excited.  Three readouts are used: sigma_x, sigma_z, and sigma_n along the axis
n = (x + z)/sqrt 2 halfway between them.

This module also holds the helpers used by the other ch05_*.py scripts (modal
decomposition, Hankel matrices, Ho--Kalman) and writes
  * figures/ch05_timeseries.tex : the data, sampled every TAU,
  * figures/ch05_spectrum.tex   : the poles, and the peaks they cast,
  * figures/ch05_poles.tex      : the poles recovered by Ho--Kalman.
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

OMEGA, GAMMA1, GPHI = 3.0, 0.2, 0.4
GAMMA = GAMMA1 / 2 + GPHI                      # transverse decay rate 1/T_2
THETA0 = 2 * np.pi / 3
X0 = np.array([np.sin(THETA0), 0.0, np.cos(THETA0), 1.0])
CX = np.array([[1.0, 0.0, 0.0, 0.0]])
CZ = np.array([[0.0, 0.0, 1.0, 0.0]])
CN = np.array([[1.0, 0.0, 1.0, 0.0]]) / np.sqrt(2)
TAU = 0.25
POLES = np.array([0.0, -GAMMA1, -GAMMA + 1j * OMEGA, -GAMMA - 1j * OMEGA])


def qubit_A(omega=OMEGA, u=0.0, gamma1=GAMMA1, gphi=GPHI):
    """Affine Bloch equation  x' = A x  for x = (<X>,<Y>,<Z>,1), drive u sigma_x / 2."""
    G = gamma1 / 2 + gphi                       # transverse decay rate
    A = np.array([[-G, -omega, 0.0],
                  [omega, -G, -u],
                  [0.0, u, -gamma1]])
    b = np.array([0.0, 0.0, gamma1])           # relaxes towards <Z>=+1 (ground)
    Aemb = np.zeros((4, 4))
    Aemb[:3, :3] = A
    Aemb[:3, 3] = b
    return Aemb


def modes(A, c, x0):
    """Eigenvalues lam_i and residues r_i with  c e^{At} x0 = sum_i r_i e^{lam_i t}."""
    lam, V = np.linalg.eig(A)
    W = np.linalg.inv(V)                        # rows: left eigenvectors w_i^dagger
    res = (np.asarray(c).reshape(-1) @ V) * (W @ x0)
    return lam, res


def outputs(A, C, x0, t):
    """Exact outputs y(t) = C e^{At} x0 at the times t, shape (len(t), p)."""
    return np.array([C @ expm(A * s) @ x0 for s in np.atleast_1d(t)])


def hankel(y, r, s):
    """Block Hankel matrix H_{rs} of an output sequence y[k] (shape (K, p))."""
    return np.vstack([np.hstack([y[i + j][:, None] for j in range(s)]) for i in range(r)])


def ho_kalman(y, n, r=None, s=None):
    """Recover (A_d, C, x0) up to similarity from Markov parameters y[k] = C A_d^k x0."""
    K, p = y.shape
    r = r or n + 1
    s = s or K - r
    H0, H1 = hankel(y, r, s), hankel(y[1:], r, s)
    P, S, Qt = np.linalg.svd(H0)
    P1, S1, Q1 = P[:, :n], np.diag(S[:n]), Qt[:n].T
    Sh = np.sqrt(S1)
    Ad = np.linalg.inv(Sh) @ P1.T @ H1 @ Q1 @ np.linalg.inv(Sh)
    C = P1[:p] @ Sh
    x0 = Sh @ Q1[0]
    return Ad, C, x0


def check_model():
    A = qubit_A()
    lam = np.linalg.eigvals(A)
    assert all(np.min(abs(lam - p)) < 1e-12 for p in POLES)
    for C in (CX, CZ, CN):
        lam_c, res = modes(A, C, X0)
        t = np.linspace(0, 7, 15)
        y = np.array([np.sum(res * np.exp(lam_c * s)) for s in t])
        assert np.allclose(y, outputs(A, C, X0, t)[:, 0])
    return A


def dtft(lam, res, nu, tau=TAU):
    """tau sum_{k>=0} y_k e^{-i nu k tau} for y_k = sum_i r_i e^{lam_i k tau} (closed form)."""
    nu = np.asarray(nu)[:, None]
    return tau * np.sum(res / (1 - np.exp((lam - 1j * nu) * tau)), axis=1)


def timeseries_figure(A):
    """<sigma_x>(t) and <sigma_z>(t) of the running example, with their samples."""
    t = np.linspace(0, 16, 641)
    y = outputs(A, np.vstack([CX, CZ]), X0, t)
    ts = TAU * np.arange(int(16 / TAU) + 1)
    ys = outputs(A, np.vstack([CX, CZ]), X0, ts)
    x0 = X0[0]
    assert np.allclose(y[:, 0], x0 * np.exp(-GAMMA * t) * np.cos(OMEGA * t))
    assert np.allclose(y[:, 1], 1 - (1 - X0[2]) * np.exp(-GAMMA1 * t))
    Pl = N.Plot((0, 16), (-1.0, 1.15), 12.0, 4.6)
    b = Pl.axes("$t$", "", xticks=[(5, "$5$"), (10, "$10$"), (15, "$15$")],
                yticks=[(-1, "$-1$"), (0, "$0$"), (1, "$1$")])
    b.append("\\draw[faint] %s -- %s;" % (Pl.pt(0, 0), Pl.pt(16, 0)))
    env = x0 * np.exp(-GAMMA * t)
    b.append(Pl.curve(t, env, "dashedink"))
    b.append(Pl.curve(t, -env, "dashedink"))
    b.append("\\draw[dashedink] %s -- %s;" % (Pl.pt(0, 1), Pl.pt(16, 1)))
    for k, col in enumerate(["ndBlue", "ndRed"]):
        b.append(Pl.curve(t, y[:, k], "line width=0.9pt, %s" % col))
        b += ["\\filldraw[fill=white, draw=%s, line width=0.5pt] %s circle (1.2pt);" % (col, Pl.pt(s, v)) for s, v in zip(ts, ys[:, k])]
    b.append("\\node[font=\\small, text=ndBlue, anchor=west] at %s {$\\langle\\sigma_x\\rangle$};" % Pl.pt(16.2, 0.0))
    b.append("\\node[font=\\small, text=ndRed, anchor=west] at %s {$\\langle\\sigma_z\\rangle$};" % Pl.pt(16.2, y[-1, 1]))
    # one period, between two successive maxima of <sigma_x>
    p0, p1 = 2 * np.pi / OMEGA, 4 * np.pi / OMEGA
    yb = -0.78
    b.append("\\draw[ink, |-|] %s -- %s node[midway, below, font=\\scriptsize] {period $2\\pi/\\omega$};" % (Pl.pt(p0, yb), Pl.pt(p1, yb)))
    for p in (p0, p1):
        b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(p, yb), Pl.pt(p, x0 * np.exp(-GAMMA * p))))
    b.append("\\node[note, anchor=west] (e) at %s {envelope $\\pm0.87\\,\\ee^{-\\Gamma t}$:\\\\ the oscillation decays at the rate $\\Gamma$};" % Pl.pt(4.6, -0.62))
    b.append("\\draw[pointer] (e.west) to[bend left=15] %s;" % Pl.pt(3.0, -x0 * np.exp(-GAMMA * 3.0) - 0.02))
    b.append("\\node[note, anchor=north] (z) at %s {$\\langle\\sigma_z\\rangle\\to1$, the ground state,\\\\ at the smaller rate $\\gamma_1$};" % Pl.pt(11.6, 0.62))
    b.append("\\draw[pointer] (z.north) to[bend right=10] %s;" % Pl.pt(10.2, 1 - 1.5 * np.exp(-GAMMA1 * 10.2) - 0.03))
    b.append("\\node[note, anchor=south east] at %s {beads: the samples,\\\\ at intervals $\\tau=%.2f$};" % (Pl.pt(16, -0.95), TAU))
    return b


def spectrum_figure(A):
    """The poles (lambda-plane turned by 90 degrees) above the peaks they cast."""
    lx, rx = modes(A, CX, X0)
    lz, rz = modes(A, CZ, X0)
    lx, rx = lx[np.abs(rx) > 1e-12], rx[np.abs(rx) > 1e-12]      # modes absent from <sigma_x>
    keep = (np.abs(lz) > 1e-12) & (np.abs(rz) > 1e-12)            # drop the steady state lambda = 0,
    lz, rz = lz[keep], rz[keep]                                    # i.e. plot <sigma_z> - 1
    nu = np.linspace(-4.5, 4.5, 721)
    px, pz = np.abs(dtft(lx, rx, nu)) ** 2, np.abs(dtft(lz, rz, nu)) ** 2
    sx, sz = px.max(), pz.max()
    # the DFT of K samples lies on the closed form
    K = 256
    ys = outputs(A, np.vstack([CX, CZ]), X0, TAU * np.arange(K))
    ys[:, 1] -= 1.0
    bins = 2 * np.pi * np.fft.fftfreq(K, d=TAU)
    dft = TAU * np.fft.fft(ys, axis=0)
    assert np.allclose(dft[:, 0], dtft(lx, rx, bins), atol=1e-3 * np.sqrt(sx))
    assert np.allclose(dft[:, 1], dtft(lz, rz, bins), atol=1e-3 * np.sqrt(sz))
    # half-widths at half height: Gamma for the side peaks, gamma_1 for the central one
    def hwhm(p, centre):
        near = np.abs(nu - centre) < 1.5
        above = nu[near & (p > p[near].max() / 2)]
        return (above.max() - above.min()) / 2
    assert abs(hwhm(px, OMEGA) - GAMMA) < 0.04 and abs(hwhm(pz, 0.0) - GAMMA1) < 0.02
    sxu, syu = 1.45, 3.2                         # cm per unit frequency / per unit rate
    X = lambda v: sxu * v
    b = []
    # ---- top: the poles, at height -Re lambda above the frequency axis
    top = 3.6
    Ptop = lambda z: (X(z.imag), top + syu * (-z.real))
    b.append("\\fill[ndSky, fill opacity=0.45] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (X(-4.6), top, X(4.6), top + syu * 0.68))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,%.2f) -- (%.2f,%.2f) node[right, text=ndInk, font=\\small] {$\\mathrm{Im}\\,\\lambda$};" % (X(-4.7), top, X(4.85), top))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,%.2f) -- (%.2f,%.2f) node[above, text=ndInk, font=\\small] {$-\\mathrm{Re}\\,\\lambda$};" % (X(-4.6), top, X(-4.6), top + syu * 0.72))
    for v, lab in [(0.2, "$0.2$"), (0.5, "$0.5$")]:
        b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f) node[left, font=\\scriptsize, text=ndInk] {%s};" % (X(-4.6) + 0.06, top + syu * v, X(-4.6) - 0.06, top + syu * v, lab))
    b.append("\\node[note, anchor=south] at (%.2f,%.2f) {the $\\lambda$-plane turned through $90^\\circ$: the stable half-plane lies above,\\\\ and the imaginary axis becomes the frequency axis of the spectrum below};" % (0.0, top + syu * 0.72))
    cones = [(POLES[2], "ndBlue"), (POLES[3], "ndBlue"), (POLES[1], "ndRed")]
    for z, col in cones:
        a, w = -z.real, z.imag
        b.append("\\fill[%s!25] (%.2f,%.2f) -- (%.2f,%.2f) -- (%.2f,%.2f) -- cycle;" % (col, *Ptop(z), X(w - a), top, X(w + a), top))
        b.append("\\draw[%s, line width=0.5pt] (%.2f,%.2f) -- (%.2f,%.2f) (%.2f,%.2f) -- (%.2f,%.2f);" % (col, X(w - a), top, *Ptop(z), *Ptop(z), X(w + a), top))
    for z, lab, pos in [(POLES[2], "$-\\Gamma+\\imath\\omega$", "above"), (POLES[3], "$-\\Gamma-\\imath\\omega$", "above"),
                        (POLES[1], "$-\\gamma_1$", "above")]:
        col = "ndRed" if z.imag == 0 else "ndBlue"
        b.append("\\node[dot, fill=%s, minimum size=5pt, label={[font=\\scriptsize, text=%s]%s:%s}] at (%.2f,%.2f) {};" % (col, col, pos, lab, *Ptop(z)))
    b.append("\\node[dot, fill=ndInk!60, minimum size=5pt, label={[font=\\scriptsize, text=ndInk!70]below:$0$}] at (%.2f,%.2f) {};" % Ptop(POLES[0]))
    b.append("\\node[note] (c) at (%.2f,%.2f) {a pole at height $a$\\\\ produces a peak of\\\\ half-width $a$};" % (X(1.45), top + syu * 0.36))
    b.append("\\draw[pointer] (c.east) to[bend left=20] (%.2f,%.2f);" % (X(OMEGA - 0.3), top + syu * 0.22))
    # ---- bottom: the power spectra, normalised
    h = 3.0
    yy = lambda p, s: h * p / s
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[right, text=ndInk, font=\\small] {$\\nu$};" % (X(-4.7), X(4.85)))
    for v in (-3, 0, 3):
        b.append("\\draw[faint] (%.2f,0.06) -- (%.2f,-0.06) node[below, font=\\scriptsize, text=ndInk] {$%s$};" % (X(v), X(v), {-3: "-\\omega", 0: "0", 3: "\\omega"}[v]))
    for p, s, col in [(px, sx, "ndBlue"), (pz, sz, "ndRed")]:
        b.append(N.polyline([(X(v), yy(q, s)) for v, q in zip(nu, p)], "line width=1.2pt, %s" % col))
    sel = (np.abs(bins) <= 4.5) & (np.arange(K) % 2 == 0)
    for col, k, s in [("ndBlue", 0, sx), ("ndRed", 1, sz)]:
        b += ["\\filldraw[fill=white, draw=%s, line width=0.5pt] (%.2f,%.2f) circle (1.2pt);" % (col, X(v), yy(abs(d) ** 2, s)) for v, d in zip(bins[sel], dft[sel, k])]
    # half-height brackets and the dashed links up to the poles
    for z, col in cones:
        a, w = -z.real, z.imag
        b.append("\\draw[%s, line width=0.8pt, |-|] (%.2f,%.2f) -- (%.2f,%.2f);" % (col, X(w - a), h / 2, X(w + a), h / 2))
        for e in (w - a, w + a):
            b.append("\\draw[dashedink] (%.2f,%.2f) -- (%.2f,%.2f);" % (X(e), h / 2, X(e), top))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=south] at (%.2f,%.2f) {$2\\Gamma$};" % (X(-OMEGA), h / 2 + 0.05))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=north] at (%.2f,%.2f) {$2\\gamma_1$};" % (X(0.0), h / 2 - 0.05))
    b.append("\\node[note, anchor=south west, text=ndBlue] at (%.2f,%.2f) {$\\abs{\\hat y(\\nu)}^2$ for $\\langle\\sigma_x\\rangle$:\\\\ broad peaks at $\\pm\\omega$};" % (X(3.55), h * 0.55))
    b.append("\\node[note, anchor=south east, text=ndRed] at (%.2f,%.2f) {for $\\langle\\sigma_z\\rangle-1$:\\\\ a narrow peak at $0$};" % (X(-0.45), h * 0.82))
    b.append("\\node[note, anchor=north west] at (%.2f,%.2f) {beads: DFT of $%d$ samples\\\\ curves: closed form};" % (X(-4.6), -0.3, K))
    b.append("\\node[note, anchor=north east] at (%.2f,%.2f) {the steady state $\\lambda=0$ lies on the axis and\\\\ gives a line of zero width at $\\nu=0$ (subtracted)};" % (X(4.7), -0.3))
    return b


def poles_figure(A):
    """True poles, and Ho--Kalman estimates from noiseless and from noisy data."""
    C = np.vstack([CX, CZ])
    K = 80
    y = outputs(A, C, X0, TAU * np.arange(K))
    Ad_hat, _, _ = ho_kalman(y, n=4)
    lam_hat = np.log(np.linalg.eigvals(Ad_hat).astype(complex)) / TAU
    assert all(np.min(abs(lam_hat - p)) < 1e-8 for p in POLES)
    rng = np.random.default_rng(1)
    sigma, runs = 0.02, 60
    clouds = []
    for _ in range(runs):
        Adn, _, _ = ho_kalman(y + sigma * rng.standard_normal(y.shape), n=4)
        clouds.append(np.log(np.linalg.eigvals(Adn).astype(complex)) / TAU)
    clouds = np.concatenate(clouds)
    sxu, syu = 7.0, 1.0                          # cm per unit of Re / Im
    P = lambda z: (sxu * z.real, syu * z.imag)
    b = []
    b.append("\\fill[ndSky, fill opacity=0.45] (%.2f,%.2f) rectangle (0,%.2f);" % (-0.85 * sxu, -3.7 * syu, 3.7 * syu))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[right, text=ndInk, font=\\small] {$\\mathrm{Re}\\,\\lambda$};" % (-0.85 * sxu, 0.25 * sxu))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[above, text=ndInk, font=\\small] {$\\mathrm{Im}\\,\\lambda$};" % (-3.7 * syu, 3.9 * syu))
    for v in (-0.8, -0.5, -0.2, 0.0):
        b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f) node[below, font=\\scriptsize, text=ndInk] {$%s$};" % (sxu * v, -3.7 * syu, sxu * v, -3.82 * syu, ("%g" % v).replace("-", "-")))
    for v in (-3, 3):
        b.append("\\draw[faint] (-0.06,%.2f) -- (0.06,%.2f) node[right, font=\\scriptsize, text=ndInk] {$%d$};" % (syu * v, syu * v, v))
    shown = clouds[(np.abs(clouds.real) < 0.84) & (np.abs(clouds.imag) < 3.6)]
    b += ["\\fill[ndBlue, fill opacity=0.55] (%.2f,%.2f) circle (1.1pt);" % P(z) for z in shown]
    for z in POLES:
        b.append("\\node[dot, fill=ndRed, minimum size=5pt] at (%.2f,%.2f) {};" % P(z))
    for z in lam_hat:
        b.append("\\draw[ndInk, line width=0.7pt] (%.2f,%.2f) circle (4.5pt);" % P(z))
    b.append("\\node[note, anchor=west] (r) at (%.2f,%.2f) {red dots: the eigenvalues of $\\mathbf A$\\\\ rings: Ho--Kalman from $%d$ exact samples,\\\\ correct to about $10^{-10}$};" % (0.5, 2.3, K))
    b.append("\\draw[pointer] (r.west) to[bend right=15] (%.2f,%.2f);" % (P(POLES[2])[0] + 0.2, P(POLES[2])[1] - 0.05))
    b.append("\\node[note, anchor=west, text=ndBlue] (n) at (%.2f,%.2f) {blue: %d repetitions with noise of\\\\ standard deviation $%.2f$ on each sample;\\\\ the slow decay $-\\gamma_1$ is the\\\\ least well determined};" % (0.5, -1.0, runs, sigma))
    b.append("\\draw[pointer] (n.west) to[bend left=15] (%.2f,%.2f);" % (P(POLES[1])[0] + 0.1, -0.15))
    b.append("\\node[note] at (%.2f,%.2f) {stable half-plane\\\\ (the $\\mathrm{Re}$ axis is\\\\ stretched $%d$-fold)};" % (-0.5 * sxu, -1.7 * syu, sxu / syu))
    spread = [np.std(np.abs(clouds[np.argmin(abs(clouds[:, None] - POLES[None, :]), axis=1) == i] - POLES[i])) for i in range(4)]
    print("noisy Ho-Kalman: spread of the estimates around each pole", np.round(spread, 3))
    return b


if __name__ == "__main__":
    A = check_model()
    N.write("ch05_timeseries", timeseries_figure(A))
    N.write("ch05_spectrum", spectrum_figure(A))
    N.write("ch05_poles", poles_figure(A))
    print("ch05 model: poles", np.round(POLES, 3))
    print("written figures/ch05_timeseries.tex, ch05_spectrum.tex, ch05_poles.tex")
