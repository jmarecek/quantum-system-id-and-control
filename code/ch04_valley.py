"""ch04_valley.py -- identifiability in the parameter plane.

The damped qubit of Chapter 4, H = -(omega/2) sigma_z (omega = 1 known),
amplitude damping gamma_1 and dephasing gamma_phi (true values 0.2 and 0.1),
prepared in |+>, |+i> and |1>, sampled at 49 times in [0, 12].  The least-
squares cost of fitting (gamma_1, gamma_phi) to noise-free data:
  (a) reading sigma_x only, the data depend on the rates only through
      1/T_2 = gamma_1/2 + gamma_phi: the cost has a flat valley along a line,
      and every point of it fits the data exactly (not identifiable);
  (b) reading sigma_x and sigma_z, the cost is a bowl with a unique minimum
      (identifiable).  With 50 shots per expectation value, the fitted
      parameters scatter around the truth (practical identifiability).
Writes figures/ch04_valley.tex.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from skimage import measure
import needham as N
import ch04_observability as M

OMEGA, TRUE = 1.0, np.array([0.2, 0.1])
PREPS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)]
TS = np.linspace(0, 12, 49)
SHOTS, NDATA = 50, 60


def bloch(g1, gphi, r0, t):
    """Closed-form solution of the Bloch equations (Chapter 3), H = -(omega/2) sigma_z."""
    s = g1 / 2 + gphi
    c, sn = np.cos(OMEGA * t), np.sin(OMEGA * t)
    x = np.exp(-s * t) * (r0[0] * c + r0[1] * sn)
    y = np.exp(-s * t) * (-r0[0] * sn + r0[1] * c)
    z = 1 - (1 - r0[2]) * np.exp(-g1 * t)
    return x, y, z


def data(g1, gphi, readout):
    out = []
    for r0 in PREPS:
        x, y, z = bloch(g1, gphi, r0, TS)
        out.append(x)
        if readout == "xz":
            out.append(z)
    return np.concatenate(out)


def check_against_lindbladian():
    sm = np.array([[0, 1], [0, 0]], complex)
    L = M.lindbladian(-OMEGA / 2 * M.Z, [np.sqrt(TRUE[0]) * sm, np.sqrt(TRUE[1] / 2) * M.Z])
    basis = M.pauli_basis(1)
    A, beta = M.coherence_representation(L, basis)
    E = np.zeros((4, 4)); E[:3, :3] = A; E[:3, 3] = beta
    for r0 in PREPS:
        for t in (0.7, 3.1):
            v = expm(E * t) @ np.append(np.array(r0) / np.sqrt(2), 1.0)
            assert np.allclose(np.sqrt(2) * v[:3], bloch(*TRUE, r0, t), atol=1e-10)


def cost_grid(readout, g1s, gps):
    ref = data(*TRUE, readout)
    return np.array([[np.sum((data(a, c, readout) - ref) ** 2) for a in g1s] for c in gps])


def contours(J, g1s, gps, levels, Pl):
    out = []
    for lev in levels:
        for path in measure.find_contours(J, lev):
            pts = [(np.interp(cj, np.arange(len(g1s)), g1s), np.interp(ci, np.arange(len(gps)), gps)) for ci, cj in path]
            out.append(N.polyline([Pl.P(a, c) for a, c in pts], "line width=0.6pt, ndInk!55"))
    return out


def main():
    check_against_lindbladian()
    # (a) the cost depends on the rates only through gamma_1/2 + gamma_phi
    for d in (0.03, -0.05):
        assert np.allclose(data(TRUE[0] + 2 * d, TRUE[1] - d, "x"), data(*TRUE, "x"))
    assert not np.allclose(data(TRUE[0] + 0.06, TRUE[1] - 0.03, "xz"), data(*TRUE, "xz"))
    g1s, gps = np.linspace(0.0, 0.45, 151), np.linspace(0.0, 0.3, 121)
    Ja, Jb = cost_grid("x", g1s, gps), cost_grid("xz", g1s, gps)
    levels = [0.02, 0.1, 0.3, 0.8, 1.6]
    # (b) noisy data: fit (gamma_1, gamma_phi) many times
    rng = np.random.default_rng(4)
    clean = data(*TRUE, "xz")
    fits = []
    for _ in range(NDATA):
        noisy = 2 * rng.binomial(SHOTS, (1 + clean) / 2) / SHOTS - 1
        fits.append(least_squares(lambda p: data(p[0], p[1], "xz") - noisy, TRUE, bounds=([0, 0], [1, 1])).x)
    fits = np.array(fits)
    spread = np.std(fits, axis=0)
    assert np.linalg.norm(fits.mean(axis=0) - TRUE) < 0.02 and spread.max() < 0.03
    print("finite shots: mean estimate %s, std %s" % (np.round(fits.mean(axis=0), 4), np.round(spread, 4)))

    b = []
    for k, (J, title) in enumerate([(Ja, "(a) reading $\\sigma_x$ only"), (Jb, "(b) reading $\\sigma_x$ and $\\sigma_z$")]):
        Pl = N.Plot((0.0, 0.45), (0.0, 0.3), 5.4, 3.6, origin=(7.6 * k, 0.0))
        b += Pl.axes("$\\gamma_1$", "$\\gamma_\\phi$", xticks=[(0, "$0$"), (0.2, "$0.2$"), (0.4, "$0.4$")],
                     yticks=[(0.1, "$0.1$"), (0.2, "$0.2$"), (0.3, "$0.3$")])
        b += contours(J, g1s, gps, levels, Pl)
        if k == 0:
            g = np.array([0.0, 0.45]); line = 0.2 - g / 2
            b.append("\\draw[line width=1.8pt, ndRed] %s -- %s;" % (Pl.pt(g[0], line[0]), Pl.pt(0.4, 0.0)))
            b.append("\\node[note, anchor=south west, text=ndRed] (v) at %s {a flat valley: every point\\\\ on $\\gamma_1/2+\\gamma_\\phi=1/T_2$\\\\ fits the data exactly};" % Pl.pt(0.2, 0.2))
            b.append("\\draw[pointer] (v.south west) to[bend right=15] %s;" % Pl.pt(0.12, 0.2 - 0.06 - 0.005))
        else:
            for p in fits:
                b.append("\\fill[ndBlue, opacity=0.7] %s circle (1.1pt);" % Pl.pt(*p))
            b.append("\\node[note, anchor=south west, text=ndBlue] (f) at %s {fits to data with\\\\ %d shots per value};" % (Pl.pt(0.26, 0.19), SHOTS))
            b.append("\\draw[pointer] (f.south west) to[bend right=15] %s;" % Pl.pt(TRUE[0] + 0.012, TRUE[1] + 0.012))
            b.append("\\node[note, anchor=north west] at %s {a bowl: a unique\\\\ minimum};" % Pl.pt(0.29, 0.085))
        b.append("\\node[dot, fill=ndInk, minimum size=4.4pt] at %s {};" % Pl.pt(*TRUE))
        b.append("\\node[font=\\small] at %s {%s};" % (N.pt(Pl.P(0.225, 0.0) + np.array([0, -0.9])), title))
    N.write("ch04_valley", b)
    print("written figures/ch04_valley.tex")


if __name__ == "__main__":
    main()
