"""ch07_defect.py -- entanglement makes a part mixed.

The qubit with a hidden defect of Chapter 4: the qubit S exchanges energy with
a two-level defect T, which decays into a bath at the rate kappa,
    H = -(omega/2)(Z_S + Z_T) + g (s+_S s-_T + s-_S s+_T),    L = sqrt(kappa) s-_T,
with g = 1, kappa = 0.8.  The qubit starts in |+> = (|0> + |1>)/sqrt 2 and the
defect in |0>.  In the frame rotating at omega the amplitude c(t) of |1>_S|0>_T
is real,
    c(t) = e^{-kappa t/4} (cos W t + kappa/(4W) sin W t),   W = sqrt(g^2 - kappa^2/16),
and the reduced Bloch vector of the qubit is r_S = (c, 0, 1 - c^2): it runs back
and forth along the parabola z = 1 - x^2 inside the Bloch disc.  It is pure
on the rim (t = 0) and at the pole |0> (c = 0, the excitation is in the defect),
and mixed in between, first through its entanglement with the defect (the
concurrence C of the joint state peaks), later through the excitation it has
lost to the bath (C vanishes, but |r_S| < 1).  For a pure joint state of two
qubits, |r_S|^2 = 1 - C^2.

The joint state is integrated with the vectorised GKSL generator; the module
also provides the helpers used by ch07_contraction.py.
Writes figures/ch07_defect.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N
import ch04_examples as E                    # OMEGA = 1, G = 1, KAPPA = 0.8, SM = |0><1|
from ch03_decoherence import lindbladian, vec, unvec

I2, X, Y, Z = T.pauli()
SM = E.SM
SP = SM.conj().T


def defect_generator(omega=E.OMEGA, g=E.G, kappa=E.KAPPA):
    """Vectorised GKSL generator of qubit (first factor) and defect (second factor)."""
    k1 = lambda P: np.kron(P, I2)
    k2 = lambda P: np.kron(I2, P)
    H = -omega / 2 * (k1(Z) + k2(Z)) + g * (k1(SP) @ k2(SM) + k1(SM) @ k2(SP))
    return lindbladian(H, [np.sqrt(kappa) * k2(SM)])


def evolve(L, rho0, ts):
    """Joint states at the equally spaced times ts."""
    P = expm(L * (ts[1] - ts[0]))
    v, out = vec(rho0), []
    for _ in ts:
        out.append(unvec(v, 4))
        v = P @ v
    return out


def reduced(rho):
    """Partial trace over the defect."""
    return np.einsum("ajbj->ab", rho.reshape(2, 2, 2, 2))


def concurrence(rho):
    """Wootters' concurrence of a two-qubit state."""
    YY = np.kron(Y, Y)
    ev = np.linalg.eigvals(rho @ YY @ rho.conj() @ YY)
    s = np.sort(np.sqrt(np.clip(np.real(ev), 0, None)))[::-1]
    return max(0.0, s[0] - s[1] - s[2] - s[3])


def amplitude(t, g=E.G, kappa=E.KAPPA):
    """c(t), the amplitude left in the qubit, in the rotating frame."""
    W = np.sqrt(g ** 2 - kappa ** 2 / 16)
    return np.exp(-kappa * t / 4) * (np.cos(W * t) + kappa / (4 * W) * np.sin(W * t))


def main():
    ts = np.linspace(0, 10, 1001)
    psi = np.kron(np.array([1, 1]) / np.sqrt(2), [1, 0]).astype(complex)
    rho0 = np.outer(psi, psi.conj())
    rot = evolve(defect_generator(omega=0.0), rho0, ts)          # the frame rotating at omega
    lab = evolve(defect_generator(), rho0, ts)
    r = np.array([T.bloch_vector(reduced(rho)) for rho in rot])
    C = np.array([concurrence(rho) for rho in rot])
    c = amplitude(ts)
    assert np.allclose(r[:, 0], c, atol=1e-10) and np.allclose(r[:, 2], 1 - c ** 2, atol=1e-10)
    assert np.allclose(r[:, 1], 0, atol=1e-12)                     # y = 0: the parabola z = 1 - x^2
    # the lab frame differs by the precession about z at omega, and nothing else
    rl = np.array([T.bloch_vector(reduced(rho)) for rho in lab])
    ph = -E.OMEGA * ts
    turned = np.stack([np.cos(ph) * r[:, 0] - np.sin(ph) * r[:, 1], np.sin(ph) * r[:, 0] + np.cos(ph) * r[:, 1], r[:, 2]], 1)
    assert np.allclose(rl, turned, atol=1e-9)
    assert np.allclose(C, [concurrence(rho) for rho in lab], atol=1e-8)
    # a pure joint state (no bath): |r_S|^2 = 1 - C^2
    pure = evolve(defect_generator(omega=0.0, kappa=0.0), rho0, ts[::10])
    rp = np.array([T.bloch_vector(reduced(rho)) for rho in pure])
    assert np.allclose(np.sum(rp ** 2, 1), 1 - np.array([concurrence(rho) for rho in pure]) ** 2, atol=1e-9)
    nr = np.linalg.norm(r, axis=1)
    zeros = ts[1:][np.diff(np.sign(c)) != 0]
    turn = [i for i in range(1, len(ts) - 1) if abs(c[i]) > abs(c[i - 1]) and abs(c[i]) >= abs(c[i + 1])]
    i1 = int(np.argmin(nr[:200]))                                  # the most mixed moment of the first sweep
    iC = int(np.argmax(C))
    print("qubit back at |0> (c = 0) at t = %s" % np.round(zeros, 2))
    print("turning points t = %s, c = %s" % (np.round(ts[turn], 2), np.round(c[turn], 3)))
    print("first sweep: min |r_S| = %.3f at t = %.2f; max C = %.3f at t = %.2f" % (nr[i1], ts[i1], C[iC], ts[iC]))
    print("at the first turning point t = %.2f: |r_S| = %.3f, C = %.4f" % (ts[turn[0]], nr[turn[0]], C[turn[0]]))
    assert abs(ts[i1] - ts[iC]) < 0.1 and C[turn[0]] < 0.01 and nr[turn[0]] < 0.9

    b = []
    # ---- (a) the (x, z) disc
    R = 2.15
    P = lambda x, z: (R * x, R * z)
    b.append("\\fill[ndSky!45] (0,0) circle (%.3f);" % R)
    b.append("\\draw[ink] (0,0) circle (%.3f);" % R)
    b.append("\\draw[faint, ->, >=stealth] (%.3f,0) -- (%.3f,0) node[right, text=ndInk] {$x$};" % (-R - 0.25, R + 0.35))
    b.append("\\draw[faint, ->, >=stealth] (0,%.3f) -- (0,%.3f) node[above, text=ndInk] {$z$};" % (-R - 0.25, R + 0.45))
    for (x, z), lab, anc in [((0, 1), "\\ket0", "south east"), ((0, -1), "\\ket1", "north east"),
                              ((1, 0), "\\ket+", "north west"), ((-1, 0), "\\ket-", "north east")]:
        b.append("\\node[font=\\small, anchor=%s, inner sep=2pt] at %s {$%s$};" % (anc, N.pt(P(x, z)), lab))
    xs = np.linspace(-1, 1, 121)
    b.append(N.polyline([P(x, 1 - x ** 2) for x in xs], "dashedink"))
    # the first sweep, |+> to |0>, then the rest of the path (it retraces the same arc)
    first = ts <= zeros[0]
    b.append(N.polyline([P(*q) for q in r[~first][:, [0, 2]]], "line width=1.0pt, ndBlue!70"))
    b.append(N.polyline([P(*q) for q in r[first][:, [0, 2]]], "line width=1.6pt, ndBlue"))
    k = int(0.55 * np.sum(first))
    b.append("\\draw[->, >=stealth, line width=1.6pt, ndBlue] %s -- %s;" % (N.pt(P(r[k, 0], r[k, 2])), N.pt(P(r[k + 3, 0], r[k + 3, 2]))))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % N.pt(P(1, 0)))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % N.pt(P(0, 1)))
    for i in turn[:2]:
        b.append("\\node[dot, fill=ndBlue!70, minimum size=2.8pt] at %s {};" % N.pt(P(r[i, 0], r[i, 2])))
    i = turn[0]
    b.append("\\node[font=\\scriptsize, anchor=north east, text=ndBlue] at %s {$t=%.1f$};" % (N.pt(P(r[i, 0], r[i, 2])), ts[i]))
    i = turn[1]
    b.append("\\node[font=\\scriptsize, anchor=south west, text=ndBlue, inner sep=1pt] at %s {$%.1f$};" % (N.pt(P(r[i, 0] + 0.03, r[i, 2] + 0.07)), ts[i]))
    b.append("\\node[font=\\scriptsize, anchor=south west, text=ndBlue] at %s {$t=0$};" % N.pt(P(1.0, 0.02)))
    b.append("\\node[dot, fill=ndRed, minimum size=2.8pt] at %s {};" % N.pt(P(r[i1, 0], r[i1, 2])))
    b.append("\\node[note, anchor=west, text=ndRed] (m) at %s {most mixed,\\\\ $t=%.1f$};" % (N.pt(P(1.0, 0.6)), ts[i1]))
    b.append("\\draw[pointer] (m.west) to[bend right=20] %s;" % N.pt(P(r[i1, 0] + 0.03, r[i1, 2] + 0.03)))
    b.append("\\node[note, anchor=east] (p) at %s {pure again at $\\ket0$:\\\\ the excitation is\\\\ in the defect};" % N.pt(P(-0.62, 1.32)))
    b.append("\\draw[pointer] (p.east) to[bend left=15] %s;" % N.pt(P(-0.03, 1.0)))
    b.append("\\node[note, anchor=north] at %s {inside the ball the\\\\ qubit is mixed:\\\\ correlated with the\\\\ defect or the bath};" % N.pt(P(-0.05, -0.18)))
    b.append("\\node[note, anchor=east] (k) at %s {track:\\\\ $z=1-x^2$};" % N.pt(P(-1.04, 0.62)))
    b.append("\\draw[pointer] (k.south east) to[bend right=15] %s;" % N.pt(P(-0.8, 1 - 0.64 - 0.03)))
    b.append("\\node[font=\\small] at (0,%.2f) {(a) the qubit's Bloch vector, frame rotating at $\\omega$};" % (-R - 0.8))

    # ---- (b) against time
    Pl = N.Plot((0, 10), (0, 1.0), 6.4, 3.9, origin=(4.45, -2.0))
    b += Pl.axes("$t$", "", xticks=[(k, "$%d$" % k) for k in (2, 4, 6, 8, 10)],
                 yticks=[(0.5, "$\\tfrac12$"), (1, "$1$")])
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(0, 1), Pl.pt(10, 1)))
    b.append(Pl.curve(ts, np.sqrt(np.clip(1 - nr ** 2, 0, None)), "line width=0.8pt, ndRed, dash pattern=on 3pt off 2pt"))
    b.append(Pl.curve(ts, C, "line width=1.4pt, ndRed"))
    b.append(Pl.curve(ts, nr, "line width=1.4pt, ndBlue"))
    for z0 in zeros[:2]:
        b.append("\\draw[ndBlue, line width=0.4pt, densely dotted] %s -- %s;" % (Pl.pt(z0, 0), Pl.pt(z0, 1)))
    b.append("\\node[font=\\small, text=ndBlue, anchor=south] at %s {$\\abs{\\Bloch_S}$};" % Pl.pt(8.6, 1.0))
    b.append("\\node[note, anchor=west, text=ndRed] (c) at %s {$C$: entanglement\\\\ of qubit and defect};" % Pl.pt(1.5, 0.66))
    b.append("\\draw[pointer] (c.west) to[bend right=25] %s;" % Pl.pt(ts[iC] + 0.12, C[iC] + 0.01))
    it = turn[0]
    b.append("\\node[note, anchor=north west, text=ndRed!80!ndInk] (d) at %s {$\\sqrt{1-\\abs{\\Bloch_S}^2}$, which\\\\ is $C$ if qubit and\\\\ defect are alone};" % Pl.pt(6.15, 0.79))
    b.append("\\draw[pointer] (d.south west) to[bend right=20] %s;" % Pl.pt(ts[turn[1]] + 0.1, np.sqrt(1 - nr[turn[1]] ** 2) + 0.02))
    for z0 in zeros[:2]:
        b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=south] at %s {$\\ket0$};" % Pl.pt(z0, 1.01))
    b.append("\\node[font=\\small] at %s {(b) the length of $\\Bloch_S$, and the concurrence $C$};"
             % N.pt(Pl.P(5, 0) + np.array([0, -1.05])))
    N.write("ch07_defect", b)
    print("written figures/ch07_defect.tex")


if __name__ == "__main__":
    main()
