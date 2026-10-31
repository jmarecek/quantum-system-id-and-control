"""ch07_contraction.py -- the trace distance never grows under a Markovian
evolution, but it can revive when part of the system is hidden.

(a) The damped qubit of Chapter 5 (omega = 3, gamma1 = 0.2, gphi = 0.4), started
    from X0 (60 degrees from |1>, in the x-z plane) and from |1>.  Both
    trajectories end at |0>.  The affine term of the Bloch equation cancels in
    the difference, so r1(t) - r2(t) = e^{A t}(r1 - r2), and since the symmetric
    part of A is diag(-Gamma, -Gamma, -gamma1) the chord between the two states
    only shrinks: D(t) = |r1(t) - r2(t)|/2 never increases.
(b) D(t) for that pair, and for the qubit with the hidden defect of Chapter 4
    (g = 1, kappa = 0.8) started from |1> and from |0>, the defect in |0>.  There
    D(t) = |c(t)|^2 with c(t) = e^{-kappa t/4}(cos W t + kappa/(4W) sin W t),
    W = sqrt(g^2 - kappa^2/16): it falls to zero and revives, which no
    CP-divisible (Markovian) evolution of the qubit alone can do.
Writes figures/ch07_contraction.tex.
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N
import ch05_spectrum as Q                    # qubit_A, X0, OMEGA = 3, GAMMA1 = 0.2, GPHI = 0.4
import ch07_defect as Dfc                    # the qubit with the hidden defect


def main():
    A = Q.qubit_A()
    ts = np.linspace(0, 10, 1001)
    one = np.array([0.0, 0.0, -1.0, 1.0])
    E = [expm(A * t) for t in ts]
    tr1 = np.array([M @ Q.X0 for M in E])[:, :3]
    tr2 = np.array([M @ one for M in E])[:, :3]
    D = 0.5 * np.linalg.norm(tr1 - tr2, axis=1)
    assert np.allclose(D, [0.5 * np.linalg.norm((M @ (Q.X0 - one))[:3]) for M in E])   # affine term cancels
    assert np.all(np.diff(D) <= 1e-12)
    S = 0.5 * (A[:3, :3] + A[:3, :3].T)
    assert np.allclose(S, np.diag([-Q.GAMMA, -Q.GAMMA, -Q.GAMMA1]))                   # why it never grows
    rng = np.random.default_rng(7)
    for _ in range(20):                                                               # any two initial states
        a, b_ = rng.normal(size=(2, 3))
        a *= rng.uniform() ** (1 / 3) / np.linalg.norm(a)
        b_ *= rng.uniform() ** (1 / 3) / np.linalg.norm(b_)
        d = [np.linalg.norm((M @ np.append(a - b_, 0.0))[:3]) for M in E[::10]]
        assert np.all(np.diff(d) <= 1e-12)
    # the hidden defect: |1> versus |0> for the qubit, the defect in |0>
    L = Dfc.defect_generator()
    rs = []
    for q in ([0, 1], [1, 0]):
        psi = np.kron(q, [1, 0]).astype(complex)
        rs.append(np.array([T.bloch_vector(Dfc.reduced(rho)) for rho in Dfc.evolve(L, np.outer(psi, psi.conj()), ts)]))
    Dd = 0.5 * np.linalg.norm(rs[0] - rs[1], axis=1)
    assert np.allclose(Dd, Dfc.amplitude(ts) ** 2, atol=1e-10)
    assert np.any(np.diff(Dd) > 1e-3)                                                 # it revives
    i0 = int(np.argmax(Dd < 1e-4))
    ir = i0 + int(np.argmax(Dd[i0:i0 + 300]))
    print("Chapter 5 pair: D(0) = %.3f, D(2) = %.3f, D(5) = %.3f, never increasing" % (D[0], D[200], D[500]))
    print("hidden defect: D(0) = %.3f, D = 0 at t = %.2f, revives to %.3f at t = %.2f" % (Dd[0], ts[i0], Dd[ir], ts[ir]))

    b = []
    # ---- (a) the two trajectories and the chord between them
    V = N.View(az=28.0, el=16.0, R=2.15)
    b += V.sphere()
    b += V.great_circle([0, 0, 1])
    b.append("\\draw[faint] %s -- %s;" % (V.pt((0, 0, -1)), V.pt((0, 0, 1))))
    for v, lab, pos in [((0, 0, 1.22), "\\ket0", "above"), ((1.38, 0, 0), "x", "below left"), ((0, 1.32, 0), "y", "right")]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s, text=ndInk] {$%s$};" % (V.pt((0, 0, 1) if lab == "\\ket0" else (0, 0, 0)), V.pt(v), pos, lab))
    b.append("\\node[below, text=ndInk] at %s {$\\ket1$};" % V.pt((0, 0, -1.04)))
    b.append(N.polyline([V.P(p) for p in tr1], "line width=0.9pt, ndInk!70"))
    b.append(N.polyline([V.P(p) for p in tr2], "line width=1.3pt, ndGold"))
    for t in (0.0, 0.6, 1.5, 3.0):
        i = int(round(t / (ts[1] - ts[0])))
        b.append("\\draw[line width=1.5pt, ndBlue] %s -- %s;" % (V.pt(tr1[i]), V.pt(tr2[i])))
        b.append("\\node[dot, fill=ndBlue, minimum size=2.8pt] at %s {};" % V.pt(tr1[i]))
        b.append("\\node[dot, fill=ndBlue, minimum size=2.8pt] at %s {};" % V.pt(tr2[i]))
    b.append("\\node[dot, fill=ndInk, minimum size=4.2pt] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[note, anchor=east] (x) at (-1.55,-1.75) {from $X_0$\\\\ at $t=0$};")
    b.append("\\draw[pointer] (x.east) to[bend left=15] %s;" % V.pt(tr1[0] + np.array([0.0, -0.04, 0.03])))
    b.append("\\node[note, anchor=west, text=ndGold!80!ndInk] (o) at (2.35,-1.3) {from $\\ket1$, up\\\\ the $z$-axis};")
    b.append("\\draw[pointer] (o.west) to[bend left=12] %s;" % V.pt(tr2[12] + np.array([0.0, 0.03, 0.0])))
    b.append("\\node[note, anchor=west, text=ndBlue] (c) at (2.35,0.95) {the chord between\\\\ the two states\\\\ only shrinks};")
    b.append("\\draw[pointer] (c.west) to[bend right=12] %s;" % V.pt(0.5 * (tr1[60] + tr2[60])))
    b.append("\\node[font=\\small] at (0.4,-2.85) {(a) the damped qubit, from $X_0$ and from $\\ket1$};")

    # ---- (b) the trace distance against time
    Pl = N.Plot((0, 10), (0, 1.0), 6.3, 3.9, origin=(5.6, -2.1))
    b += Pl.axes("$t$", "$D(t)$", xticks=[(k, "$%d$" % k) for k in (2, 4, 6, 8, 10)],
                 yticks=[(0.5, "$\\tfrac12$"), (1, "$1$")])
    b.append(Pl.curve(ts, Dd, "line width=1.4pt, ndRed"))
    b.append(Pl.curve(ts, D, "line width=1.4pt, ndBlue"))
    b.append("\\node[dot, fill=ndRed] at %s {};" % Pl.pt(ts[ir], Dd[ir]))
    b.append("\\node[note, anchor=west, text=ndBlue] (m) at %s {damped qubit: Markovian,\\\\ $D$ never grows};" % Pl.pt(4.6, 0.53))
    b.append("\\draw[pointer] (m.south) to[bend left=15] %s;" % Pl.pt(7.2, D[720] + 0.02))
    b.append("\\node[note, anchor=south west, text=ndRed] (r) at %s {qubit with a hidden defect:\\\\ $D$ revives, information\\\\ comes back from the defect};" % Pl.pt(1.6, 0.73))
    b.append("\\draw[pointer] (r.south) to[bend right=10] %s;" % Pl.pt(ts[ir] + 0.05, Dd[ir] + 0.03))
    b.append("\\node[font=\\small] at %s {(b) the trace distance between the two states};"
             % N.pt(Pl.P(5, 0) + np.array([0, -0.75])))
    N.write("ch07_contraction", b)
    print("written figures/ch07_contraction.tex")


if __name__ == "__main__":
    main()
