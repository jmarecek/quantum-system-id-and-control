"""ch04_examples.py -- two first examples of bilinear systems.

(1) A driven qubit, nothing hidden: H(t) = -(omega/2) sigma_z + (u(t)/2) sigma_x.
    The Bloch vector obeys r' = (A + u N) r with
        A = [[0, omega, 0], [-omega, 0, 0], [0, 0, 0]],
        N = [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
    both antisymmetric (rotations).  Left: trajectories from |0> for constant
    controls.  Right: <sigma_z> after a resonant pulse of area uT is cos(uT),
    not linear in u.  Writes figures/ch04_driven.tex.
(2) A qubit with a hidden defect: the qubit S exchanges energy with a two-level
    defect T, which decays into a bath at the rate kappa,
        H = -(omega/2)(Z_S + Z_T) + g (s+_S s-_T + s-_S s+_T) + (u/2) X_S,
        L = sqrt(kappa) s-_T.
    Only the qubit is measured: 3 of the 15 components of the coherence vector
    are seen, 12 are hidden.  Starting from |1>_S |0>_T, the excited population
    of the qubit revives as the excitation returns from the defect, which no
    Markovian model of the qubit alone can produce.  Writes figures/ch04_hidden.tex.
"""
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
import needham as N
import ch04_observability as M

I2, X, Y, Z = M.I2, M.X, M.Y, M.Z
SM = np.array([[0, 1], [0, 0]], complex)                 # sigma_- = |0><1|
SP = SM.conj().T
OMEGA, G, KAPPA = 1.0, 1.0, 0.8
NAMES = ["".join(p) for p in itertools.product("IXYZ", repeat=2)][1:]


def driven():
    b2 = M.pauli_basis(1)
    A, beta0 = M.coherence_representation(M.lindbladian(-OMEGA / 2 * Z, []), b2)
    Nx, beta1 = M.coherence_representation(M.lindbladian(0.5 * X, []), b2)
    assert np.allclose(A, [[0, OMEGA, 0], [-OMEGA, 0, 0], [0, 0, 0]]) and np.allclose(Nx, [[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.allclose(A, -A.T) and np.allclose(Nx, -Nx.T) and np.allclose(beta0, 0) and np.allclose(beta1, 0)

    b = []
    V = N.View(az=-60.0, el=18.0, R=2.0)
    b += V.sphere()
    b += V.great_circle([0, 0, 1])
    for v, lab, pos in [((0, 0, 1.25), "\\ket0", "above"), ((1.45, 0, 0), "x", "right")]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s, text=ndInk] {$%s$};" % (V.pt((0, 0, 0)), V.pt(v), pos, lab))
    cols = ["ndGold", "ndGreen", "ndRed"]
    for u, col in zip((0.6, 1.5, 4.0), cols):
        G_ = A + u * Nx
        h = np.hypot(OMEGA, u)
        ts = np.linspace(0, 2 * np.pi / h, 240)
        tr = np.array([expm(G_ * t) @ np.array([0, 0, 1.0]) for t in ts])
        assert np.allclose(np.linalg.norm(tr, axis=1), 1)                # nothing is lost
        b += V.curve(tr, "line width=1.3pt, %s" % col, "line width=0.6pt, %s, densely dotted" % col)
        front = tr[tr @ V.d >= 0]
        left = front[np.argmin([V.P(p)[0] for p in front])]
        b.append("\\node[font=\\scriptsize, text=%s, anchor=east] at %s {$u=%.1f$};" % (col, V.pt(left), u))
    b.append("\\node[dot, minimum size=4.2pt] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[note, anchor=west] at (2.7,1.5) {constant control $u$: a\\\\ rotation about a tilted axis;\\\\ every path stays on the\\\\ sphere, nothing is hidden};")
    b.append("\\node[font=\\small] at (0,-2.75) {(a) trajectories from $\\ket0$, $\\omega=%d$};" % OMEGA)
    # (b) the response is not linear in the input
    P = N.Plot((0, 3 * np.pi), (-1.1, 1.1), 5.6, 3.6, origin=(8.0, -1.8))
    b += P.axes("pulse area $uT$", "$\\langle\\sigma_z\\rangle$ after the pulse",
                xticks=[(np.pi, "$\\pi$"), (2 * np.pi, "$2\\pi$"), (3 * np.pi, "$3\\pi$")], yticks=[(-1, "$-1$"), (1, "$1$")], yzero=0.0)
    a = np.linspace(0, 3 * np.pi, 200)
    zf = np.array([(expm(Nx * ai) @ np.array([0, 0, 1.0]))[2] for ai in a])     # resonant drive (omega = 0), area ai
    assert np.allclose(zf, np.cos(a))
    b.append(P.curve(a, zf, "line width=1.5pt, ndBlue"))
    lin = a[a <= 1.04 * np.pi]
    b.append(P.curve(lin, 1 - 2 * lin / np.pi, "dashedink, line width=0.8pt"))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % P.pt(np.pi, -1))
    b.append("\\node[note, anchor=north] at %s {$\\pi$: flipped};" % P.pt(np.pi, -1.12))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % P.pt(2 * np.pi, 1))
    b.append("\\node[note, anchor=south] at %s {$2\\pi$: back to $\\ket0$};" % P.pt(2 * np.pi, 1.08))
    b.append("\\node[font=\\small] at %s {(b) twice the input is not twice the effect};" % N.pt(P.P(1.5 * np.pi, -1.1) + np.array([0, -0.75])))
    N.write("ch04_driven", b)
    print("written figures/ch04_driven.tex")


def hidden():
    k1 = lambda P: np.kron(P, I2)
    k2 = lambda P: np.kron(I2, P)
    H = -OMEGA / 2 * (k1(Z) + k2(Z)) + G * (k1(SP) @ k2(SM) + k1(SM) @ k2(SP))
    b4 = M.pauli_basis(2)
    A, beta = M.coherence_representation(M.lindbladian(H, [np.sqrt(KAPPA) * k2(SM)]), b4)
    Nu, _ = M.coherence_representation(M.lindbladian(0.5 * k1(X), []), b4)
    E = np.zeros((16, 16)); E[:15, :15] = A; E[:15, 15] = beta
    Ne = np.zeros((16, 16)); Ne[:15, :15] = Nu
    idx = {n: i for i, n in enumerate(NAMES)}
    seen = ["XI", "YI", "ZI"]
    C = np.zeros((3, 16))
    for r, n in enumerate(seen):
        C[r, idx[n]] = 1.0
    rank0 = np.linalg.matrix_rank(M.observability_matrix(E, C), 1e-9)
    basis = np.linalg.svd(C)[2][:np.linalg.matrix_rank(C)]       # rows: span of C W, W a word in {A, N}
    while True:
        grown = np.vstack([basis, basis @ E, basis @ Ne])
        r = np.linalg.matrix_rank(grown, 1e-9)
        if r == basis.shape[0]:
            break
        basis = np.linalg.svd(grown)[2][:r]
    rank_u = basis.shape[0]
    print("ranks:", rank0, rank_u)
    assert rank0 == 8 and rank_u == 11
    # the qubit's excited population from |1>_S |0>_T, u = 0
    psi = np.kron([0, 1], [1, 0]).astype(complex)
    rho = np.outer(psi, psi.conj())
    x0 = np.append([np.real(np.trace(F @ rho)) for F in b4], 1.0)
    ts = np.linspace(0, 14, 500)
    vs = np.array([expm(E * t) @ x0 for t in ts])
    pe_s = (1 - 2 * vs[:, idx["ZI"]]) / 2
    pe_t = (1 - 2 * vs[:, idx["IZ"]]) / 2
    om = np.sqrt(G ** 2 - KAPPA ** 2 / 16)
    cs = np.exp(-KAPPA * ts / 4) * (np.cos(om * ts) + KAPPA / (4 * om) * np.sin(om * ts))
    assert np.allclose(pe_s, np.abs(cs) ** 2, atol=1e-9)              # exact single-excitation solution
    assert np.any(np.diff(pe_s) > 1e-3)                                 # revivals: not Markovian for the qubit alone
    fit = minimize_scalar(lambda gam: np.sum((np.exp(-gam * ts) - pe_s) ** 2), bounds=(0.01, 5), method="bounded")
    print("observable dimension through the qubit: %d (u = 0), %d (with control); best Markov rate %.3f" % (rank0, rank_u, fit.x))

    b = []
    # (a) schematic
    b.append("\\node[circle, draw=ndInk, line width=1.0pt, fill=ndSky, minimum size=1.05cm] (S) at (0,0.6) {qubit};")
    b.append("\\node[circle, draw=ndInk!60, line width=0.9pt, dash pattern=on 3pt off 2pt, fill=ndInk!8, minimum size=1.05cm, font=\\small] (T) at (2.6,0.6) {defect};")
    b.append("\\draw[<->, >=stealth, line width=1.0pt, ndInk] (S) -- node[above, font=\\small] {$g$} (T);")
    w = np.linspace(0, 1.1, 40)
    b.append(N.polyline([(3.15 + x, 0.6 + 0.1 * np.sin(14 * x)) for x in w], "->, >=stealth, line width=0.9pt, ndRed"))
    b.append("\\node[font=\\small, text=ndRed, above] at (3.7,0.75) {$\\kappa$};")
    b.append("\\node[note, anchor=west] at (4.3,0.6) {bath};")
    b.append("\\draw[->, >=stealth, line width=1.0pt, ndRose!70!ndRed] (-1.6,1.45) node[left, text=ndInk, font=\\small] {$u(t)$} -- (S.north west);")
    b.append("\\draw[->, >=stealth, line width=1.0pt, ndRose!70!ndRed] (S.south west) -- (-1.6,-0.25) node[left, text=ndInk, font=\\small] {$\\vec y(t)$};")
    b.append("\\node[note, anchor=north] at (0,-0.15) {driven and measured};")
    b.append("\\node[note, anchor=north] at (2.6,-0.15) {neither driven\\\\ nor measured};")
    # the state vector: 3 components seen, 12 hidden
    cw = 0.36
    order = seen + [n for n in NAMES if n not in seen]
    for i, n in enumerate(order):
        x = -0.9 + i * cw
        fill = "ndSky" if n in seen else "ndInk!12"
        b.append("\\filldraw[fill=%s, draw=ndInk!70, line width=0.5pt] (%.3f,-1.75) rectangle (%.3f,-2.11);" % (fill, x, x + cw))
        b.append("\\node[font=\\tiny, rotate=90, anchor=east] at (%.3f,-2.16) {$%s$};" % (x + cw / 2, n))
    b.append("\\node[note, anchor=south] at (%.3f,-1.72) {seen: $3$};" % (-0.9 + 1.5 * cw))
    b.append("\\node[note, anchor=south] at (%.3f,-1.72) {hidden: $12$};" % (-0.9 + 9 * cw))
    b.append("\\node[font=\\small] at (1.8,-3.25) {(a) the coherence vector of qubit and defect};")
    # (b) the qubit's excited population
    P = N.Plot((0, 14), (0, 1.05), 6.4, 3.6, origin=(7.2, -2.2))
    b += P.axes("$t$", "excited population", yticks=[(0, "$0$"), (1, "$1$")])
    b.append(P.curve(ts, pe_t, "line width=1.0pt, ndInk!55, dash pattern=on 3pt off 2pt"))
    b.append(P.curve(ts, np.exp(-fit.x * ts), "line width=1.0pt, ndRed, dash pattern=on 1.5pt off 1.5pt"))
    b.append(P.curve(ts, pe_s, "line width=1.6pt, ndBlue"))
    i_rev = np.argmax(pe_s * (ts > 2.5) * (ts < 5.5))
    b.append("\\node[note, anchor=west, text=ndBlue] (r) at %s {qubit: the excitation\\\\ comes back};" % P.pt(7.6, 0.6))
    b.append("\\draw[pointer] (r.west) to[bend right=15] %s;" % P.pt(ts[i_rev] + 0.15, pe_s[i_rev] + 0.02))
    i_t = np.argmax(pe_t)
    b.append("\\node[note, anchor=west, text=ndInk!70] (d) at %s {defect (hidden)};" % P.pt(7.6, 0.33))
    b.append("\\draw[pointer] (d.west) to[bend right=15] %s;" % P.pt(ts[i_t] + 0.3, pe_t[i_t] - 0.02))
    b.append("\\node[note, anchor=west, text=ndRed] (m) at %s {best fit without\\\\ a hidden state};" % P.pt(7.6, 0.9))
    b.append("\\draw[pointer] (m.west) to[bend right=12] %s;" % P.pt(1.3, np.exp(-fit.x * 1.3) + 0.02))
    b.append("\\node[font=\\small] at %s {(b) starting from $\\ket1$ in the qubit, $u=0$};" % N.pt(P.P(7, 0) + np.array([0, -0.8])))
    N.write("ch04_hidden", b)
    print("written figures/ch04_hidden.tex")


if __name__ == "__main__":
    driven()
    hidden()
