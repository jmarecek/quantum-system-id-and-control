"""ch08_diamond.py -- the diamond distance: why the ancilla matters, and why
the worst case can be far from the average case.

(a) The qubit Werner--Holevo pair
        Phi_+(rho) = (Tr(rho) 1 + rho^T)/3,     Phi_-(rho) = Tr(rho) 1 - rho^T = Y rho Y.
    On Bloch vectors Phi_+ is r -> (x, -y, z)/3 and Phi_- the rotation by pi
    about y.  Fed a single qubit, the two outputs are at trace distance at
    most 2/3, attained by every pure input.  Fed one half of the maximally
    entangled state |Phi+>, they return Pi_sym/3 and the singlet, which are
    orthogonal: with an ancilla the channels are perfectly distinguishable,
    and the diamond distance is 1.
(b) For a qubit, the diamond distance (1/2)||Phi - id|| from the identity
    against the average gate infidelity r = 1 - F_avg, F_avg = (N F_pro + 1)/(N + 1):
    a coherent over-rotation exp(-i theta Z/2) has (1/2)||.|| = sin(theta/2) = sqrt(3r/2),
    a stochastic Z error of probability p has (1/2)||.|| = p = 3r/2.  The band
    is that of the general bounds (N+1) r/N <= (1/2)||.|| <= N sqrt((N+1) r/N).

The diamond distances are computed in closed form and confirmed by a brute
maximisation of the trace distance of ((Phi - Psi) (x) id)(psi psi^dagger) over
pure two-qubit inputs; every such input is (sqrt(rho) (x) 1)|Omega> up to a
unitary on the ancilla, so the search runs over the Bloch ball of rho.
Writes figures/ch08_diamond.tex.
"""
import numpy as np
from scipy.optimize import minimize
import needham as N

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
OMEGA = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)          # |Phi+>, system first


def wh_plus(A):
    return (np.trace(A) * I2 + A.T) / 3


def wh_minus(A):
    return np.trace(A) * I2 - A.T


def rotation(theta):
    U = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
    return lambda A: U @ A @ U.conj().T


def zflip(p):
    return lambda A: (1 - p) * A + p * Z @ A @ Z


def identity(A):
    return A


def superop(f):
    """The 2x2x2x2 array S with f(A)[s, t] = sum_uv S[s, t, u, v] A[u, v]."""
    S = np.zeros((2, 2, 2, 2), dtype=complex)
    for u in range(2):
        for v in range(2):
            E = np.zeros((2, 2), dtype=complex)
            E[u, v] = 1
            S[:, :, u, v] = f(E)
    return S


def extend(f, M, S=None):
    """(f (x) id)(M) for a 4x4 matrix M on system (x) ancilla."""
    S = superop(f) if S is None else S
    T = M.reshape(2, 2, 2, 2)                                       # [s, a, s', a']
    return np.einsum("stuv,uavb->satb", S, T).reshape(4, 4)


def tdist(A, B):
    return 0.5 * np.abs(np.linalg.eigvalsh(A - B)).sum()


def bloch_state(r):
    return 0.5 * (I2 + r[0] * X + r[1] * Y + r[2] * Z)


def probe(r):
    """The pure input (sqrt(rho) (x) 1)|Omega> sqrt(2) whose system marginal has Bloch vector r."""
    w, V = np.linalg.eigh(bloch_state(r))
    s = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.conj().T
    psi = np.kron(s, I2) @ OMEGA * np.sqrt(2)
    return np.outer(psi, psi.conj())


def diamond_brute(f, g, starts=3, maxiter=600):
    """max over pure two-qubit inputs of D(((f - g) (x) id)(psi)), by search over the Bloch ball."""
    Sd = superop(f) - superop(g)
    val = lambda r: 0.5 * np.abs(np.linalg.eigvalsh(extend(None, probe(r), Sd))).sum()
    k = np.arange(60) + 0.5
    dirs = np.stack([np.sqrt(1 - (1 - 2 * k / 60) ** 2) * np.cos(np.pi * (1 + 5 ** 0.5) * k),
                     np.sqrt(1 - (1 - 2 * k / 60) ** 2) * np.sin(np.pi * (1 + 5 ** 0.5) * k),
                     1 - 2 * k / 60], 1)
    cands = [np.zeros(3)] + [rad * d for rad in (0.5, 0.9, 0.999) for d in dirs]
    vals = [val(c) for c in cands]
    best = max(vals)
    to_r = lambda u: u / np.sqrt(1 + u @ u)
    for i in np.argsort(vals)[-starts:]:
        c = cands[i]
        u0 = c / np.sqrt(max(1e-12, 1 - c @ c))
        res = minimize(lambda u: -val(to_r(u)), u0, method="Nelder-Mead",
                       options=dict(xatol=1e-9, fatol=1e-12, maxiter=maxiter))
        best = max(best, -res.fun)
    return best


def f_pro(f):
    """Process fidelity with the identity: <Omega|(f (x) id)(Omega Omega^dagger)|Omega>."""
    return np.real(OMEGA.conj() @ extend(f, np.outer(OMEGA, OMEGA.conj())) @ OMEGA)


def r_avg(f, n=2):
    return 1 - (n * f_pro(f) + 1) / (n + 1)


def random_channel(rng):
    """A random qubit channel, sometimes close to the identity."""
    k = rng.integers(1, 5)
    G = rng.normal(size=(2 * k, 2)) + 1j * rng.normal(size=(2 * k, 2))
    V, _ = np.linalg.qr(G)                                          # isometry C^2 -> C^(2k)
    Ks = [V[2 * j:2 * j + 2, :] for j in range(k)]
    eps = 10 ** rng.uniform(-3, 0)
    H = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    U, _ = np.linalg.qr(np.eye(2) + 1j * eps * (H + H.conj().T) / 8)     # a unitary near the identity
    return lambda A: (1 - eps) * U @ A @ U.conj().T + eps * sum(K @ A @ K.conj().T for K in Ks)


def checks(rng):
    # the Werner--Holevo pair
    for _ in range(20):
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        assert np.allclose(wh_minus(A), Y @ A @ Y)                  # Phi_- is unitary
    J = extend(wh_plus, 2 * np.outer(OMEGA, OMEGA.conj()))
    assert np.linalg.eigvalsh(J).min() > -1e-12                     # Phi_+ is completely positive
    dirs = rng.normal(size=(400, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    no_anc = max(tdist(wh_plus(bloch_state(r)), wh_minus(bloch_state(r))) for r in dirs)
    assert abs(no_anc - 2 / 3) < 1e-9                               # every pure input gives 2/3
    for rad in (0.3, 0.7):
        r = rad * np.array([0.6, 0.0, 0.8])
        assert abs(tdist(wh_plus(bloch_state(r)), wh_minus(bloch_state(r))) - 2 * rad / 3) < 1e-12
    P = np.outer(OMEGA, OMEGA.conj())
    out_p, out_m = extend(wh_plus, P), extend(wh_minus, P)
    SW = np.eye(4)[[0, 2, 1, 3]]
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    assert np.allclose(out_p, (np.eye(4) + SW) / 6)                 # Pi_sym / 3
    assert np.allclose(out_m, np.outer(singlet, singlet))           # the singlet
    assert abs(np.trace(out_p @ out_m)) < 1e-12 and abs(tdist(out_p, out_m) - 1) < 1e-12
    assert abs(diamond_brute(wh_plus, wh_minus) - 1) < 1e-9
    # closed forms of the two errors
    for th in (0.05, 0.4, 1.3):
        U = rotation(th)
        assert abs(f_pro(U) - np.cos(th / 2) ** 2) < 1e-12
        assert abs(diamond_brute(U, identity) - np.sin(th / 2)) < 1e-6
        assert abs(diamond_brute(U, identity) - np.sqrt(1.5 * r_avg(U))) < 1e-6
    for p in (0.001, 0.05, 0.3):
        F = zflip(p)
        assert abs(f_pro(F) - (1 - p)) < 1e-12
        assert abs(diamond_brute(F, identity) - p) < 1e-6
        assert abs(diamond_brute(F, identity) - 1.5 * r_avg(F)) < 1e-6
    # the average gate fidelity, (N F_pro + 1)/(N + 1), against a Haar average over pure inputs
    v = rng.normal(size=(20000, 2)) + 1j * rng.normal(size=(20000, 2))
    v /= np.linalg.norm(v, axis=1)[:, None]
    for f in (rotation(0.9), zflip(0.2)):
        mc = np.mean([np.real(s.conj() @ f(np.outer(s, s.conj())) @ s) for s in v])
        assert abs(mc - (1 - r_avg(f))) < 3e-3
    # the general bounds, for random channels against the identity
    worst = 1.0
    for _ in range(200):
        f = random_channel(rng)
        r = r_avg(f)
        d = diamond_brute(f, identity, starts=1, maxiter=150)
        lo, hi = 1.5 * r, 2 * np.sqrt(1.5 * r)
        assert lo - 1e-9 <= d <= hi + 1e-9
        worst = min(worst, d / hi)
    print("Werner-Holevo: no ancilla %.4f, with ancilla %.4f; bounds hold for 200 random channels" % (no_anc, 1.0))


def main():
    rng = np.random.default_rng(8)
    checks(rng)
    r0 = 1e-4
    d_coh, d_sto = np.sqrt(1.5 * r0), 1.5 * r0
    th0 = 2 * np.arcsin(d_coh)
    assert abs(r_avg(rotation(th0)) - r0) < 1e-12 and abs(r_avg(zflip(d_sto)) - r0) < 1e-12
    print("at r = 1e-4: stochastic %.2e, coherent %.2e (theta = %.4f rad)" % (d_sto, d_coh, th0))

    b = []
    # ---- (a) left: the two images of the (x, z) disc
    R = 1.55
    C = np.array([0.0, 0.0])
    P = lambda v: C + R * np.asarray(v)
    b.append("\\fill[ndSky, fill opacity=0.55] %s circle (%.3f);" % (N.pt(C), R))
    b.append("\\draw[ink] %s circle (%.3f);" % (N.pt(C), R))
    b.append("\\fill[ndRose, fill opacity=0.9] %s circle (%.3f);" % (N.pt(C), R / 3))
    b.append("\\draw[ndRed, line width=0.8pt] %s circle (%.3f);" % (N.pt(C), R / 3))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$x$};"
             % (N.pt(P((-1.18, 0))), N.pt(P((1.2, 0)))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$z$};"
             % (N.pt(P((0, -1.18))), N.pt(P((0, 1.2)))))
    a = np.radians(122.0)
    rin = np.array([np.cos(a), np.sin(a)])                          # input, on the sphere
    ip, im = rin / 3, -rin                                          # its images under Phi_+ and Phi_-
    assert abs(0.5 * np.linalg.norm(ip - im) - 2 / 3) < 1e-12
    b.append("\\draw[dashedink, line width=0.7pt] %s -- %s;" % (N.pt(P(ip)), N.pt(P(im))))
    b.append("\\draw[faint, densely dotted] %s -- %s;" % (N.pt(P(rin)), N.pt(P(ip))))
    b.append("\\node[dot] at %s {};" % N.pt(P(rin)))
    b.append("\\node[dot, fill=ndRed] at %s {};" % N.pt(P(ip)))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % N.pt(P(im)))
    b.append("\\node[font=\\small, anchor=south east] at %s {$\\Bloch$};" % N.pt(P(rin) + np.array([-0.03, 0.02])))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=south west] at %s {$\\Phi_+(\\Bloch)$};"
             % N.pt(P(ip) + np.array([0.06, 0.04])))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=north west] at %s {$\\Phi_-(\\Bloch)=-\\Bloch$};"
             % N.pt(P(im) + np.array([0.06, -0.03])))
    mid = 0.5 * (P(ip) + P(im))
    b.append("\\node[note, anchor=north east] (dd) at %s {two images of one\\\\ input, at trace\\\\ distance $\\tfrac23$};"
             % N.pt(np.array([-1.2 * R, -0.45 * R])))
    b.append("\\draw[pointer] (dd.east) to[bend right=20] %s;" % N.pt(mid + np.array([-0.04, -0.06])))
    b.append("\\node[note, anchor=south, text=ndRed!80!ndInk] (np) at %s {$\\Phi_+$: the ball\\\\ shrunk to radius $\\tfrac13$};"
             % N.pt(np.array([1.75 * R, 0.55 * R])))
    b.append("\\draw[pointer] (np.south) to[bend left=15] %s;" % N.pt(P((0.26, -0.16))))
    b.append("\\node[note, anchor=south, text=ndBlue!80!ndInk] (nm) at %s {$\\Phi_-$: the whole ball,\\\\ turned by $\\pi$ about $y$};"
             % N.pt(np.array([-1.65 * R, 0.55 * R])))
    b.append("\\draw[pointer] (nm.south) to[bend right=15] %s;" % N.pt(P((-0.78, 0.3))))
    b.append("\\node[font=\\small] at %s {one qubit: at most $\\tfrac23$};" % N.pt(np.array([0.0, -1.62 * R])))

    # ---- (a) right: the entangled probe
    ox = 4.9
    cy = 0.35
    b.append("\\node[circle, fill=ndSky, draw=ndBlue, line width=0.9pt, minimum size=0.55cm, font=\\scriptsize] (st) at (%.2f,%.2f) {$\\Phi^+$};"
             % (ox + 0.55, cy - 1.15))
    b.append("\\draw[ink] (st.north west) -- (%.2f,%.2f);" % (ox, cy - 0.35))
    b.append("\\draw[ink] (st.north east) -- (%.2f,%.2f);" % (ox + 1.1, cy + 0.9))
    b.append("\\node[rectangle, rounded corners=3pt, fill=ndRose, draw=ndRed, line width=0.9pt, minimum size=0.6cm, font=\\scriptsize] (ch) at (%.2f,%.2f) {$\\Phi_\\pm$};"
             % (ox, cy + 0.0))
    b.append("\\draw[ink] (ch.north) -- (%.2f,%.2f);" % (ox, cy + 0.9))
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.2f,%.2f) {system};" % (ox - 0.42, cy - 0.42))
    b.append("\\node[font=\\scriptsize, anchor=west] at (%.2f,%.2f) {ancilla};" % (ox + 1.12, cy - 0.15))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=south] at (%.2f,%.2f) {$\\Phi_+$: $\\tfrac13\\Pi_{\\rm sym}$};"
             % (ox + 0.55, cy + 1.48))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=south] at (%.2f,%.2f) {$\\Phi_-$: the singlet};"
             % (ox + 0.55, cy + 1.1))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {orthogonal outputs:\\\\ perfectly\\\\ distinguishable};"
             % (ox + 0.95, cy - 1.15))
    b.append("\\node[font=\\small] at (%.2f,%.2f) {with an ancilla: $1$};" % (ox + 0.85, -1.62 * R))
    b.append("\\node[font=\\small] at (%.2f,%.2f) {(a) why the ancilla matters};" % (2.0, -1.62 * R - 0.6))

    # ---- (b) average case against worst case, log-log
    Pl = N.Plot((-6.0, -1.0), (-6.0, 0.0), 5.4, 4.4, origin=(9.7, -2.45))
    b += Pl.axes("$r=1-\\bar F$", "$\\tfrac12\\|\\Phi-\\mathrm{id}\\|_\\diamond$",
                 xticks=[(k, "$10^{%d}$" % k) for k in (-6, -4, -2)],
                 yticks=[(k, "$10^{%d}$" % k) for k in (-6, -4, -2)] + [(0, "$1$")])
    lr = np.linspace(-6, -1, 200)
    r = 10 ** lr
    lo = np.log10(1.5 * r)
    hi = np.log10(np.minimum(2 * np.sqrt(1.5 * r), 1.0))
    band = [Pl.P(x, y) for x, y in zip(lr, lo)] + [Pl.P(x, y) for x, y in zip(lr[::-1], hi[::-1])]
    b.append("\\fill[ndSky, fill opacity=0.6] %s -- cycle;" % " -- ".join(N.pt(p) for p in band))
    b.append(Pl.curve(lr, hi, "faint"))
    b.append(Pl.curve(lr, np.log10(np.sqrt(1.5 * r)), "line width=1.4pt, ndRed"))
    b.append(Pl.curve(lr, lo, "line width=1.4pt, ndBlue"))
    b.append("\\draw[dashedink] %s -- %s;" % (Pl.pt(-4, -6), Pl.pt(-4, 0)))
    b.append("\\node[dot, fill=ndRed] at %s {};" % Pl.pt(-4, np.log10(d_coh)))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % Pl.pt(-4, np.log10(d_sto)))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=north west] at %s {$1.2\\cdot10^{-2}$};"
             % Pl.pt(-3.92, np.log10(d_coh) - 0.12))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=north west] at %s {$1.5\\cdot10^{-4}$};"
             % Pl.pt(-3.92, np.log10(d_sto) - 0.05))
    b.append("\\node[note, anchor=south west, text=ndRed!80!ndInk] at %s {coherent: $\\sqrt{3r/2}$};"
             % Pl.pt(-3.7, -0.55))
    b.append("\\node[note, anchor=north west, text=ndBlue!80!ndInk] at %s {stochastic: $3r/2$};"
             % Pl.pt(-2.6, -3.2))
    b.append("\\node[note, anchor=west] (bd) at %s {every channel\\\\ lies in the band};" % Pl.pt(-5.9, -0.75))
    b.append("\\draw[pointer] (bd.south) to[bend right=15] %s;" % Pl.pt(-5.2, -3.0))
    b.append("\\node[note, anchor=west] at %s {a good gate today,\\\\ $r=10^{-4}$};" % Pl.pt(-3.9, -5.15))
    b.append("\\node[font=\\small] at %s {(b) average case against worst case};" % N.pt(Pl.P(-3.5, -6) + np.array([0, -1.05])))
    N.write("ch08_diamond", b)
    print("written figures/ch08_diamond.tex")


if __name__ == "__main__":
    main()
