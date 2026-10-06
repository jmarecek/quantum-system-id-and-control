"""ch03_decoherence.py -- a qubit under amplitude damping and dephasing.

Integrates the GKSL master equation in vectorised form,
    d/dt vec(rho) = L vec(rho),
using column-stacking vectorisation, vec(A X B) = (B^T kron A) vec(X).
Writes figures/ch03_decoherence.tex (Bloch components) and
figures/ch03_bloch_shrink.tex (trajectories on the Bloch sphere).
"""
import numpy as np
from scipy.linalg import expm
import tikzexport as T
import needham as N

I, X, Y, Z = T.pauli()


def vec(A):
    """Column-stacking vectorisation."""
    return A.reshape(-1, order="F")


def unvec(v, n=2):
    return v.reshape((n, n), order="F")


def lindbladian(H, jumps):
    """Matrix L with  vec(-i[H,rho] + sum_k D[L_k] rho) = L vec(rho)."""
    n = H.shape[0]
    Id = np.eye(n)
    L = -1j * (np.kron(Id, H) - np.kron(H.T, Id))          # -i[H, .]
    for Lk in jumps:
        LdL = Lk.conj().T @ Lk
        L += np.kron(Lk.conj(), Lk)                          # Lk rho Lk^dag
        L -= 0.5 * (np.kron(Id, LdL) + np.kron(LdL.T, Id))  # -1/2 {Lk^dag Lk, rho}
    return L


def evolve(rho0, L, times):
    return [unvec(expm(L * t) @ vec(rho0)) for t in times]


def components_figure(times, r, omega, T1, T2):
    """Bloch components with the envelopes e^{-t/T2} and the asymptote z -> 1."""
    Pl = N.Plot((0, times[-1]), (-1.05, 1.1), 10.0, 4.6)
    b = Pl.axes("$t$", "", xticks=[(T2, "$T_2$"), (T1, "$T_1$")],
                yticks=[(-1, "$-1$"), (0, "$0$"), (1, "$1$")], yzero=0.0)
    env = np.exp(-times / T2)
    b.append(Pl.curve(times, env, "dashedink"))
    b.append(Pl.curve(times, -env, "dashedink"))
    b.append("\\draw[dashedink] %s -- %s;" % (Pl.pt(0, 1), Pl.pt(times[-1], 1)))
    first = lambda arr, sign: int(np.argmax(sign * arr[: len(arr) // 3]))      # first trough / peak
    spots = {"x": (first(r[:, 0], -1), "below"), "y": (first(r[:, 1], +1), "above right"), "z": (int(0.92 * len(times)), "above")}
    for k, (col, name) in enumerate([("ndBlue", "x"), ("ndGold", "y"), ("ndRed", "z")]):
        b.append(Pl.curve(times, r[:, k], "line width=1.3pt, %s" % col))
        i, pos = spots[name]
        b.append("\\node[font=\\small, text=%s, %s] at %s {$%s=\\langle\\sigma_%s\\rangle$};"
                 % (col, pos, Pl.pt(times[i], r[i, k]), name, name))
    assert np.allclose(np.hypot(r[:, 0], r[:, 1]), env, atol=1e-6)
    assert np.allclose(r[:, 2], 1 - np.exp(-times / T1), atol=1e-6)
    tn = 4.6
    b.append("\\node[note, anchor=south west] (e) at %s {precession at $\\omega$, damped:\\\\ envelope $\\pm\\ee^{-t/T_2}$};" % Pl.pt(tn, -1.0))
    b.append("\\draw[pointer] (e.north west) to[bend left=20] %s;" % Pl.pt(3.4, -np.exp(-3.4 / T2) - 0.02))
    b.append("\\node[note, anchor=south] (z) at %s {$z\\to1$: relaxation to the\\\\ ground state on the scale $T_1$};" % Pl.pt(6.0, 1.15))
    b.append("\\draw[pointer] (z.south) to[bend left=10] %s;" % Pl.pt(6.0, r[np.searchsorted(times, 6.0), 2] + 0.03))
    return b


def shrink_figure(L, T1, T2):
    """Trajectories on the Bloch sphere, and the image of the sphere at time t1."""
    V = N.View(az=25.0, el=18.0, R=2.5)
    b = V.sphere()
    b += V.great_circle([0, 0, 1])
    for v, lab, pos in [((0, 0, 1.25), "\\ket0", "above"), ((1.35, 0, 0), "x", "below left"), ((0, 1.3, 0), "y", "right")]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s, text=ndInk] {$%s$};" % (V.pt((0, 0, 0)), V.pt(v), pos, lab))
    # image of the sphere at time t1: an ellipsoid, centre (0,0,1-e^{-t1/T1}),
    # semi-axes e^{-t1/T2} (x, y) and e^{-t1/T1} (z); drawn by two of its rings
    t1 = 1.2
    a_xy, a_z, c = np.exp(-t1 / T2), np.exp(-t1 / T1), 1 - np.exp(-t1 / T1)
    s = np.linspace(0, 2 * np.pi, 200)
    eq = np.stack([a_xy * np.cos(s), a_xy * np.sin(s), c + 0 * s], 1)
    mer = np.stack([a_xy * np.cos(s), 0 * s, c + a_z * np.sin(s)], 1)
    for ring in (eq, mer):
        b.append(N.polyline([V.P(p) for p in ring], "line width=0.9pt, ndRed, dash pattern=on 3pt off 1.5pt"))
    # check the ellipsoid against the exact map on a few points of the sphere
    for th, ph in [(0.4, 0.3), (2.0, 1.7), (2.9, -1.0)]:
        psi = np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])
        rt = T.bloch_vector(evolve(np.outer(psi, psi.conj()), L, [t1])[0])
        assert np.isclose((rt[0] ** 2 + rt[1] ** 2) / a_xy ** 2 + (rt[2] - c) ** 2 / a_z ** 2, 1.0)
    cols = ["ndBlue", "ndGreen", "ndGold"]
    for theta, col in zip([np.pi / 2, 2.2, 2.8], cols):
        psi = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
        rhos = evolve(np.outer(psi, psi.conj()), L, np.linspace(0, 6, 400))
        tr = np.array([T.bloch_vector(rho) for rho in rhos])
        b += V.curve(tr, "line width=1.3pt, %s" % col, "line width=0.6pt, %s, densely dotted" % col)
        b.append("\\node[dot, fill=%s] at %s {};" % (col, V.pt(tr[0])))
    b.append("\\node[dot, fill=ndInk, minimum size=4.4pt] at %s {};" % V.pt((0, 0, 1)))
    x0 = V.R + 0.7
    b.append("\\node[note, anchor=west] (a) at (%.2f,2.0) {every state spirals\\\\ to the ground state $\\ket0$};" % x0)
    b.append("\\draw[pointer] (a.west) to[bend right=15] %s;" % V.pt((0.05, 0, 1.0)))
    b.append("\\node[note, anchor=west, text=ndRed] (e) at (%.2f,0.0) {the whole Bloch ball at\\\\ $t=%.1f$: an ellipsoid,\\\\ shrunk and pushed up};" % (x0, t1))
    b.append("\\draw[pointer] (e.west) to[bend left=10] %s;" % V.pt(eq[20]))
    b.append("\\node[note, anchor=west] at (%.2f,-1.9) {dots: three pure\\\\ initial states};" % x0)
    return b


if __name__ == "__main__":
    omega, gamma1, gammaphi = 2.0, 0.4, 0.3
    H = -0.5 * omega * Z                                  # |0> is the ground state
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)           # |0><1|
    jumps = [np.sqrt(gamma1) * sigma_minus, np.sqrt(gammaphi / 2) * Z]
    L = lindbladian(H, jumps)
    T1, T2 = 1 / gamma1, 1 / (gamma1 / 2 + gammaphi)
    times = np.linspace(0, 8, 300)
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)                 # |+>
    rhos = evolve(np.outer(psi, psi.conj()), L, times)
    r = np.array([T.bloch_vector(rho) for rho in rhos])
    N.write("ch03_decoherence", components_figure(times, r, omega, T1, T2))
    N.write("ch03_bloch_shrink", shrink_figure(L, T1, T2))
    print("written figures/ch03_decoherence.tex, figures/ch03_bloch_shrink.tex")
