"""ch09_switchsphere.py -- the adjoint of the time-optimal transfer, on the sphere.

The time-optimal two-bang transfer of the driven qubit (ch09_timeoptimal.py:
Delta = -1, u = -u0 = -1.5 for t1 = 1.127, then u = +u0 for t2 = 2.359,
t* = 3.485, Omega = 1.803).  Along it, the vector H = (H_x, H_y, H_z),
H_k = (1/2) Im<chi|sigma_k|psi>, computed from the state and the adjoint.
  * H is orthogonal to the Bloch vector r, since H.r is conserved and vanishes
    for a target defined up to a phase: H is a tangent vector of the sphere at
    r(t), and p = H x r is the adjoint of the Bloch-vector formulation, with
    p . r' = h . H = 1 > 0 (H_P = 0 with chi_0 = -1).
  * |H| = 1/|Delta| = 1 is conserved, and h . H = 1 on both bangs, h = (u, 0, Delta).
  * The control is u = u0 sgn H_x: H_x < 0 on the first bang, > 0 on the second.
    At the switch H = (0, 0, 1/Delta) = (0, 0, -1), perpendicular to the x-axis.
(a) The optimal path on the Bloch sphere with H drawn as tangent arrows at
    nine instants (gold), the axes h/Omega of the two bangs, and the x-axis.
(b) H itself, a unit vector, on a second sphere: on each bang it turns about
    the axis of that bang (Proposition 9.x), and the control switches where H
    meets the plane H_x = 0 (red), at (0, 0, -1), touching it with velocity
    (0, -u0, 0) in the plane: the zero of H_x is double, so that H_x changes
    sign with zero slope.
Writes figures/ch09_switchsphere.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N
import ch09_timeoptimal as TO

SX, SY, SZ = TO.SX, TO.SY, TO.SZ
DELTA, U0, W, T1, TSTAR = TO.DELTA, TO.U0, TO.W, TO.T1, TO.TSTAR
S0 = -1                                                  # the solution drawn in Figure 9.2(a)


def Hm(u):
    return 0.5 * (DELTA * SZ + u * SX)


def extremal(ts):
    """Bloch vector r(t), the vector H(t) and the control u(t) along the optimal path."""
    psi1 = expm(-1j * Hm(S0 * U0) * T1) @ np.array([1, 0], dtype=complex)
    chi1 = -2j * (1 / DELTA) * SZ @ psi1                  # H(t1) = (0, 0, 1/Delta)
    rs, Hs, us = [], [], []
    for t in ts:
        u = S0 * U0 if t < T1 else -S0 * U0
        V = expm(-1j * Hm(u) * (t - T1))
        psi, chi = V @ psi1, V @ chi1
        rs.append(np.real([psi.conj() @ P @ psi for P in (SX, SY, SZ)]))
        Hs.append([0.5 * np.imag(chi.conj() @ P @ psi) for P in (SX, SY, SZ)])
        us.append(u)
    return np.array(rs), np.array(Hs), np.array(us)


def checks():
    ts = np.linspace(0, TSTAR, 1201)
    r, H, u = extremal(ts)
    assert np.allclose(r[0], [0, 0, 1]) and np.allclose(r[-1], [0, 0, -1], atol=1e-9)
    assert np.allclose(np.sum(H * r, 1), 0, atol=1e-12)                       # tangent to the sphere
    assert np.allclose(np.linalg.norm(H, axis=1), 1 / abs(DELTA))              # conserved length
    h = np.stack([u, 0 * u, DELTA + 0 * u], 1)
    assert np.allclose(np.sum(h * H, 1), 1.0)                                  # H_P = h.H - 1 = 0
    off = np.abs(ts - T1) > 1e-3
    assert np.all(np.sign(H[off, 0]) == np.sign(u[off]))                       # u = u0 sgn H_x
    k = np.searchsorted(ts, T1)
    _, H1, _ = extremal(np.array([T1]))
    assert np.allclose(H1[0], [0, 0, 1 / DELTA])
    # the velocity of H at the switch lies in the plane H_x = 0: a double zero of H_x
    for uu in (S0 * U0, -S0 * U0):
        v = np.cross([uu, 0, DELTA], H1[0])
        assert abs(v[0]) < 1e-12 and abs(abs(v[1]) - U0) < 1e-12
    # p = H x r is the Bloch adjoint: it turns with the state, and p . r' = h . H = 1
    p = np.cross(H, r)
    rdot = np.cross(h, r)
    assert np.allclose(np.sum(p * rdot, 1), 1.0)
    # on each bang H moves on a circle about its axis: h.H constant and |H| constant
    return ts, r, H, u


def arrow(V, base, vec, scale, style):
    q0, q1 = V.P(base), V.P(base + scale * vec)
    return "\\draw[%s] %s -- %s;" % (style, N.pt(q0), N.pt(q1))


def main():
    ts, r, H, u = checks()
    b = []
    nm = np.array([S0 * U0, 0, DELTA]) / W                # axis of the first bang
    npl = np.array([-S0 * U0, 0, DELTA]) / W              # axis of the second
    # ---- (a) the Bloch sphere with H along the path
    V = N.View(az=12.0, el=10.0, R=2.0, shift=(2.4, 2.5))
    b += V.sphere()
    b += V.great_circle([0, 0, 1], front="faint", back="faint, densely dotted")
    o = V.P((0, 0, 0))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (V.pt((0, 0, 0)), V.pt((0, 0, 1.25))))
    b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$z$};" % N.pt(V.P((0, 0, 1.25)) + np.array([0.0, 0.14])))
    b.append("\\draw[vec, ndRed!80, line width=0.9pt] %s -- %s;" % (V.pt((0, 0, 0)), V.pt((1.3, 0, 0))))
    b.append("\\node[font=\\scriptsize, text=ndRed] at %s {$x$};" % N.pt(V.P((1.3, 0, 0)) + np.array([-0.05, -0.17])))
    for ax, lab, off in [(nm, "\\vec h_1", (-0.25, -0.1)), (npl, "\\vec h_2", (0.05, -0.2))]:
        b.append("\\draw[vec, ndInk!55, line width=0.8pt] %s -- %s;" % (V.pt((0, 0, 0)), V.pt(0.95 * ax)))
        b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%s$};" % (N.pt(V.P(0.95 * ax) + np.array(off)), lab))
    b += V.curve(r, "line width=1.5pt, ndBlue", "line width=0.6pt, ndBlue!50, densely dotted")
    picks = [int(i) for i in np.linspace(60, len(ts) - 61, 9)]
    k1 = np.searchsorted(ts, T1)
    picks = sorted(set(picks + [k1]))
    for i in picks:
        if r[i] @ V.d < -0.05:
            continue
        sty = "vec, ndGold!85!ndInk, line width=1.0pt" if i != k1 else "vec, ndRed, line width=1.2pt"
        b.append(arrow(V, r[i], H[i], 0.42, sty))
        b.append("\\fill[ndBlue] %s circle (1.1pt);" % V.pt(r[i]))
    b.append("\\node[dot] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[font=\\scriptsize, anchor=south east] at %s {$\\ket0$};" % N.pt(V.P((0, 0, 1)) + np.array([-0.05, 0.03])))
    b.append("\\node[font=\\scriptsize, anchor=north east] at %s {$\\ket1$};" % N.pt(V.P((0, 0, -1)) + np.array([-0.05, -0.03])))
    b.append("\\node[note, anchor=west, text=ndGold!55!ndInk] (hn) at %s {$\\vec H$: tangent to\\\\ the sphere, of\\\\ length $1/\\abs\\Delta$};" % N.pt(o + np.array([2.25, 1.25])))
    i2 = picks[2]
    b.append("\\draw[pointer] (hn.west) to[bend right=15] %s;" % N.pt(V.P(r[i2] + 0.25 * H[i2]) + np.array([0.04, 0.03])))
    b.append("\\node[note, anchor=west, text=ndRed] (sw) at %s {at the switch\\\\ $\\vec H\\perp x$: $H_x=0$};" % N.pt(o + np.array([2.25, -0.55])))
    b.append("\\draw[pointer] (sw.west) to[bend left=10] %s;" % N.pt(V.P(r[k1] + 0.3 * H[k1]) + np.array([0.06, 0.0])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) $\\vec H$ along the optimal path};" % N.pt(o + np.array([0.0, -2.3])))

    # ---- (b) H on its own sphere
    V2 = N.View(az=40.0, el=-25.0, R=1.75, shift=(9.9, 2.5))
    o2 = V2.P((0, 0, 0))
    b += V2.sphere()
    b += V2.great_circle([1, 0, 0], front="line width=0.9pt, ndRed, dash pattern=on 3pt off 2pt", back="line width=0.5pt, ndRed!60, densely dotted")
    b += V2.great_circle([0, 0, 1], front="faint", back="faint, densely dotted")
    for e, lab, off in [((1.3, 0, 0), "H_x", (-0.1, -0.18)), ((0, 1.3, 0), "H_y", (0.2, 0.0)), ((0, 0, 1.3), "H_z", (0.0, 0.16))]:
        b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (V2.pt((0, 0, 0)), V2.pt(e)))
        b.append("\\node[font=\\scriptsize, text=ndInk!70] at %s {$%s$};" % (N.pt(V2.P(e) + np.array(off)), lab))
    for ax in (nm, npl):
        b.append("\\draw[vec, ndInk!55, line width=0.8pt] %s -- %s;" % (V2.pt((0, 0, 0)), V2.pt(0.95 * ax)))
    first, second = H[:k1 + 1], H[k1:]
    b += V2.curve(first, "line width=1.5pt, ndGold!85!ndInk", "line width=0.6pt, ndGold!70, densely dotted")
    b += V2.curve(second, "line width=1.5pt, ndGold", "line width=0.6pt, ndGold!70, densely dotted")
    for q, txt, d, anc in [(H[0], "$t=0$", (0.06, 0.03), "south west"), (H[-1], "$t^\\star$", (0.06, 0.0), "west")]:
        b.append("\\node[dot, fill=ndGold!85!ndInk, minimum size=3pt] at %s {};" % V2.pt(q))
        b.append("\\node[font=\\scriptsize, anchor=%s] at %s {%s};" % (anc, N.pt(V2.P(q) + np.array(d)), txt))
    b.append("\\node[dot, fill=ndRed] at %s {};" % V2.pt(H[k1]))
    b.append("\\node[note, anchor=west, text=ndRed] (pl) at %s {the plane $H_x=0$:\\\\ the control switches\\\\ where $\\vec H$ meets it};" % N.pt(o2 + np.array([2.15, -1.05])))
    b.append("\\draw[pointer] (pl.west) to[bend left=15] %s;" % N.pt(V2.P(H[k1]) + np.array([0.08, -0.02])))
    b.append("\\node[note, anchor=west] (cr) at %s {on each bang $\\vec H$\\\\ turns about $\\vec h$};" % N.pt(o2 + np.array([2.15, 1.3])))
    j = int(0.35 * k1)
    b.append("\\draw[pointer] (cr.west) to[bend right=10] %s;" % N.pt(V2.P(H[j]) + np.array([0.07, 0.03])))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) $\\vec H$ on its own sphere};" % N.pt(o2 + np.array([0.0, -2.05])))
    N.write("ch09_switchsphere", b)


if __name__ == "__main__":
    main()
    print("written figures/ch09_switchsphere.tex")
