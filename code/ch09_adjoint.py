"""ch09_adjoint.py -- the state forwards, the adjoint backwards.

The driven qubit of Chapter 4, H = (1/2)(Delta sigma_z + u sigma_x) with
Delta = -omega = -1, a fixed final time T = 3, and the terminal cost
G = 1 - F with F = |<1|psi(T)>|^2.  The control u(t) = sin(pi t/T) is a guess,
not optimal: it brings F = 0.419.  The state is integrated forwards from |0>;
the adjoint, with the terminal value |chi(T)> = 2 |1><1|psi(T)> given by
eq. (ch09:qadj) with chi_0 = -1, is integrated backwards under the same
propagators; and
    dH_P/du(t) = Im <chi(t)| sigma_x/2 |psi(t)>
is the derivative of F with respect to the control at the time t (eq.
ch09:qmax): for a piecewise-constant control with steps of length dt,
dF/du_k = dt * dH_P/du at the middle of the step, which is checked against
finite differences to a relative error below 1e-4.

In the Bloch picture F = (1 - z)/2, the costate is p(T) = grad F = (0, 0, -1/2),
carried backwards by the same rotations as the state, and
dH_P/du = p . (x x r), the component along p of the velocity x x r that a unit
of control would add; the script checks that this equals the complex formula.
(a) The Bloch trajectory (blue) from |0>, ending at the height z(T) = 0.163,
    and the tangent part of the costate (gold) at t = 0, 0.75, 1.5, 2.25, 3:
    at T it points down the meridian towards |1>, and at earlier times it is
    that direction carried back along the trajectory.
(b) The control (blue) and dH_P/du (red) against t; the state is computed
    forwards from t = 0 and the adjoint backwards from t = T.
(c) F along the direction of the gradient, u + eps dH_P/du: its slope at
    eps = 0 is int (dH_P/du)^2 dt = 0.207, the first variation of Section 9.1
    (dashed), and it reaches 0.657 at eps = 2.0 before falling again.
Writes figures/ch09_adjoint.tex.
"""
import numpy as np
import tikzexport as T
import needham as N

I2, SX, SY, SZ = T.pauli()
DELTA = -1.0
TT = 3.0
NSTEP = 600


def step(u, dt):
    """exp(-i dt (Delta sigma_z + u sigma_x)/2), in closed form."""
    w = np.hypot(DELTA, u)
    return np.cos(w * dt / 2) * I2 - 1j * np.sin(w * dt / 2) * (DELTA * SZ + u * SX) / w


def guess(n=NSTEP):
    ts = (np.arange(n) + 0.5) * TT / n
    return ts, np.sin(np.pi * ts / TT)


def propagate(us):
    """psi at the step boundaries (forwards), chi at the step boundaries (backwards), F, and dH_P/du at the midpoints."""
    n = len(us)
    dt = TT / n
    Us = [step(u, dt) for u in us]
    halves = [step(u, dt / 2) for u in us]
    psi = [np.array([1, 0], dtype=complex)]
    for U in Us:
        psi.append(U @ psi[-1])
    a = psi[-1][1]
    chi = [None] * (n + 1)
    chi[n] = 2 * np.array([0, a])                               # |chi(T)> = 2 |1><1|psi(T)>
    for k in range(n - 1, -1, -1):
        chi[k] = Us[k].conj().T @ chi[k + 1]
    g = np.array([np.imag((h.conj().T @ chi[k + 1]).conj() @ (SX / 2) @ (h @ psi[k])) for k, h in enumerate(halves)])
    return np.array(psi), np.array(chi), abs(a) ** 2, g


def bloch(psi):
    return np.real(np.einsum('...i,kij,...j->...k', psi.conj(), np.array([SX, SY, SZ]), psi))


def rot(n, th):
    n = np.asarray(n, float) / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def costate(us):
    """The Bloch costate p at the step boundaries, from p(T) = (0, 0, -1/2) carried backwards."""
    n = len(us)
    dt = TT / n
    Rs = [rot([u, 0, DELTA], np.hypot(u, DELTA) * dt) for u in us]
    p = [None] * (n + 1)
    p[n] = np.array([0, 0, -0.5])
    for k in range(n - 1, -1, -1):
        p[k] = Rs[k].T @ p[k + 1]
    return np.array(p)


def checks():
    ts, us = guess()
    psi, chi, F, g = propagate(us)
    dt = TT / len(us)
    assert abs(F - 0.4185) < 1e-3
    # the gradient against finite differences of F
    for k in (50, 200, 350, 500):
        e = 1e-5
        up, dn = us.copy(), us.copy()
        up[k] += e
        dn[k] -= e
        fd = (propagate(up)[2] - propagate(dn)[2]) / (2 * e)
        assert abs(fd / (g[k] * dt) - 1) < 1e-4
    # the Bloch picture gives the same gradient: p . (x x r) at the midpoints
    r = bloch(psi)
    p = costate(us)
    for k in (50, 200, 350, 500):
        h = np.array([us[k], 0, DELTA])
        Rh = rot(h, np.linalg.norm(h) * dt / 2)
        rm, pm = Rh @ r[k], Rh.T @ p[k + 1]
        assert abs(pm @ np.cross([1, 0, 0], rm) - g[k]) < 1e-9
    # the conserved pairing: p . r is constant, and p(T) is the gradient of F = (1 - z)/2
    assert np.ptp(np.sum(p * r, 1)) < 1e-9
    assert abs((1 - r[-1, 2]) / 2 - F) < 1e-12
    # the first variation and the line search
    slope = np.sum(g ** 2) * dt
    assert abs(slope - 0.2067) < 1e-3
    eps = 1e-4
    assert abs((propagate(us + eps * g)[2] - F) / eps - slope) < 1e-3
    es = np.linspace(0, 5, 101)
    Fs = np.array([propagate(us + e * g)[2] for e in es])
    assert abs(es[np.argmax(Fs)] - 2.0) < 0.1 and abs(Fs.max() - 0.657) < 2e-3
    return es, Fs, slope


def main():
    es, Fs, slope = checks()
    ts, us = guess()
    psi, chi, F, g = propagate(us)
    r = bloch(psi)
    p = costate(us)
    tb = np.linspace(0, TT, NSTEP + 1)
    b = []
    # ---- (a) the Bloch sphere
    V = N.View(az=-160.0, el=22.0, R=1.75, shift=(2.1, 2.45))
    b += V.sphere()
    b += V.great_circle([0, 0, 1])
    b += V.curve(r, "line width=1.4pt, ndBlue", "line width=0.6pt, ndBlue!55, densely dotted")
    for t in (0.0, 0.75, 1.5, 2.25, 3.0):
        k = int(round(t / TT * NSTEP))
        pt_ = p[k] - (p[k] @ r[k]) * r[k]
        vis = r[k] @ V.d >= 0
        b.append("\\draw[vec, %s] %s -- %s;" % ("ndGold!85!ndInk" if vis else "ndGold!50", V.pt(r[k]), V.pt(r[k] + 1.0 * pt_)))
        b.append("\\node[dot, minimum size=2.8pt, fill=%s] at %s {};" % ("ndBlue" if vis else "ndBlue!50", V.pt(r[k])))
    b.append("\\node[dot] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[dot, fill=ndRed] at %s {};" % V.pt((0, 0, -1)))
    b.append("\\node[font=\\scriptsize, anchor=south] at %s {$\\ket0$};" % N.pt(V.P((0, 0, 1)) + [0, 0.05]))
    b.append("\\node[font=\\scriptsize, anchor=north, text=ndRed] at %s {$\\ket1$};" % N.pt(V.P((0, 0, -1)) + [0, -0.05]))
    c = V.P((0, 0, 0))
    b.append("\\node[note, anchor=west, text=ndBlue] (st) at %s {the state, forwards\\\\ from $\\ket0$};" % N.pt(c + [1.95, 1.2]))
    k1 = int(0.35 * NSTEP)
    b.append("\\draw[pointer] (st.west) to[bend right=15] %s;" % N.pt(V.P(r[k1]) + [0.05, 0.03]))
    b.append("\\node[note, anchor=west, text=ndGold!55!ndInk] (co) at %s {the costate: the gradient\\\\ of $F$ at $T$, carried back};" % N.pt(c + [1.95, -1.05]))
    kT = NSTEP
    b.append("\\draw[pointer] (co.west) to[bend left=15] %s;" % N.pt(V.P(r[kT] + 0.65 * (p[kT] - (p[kT] @ r[kT]) * r[kT])) + [0.05, 0.0]))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) state and costate on the sphere};" % N.pt(c + [0.3, -2.05]))

    # ---- (b) the control and the gradient against time
    pl = N.Plot((0, TT), (-0.55, 1.1), 4.6, 3.1, origin=(7.6, 0.85))
    b += pl.axes(xlabel="{\\scriptsize $t$}", xticks=[(1, "$1$"), (2, "$2$"), (3, "$T$")], yticks=[(1, "$1$")], yzero=0)
    b.append(pl.curve(ts, us, "line width=1.3pt, ndBlue"))
    b.append(pl.curve(ts, g, "line width=1.3pt, ndRed"))
    b.append("\\node[font=\\scriptsize, text=ndBlue, anchor=east] at %s {$u(t)$};" % pl.pt(0.48, 0.62))
    km = int(np.argmax(g))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=north west] at %s {$\\partial_u\\HP(t)$};" % pl.pt(0.12, -0.2))
    ya, yb = 1.42, 1.25
    b.append("\\draw[->, >=stealth, line width=0.8pt, ndBlue] %s -- %s node[midway, above, font=\\scriptsize\\itshape, text=ndBlue] {$\\ket\\psi$ forwards from $t=0$};" % (pl.pt(0, ya + 0.12), pl.pt(TT, ya + 0.12)))
    b.append("\\draw[->, >=stealth, line width=0.8pt, ndGold!85!ndInk] %s -- %s node[midway, below, font=\\scriptsize\\itshape, text=ndGold!55!ndInk] {$\\ket\\chi$ backwards from $t=T$};" % (pl.pt(TT, yb + 0.1), pl.pt(0, yb + 0.1)))
    b.append("\\node[note, anchor=north] at %s {raise $u$ where $\\partial_u\\HP>0$,\\\\ lower it where $\\partial_u\\HP<0$};" % pl.pt(1.5, -0.62))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) the control and the gradient};" % pl.pt(1.5, -1.25))

    # ---- (c) the line search along the gradient
    pc = N.Plot((0, 5), (0, 1.0), 3.6, 3.1, origin=(13.75, 0.85))
    b += pc.axes(xlabel="{\\scriptsize $\\epsilon$}", ylabel="{\\scriptsize $F$}", xticks=[(2, "$2$"), (4, "$4$")],
                 yticks=[(F, "$%.2f$" % F), (Fs.max(), "$%.2f$" % Fs.max())])
    b.append("\\draw[faint, dash pattern=on 1pt off 1.5pt] %s -- %s;" % (pc.pt(0, Fs.max()), pc.pt(es[np.argmax(Fs)], Fs.max())))
    b.append(pc.curve(es, Fs, "line width=1.3pt, ndBlue"))
    b.append("\\draw[line width=0.9pt, ndRed, dash pattern=on 3pt off 2pt] %s -- %s;" % (pc.pt(0, F), pc.pt(2.6, F + 2.6 * slope)))
    b.append("\\node[dot, fill=ndBlue] at %s {};" % pc.pt(es[np.argmax(Fs)], Fs.max()))
    b.append("\\node[note, anchor=north west, text=ndRed] (sl) at %s {slope $\\int(\\partial_u\\HP)^2\\mathrm dt$};" % pc.pt(2.05, 0.5))
    b.append("\\draw[pointer] (sl.north west) to[bend left=20] %s;" % pc.pt(1.9, F + 1.9 * slope - 0.015))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) $F$ along $u+\\epsilon\\,\\partial_u\\HP$};" % pc.pt(2.5, -0.4))
    N.write("ch09_adjoint", b)
    return F, slope, es, Fs


if __name__ == "__main__":
    F, slope, es, Fs = main()
    print("F = %.4f, slope = %.4f, best eps = %.2f with F = %.4f" % (F, slope, es[np.argmax(Fs)], Fs.max()))
    print("written figures/ch09_adjoint.tex")
