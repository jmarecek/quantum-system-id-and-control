"""ch02_qutrit.py -- the qutrit state space is not a ball.

The diagonal slice of the qutrit state space in coherence-vector coordinates
x3 = Tr(F3 rho), x8 = Tr(F8 rho), with F3 = diag(1,-1,0)/sqrt2 and
F8 = diag(1,1,-2)/sqrt6.  The states rho = I/3 + x3 F3 + x8 F8 with rho >= 0
form a triangle (the simplex of diagonal probability vectors) whose corners are
the pure states |0>,|1>,|2>.  It lies between the circles of radius
1/sqrt(N(N-1)) = 1/sqrt6 and sqrt(1 - 1/N) = sqrt(2/3), touching the outer one at
the corners and the inner one at the midpoints of the edges.  The point
0.8 F8 lies inside the outer circle but outside the triangle.
Writes figures/ch02_qutrit.tex.
"""
import numpy as np
import needham as N

F3 = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2)
F8 = np.diag([1.0, 1.0, -2.0]) / np.sqrt(6)
S = 3.3                                           # page units per coherence unit
R_OUT, R_IN = np.sqrt(2 / 3), 1 / np.sqrt(6)


def coords(rho):
    return np.array([np.trace(F3 @ rho).real, np.trace(F8 @ rho).real])


def rho_of(x):
    return np.eye(3) / 3 + x[0] * F3 + x[1] * F8


def page(x):
    return S * np.asarray(x)


def main():
    corners = [coords(np.diag(e)) for e in np.eye(3)]
    for c in corners:
        assert np.isclose(np.linalg.norm(c), R_OUT)
    mids = [(corners[i] + corners[(i + 1) % 3]) / 2 for i in range(3)]
    for m in mids:
        assert np.isclose(np.linalg.norm(m), R_IN)
    bad = np.array([0.0, 0.8])
    assert np.linalg.norm(bad) < R_OUT and np.linalg.eigvalsh(rho_of(bad)).min() < 0
    # the triangle is exactly the positive part of the slice (checked on a grid)
    for x in np.random.default_rng(0).uniform(-0.9, 0.9, (4000, 2)):
        lam = np.linalg.eigvalsh(rho_of(x)).min()
        bary = np.linalg.solve(np.vstack([np.array(corners).T, np.ones(3)]), np.append(x, 1))
        assert (lam >= -1e-12) == (bary.min() >= -1e-12)
    b = []
    t = np.linspace(0, 2 * np.pi, 200)
    b.append("\\draw[dashedink] (0,0) circle (%.3f);" % (S * R_OUT))
    b.append("\\fill[ndRose, fill opacity=0.35] (0,0) circle (%.3f);" % (S * R_OUT))
    b.append("\\fill[ndSky] %s -- cycle;" % " -- ".join(N.pt(page(c)) for c in corners))
    b.append("\\draw[line width=1.4pt, ndBlue] %s -- cycle;" % " -- ".join(N.pt(page(c)) for c in corners))
    b.append("\\draw[line width=0.8pt, ndGreen] (0,0) circle (%.3f);" % (S * R_IN))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[right] {$x_3$};" % (-S * 0.95, S * 0.95))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[above] {$x_8$};" % (-S * 0.95, S * 0.95))
    labels = ["\\ket0\\bra0", "\\ket1\\bra1", "\\ket2\\bra2"]
    anchors = ["above right", "above left", "below"]
    for c, lab, anc in zip(corners, labels, anchors):
        b.append("\\node[dot, fill=ndBlue, label={[font=\\footnotesize]%s:$%s$}] at %s {};" % (anc, lab, N.pt(page(c))))
    for m in mids:
        b.append("\\node[dot, fill=ndGreen] at %s {};" % N.pt(page(m)))
    b.append("\\node[dot, fill=ndInk!60, label={[font=\\scriptsize]below right:$\\tfrac13\\Id$}] at (0,0) {};")
    b.append("\\node[dot, fill=ndRed, minimum size=4.2pt] at %s {};" % N.pt(page(bad)))
    b.append("\\draw[->, >=stealth, line width=0.7pt, ndRed] (0,0) -- %s;" % N.pt(page(bad) + np.array([0, -0.07])))
    # notes
    x0 = S * 0.95 + 0.4
    b.append("\\node[note, anchor=west, text=ndBlue] (a) at (%.2f,1.9) {states in this slice:\\\\ a \\emph{triangle}, corners pure};" % x0)
    b.append("\\draw[pointer] (a.west) to[bend right=15] %s;" % N.pt(page(0.75 * corners[0] + 0.25 * corners[2]) + np.array([0.06, 0.03])))
    b.append("\\node[note, anchor=west] (o) at (%.2f,0.6) {outer circle $\\norm{\\vec x}=\\sqrt{2/3}$:\\\\ touched only at the pure states};" % x0)
    b.append("\\draw[pointer] (o.west) to[bend left=10] %s;" % N.pt(page(R_OUT * np.array([np.cos(0.12), np.sin(0.12)]))))
    b.append("\\node[note, anchor=west, text=ndGreen!70!black] (i) at (%.2f,-0.7) {inner circle $\\norm{\\vec x}=1/\\sqrt6$:\\\\ touched at the midpoints};" % x0)
    b.append("\\draw[pointer] (i.west) to[bend left=15] %s;" % N.pt(page(mids[1 if mids[1][0] > 0 else 2]) + np.array([0.08, -0.05])))
    b.append("\\node[note, anchor=east, text=ndRed] (r) at (%.2f,%.2f) {$0.8\\,F_8$: inside the outer\\\\"
             "circle, but not a state\\\\ (negative eigenvalue):\\\\ no symmetry $\\vec x\\mapsto-\\vec x$};" % (-S * 0.45, S * 0.95))
    b.append("\\draw[pointer] (r.east) to[bend left=15] %s;" % N.pt(page(bad) + np.array([-0.1, 0.02])))
    N.write("ch02_qutrit", b)
    print("written figures/ch02_qutrit.tex; corners", np.round(corners, 3).tolist())


if __name__ == "__main__":
    main()
