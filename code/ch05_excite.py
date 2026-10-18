"""ch05_excite.py -- what an experiment reveals: excitation and observation.

The qubit of the chapter (ch05_spectrum), prepared in |1> (Bloch vector
(0,0,-1), on the z-axis) or in the tilted state X0, and measured along x or
along n = (x + z)/sqrt 2.  A mode lambda_i appears in the data with the weight
    r_i = (C v_i)(w_i^dagger x0),
the product of how much the measurement sees of it and how much the
preparation excites it.  The four experiments:
    |1>, sigma_x : nothing excited is seen; the data vanish identically,
    |1>, sigma_n : the precession -Gamma +- i omega is never excited,
    X0,  sigma_x : the relaxation 0, -gamma_1 is never seen,
    X0,  sigma_n : all four modes appear.
Writes figures/ch05_excite.tex.
"""
import numpy as np
import needham as N
from ch05_spectrum import qubit_A, outputs, X0, CX, CN, OMEGA, GAMMA, GAMMA1

ONE = np.array([0.0, 0.0, -1.0, 1.0])              # |1>, the excited state
TOL = 1e-12


def classify(A, C, x0):
    """For each eigenvalue: (lambda, seen, excited) with seen = C v != 0, excited = w^dag x0 != 0."""
    lam, V = np.linalg.eig(A)
    W = np.linalg.inv(V)
    seen = np.abs(np.asarray(C).reshape(-1) @ V) > TOL
    exc = np.abs(W @ x0) > TOL
    return lam, seen, exc


def pole_inset(b, lam, seen, exc, o):
    """A small lambda-plane at page position o (the origin of the plane)."""
    sx, sy = 1.5, 0.27
    P = lambda z: (o[0] + sx * z.real, o[1] + sy * z.imag)
    b.append("\\draw[faint, ->, >=stealth] (%.2f,%.2f) -- (%.2f,%.2f) node[right, font=\\tiny, text=ndInk!70] {Re};" % (o[0] - 0.95, o[1], o[0] + 0.3, o[1]))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,%.2f) -- (%.2f,%.2f) node[above, font=\\tiny, text=ndInk!70] {Im\\,$\\lambda$};" % (o[0], o[1] - 1.0, o[0], o[1] + 1.05))
    for z, s, e in zip(lam, seen, exc):
        x, y = P(z)
        if s and e:
            b.append("\\fill[ndRed] (%.3f,%.3f) circle (2.2pt);" % (x, y))
        elif not s:                                   # not seen: a crossed ring
            b.append("\\draw[line width=0.6pt, ndInk!60] (%.3f,%.3f) circle (2.4pt) (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);"
                     % (x, y, x - 0.06, y - 0.06, x + 0.06, y + 0.06, x - 0.06, y + 0.06, x + 0.06, y - 0.06))
        else:                                         # seen, but not excited: an empty ring
            b.append("\\draw[line width=0.7pt, ndInk!60, fill=white] (%.3f,%.3f) circle (2.4pt);" % (x, y))


def state_icon(b, c, v, label):
    """The x-z section of the Bloch ball with the initial Bloch vector."""
    r = 0.5
    b.append("\\fill[ndSky, fill opacity=0.6] (%.2f,%.2f) circle (%.2f);" % (c[0], c[1], r))
    b.append("\\draw[ink] (%.2f,%.2f) circle (%.2f);" % (c[0], c[1], r))
    b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f) (%.2f,%.2f) -- (%.2f,%.2f);"
             % (c[0] - r - 0.1, c[1], c[0] + r + 0.1, c[1], c[0], c[1] - r - 0.1, c[0], c[1] + r + 0.1))
    b.append("\\node[font=\\scriptsize, text=ndInk!70, anchor=west] at (%.2f,%.2f) {$x$};" % (c[0] + r + 0.08, c[1]))
    b.append("\\node[font=\\scriptsize, text=ndInk!70, anchor=south] at (%.2f,%.2f) {$z$};" % (c[0], c[1] + r + 0.08))
    b.append("\\draw[vec, ndInk] (%.2f,%.2f) -- (%.2f,%.2f);" % (c[0], c[1], c[0] + r * v[0], c[1] + r * v[2]))
    b.append("\\node[font=\\small, align=center, anchor=north] at (%.2f,%.2f) {%s};" % (c[0], c[1] - r - 0.15, label))


def axis_icon(b, c, d, label, col):
    """A measurement direction in the x-z plane, drawn as a short arrow."""
    b.append("\\draw[vec, %s] (%.2f,%.2f) -- (%.2f,%.2f);" % (col, c[0] - 0.3 * d[0], c[1] - 0.3 * d[1], c[0] + 0.3 * d[0], c[1] + 0.3 * d[1]))
    b.append("\\node[font=\\small, text=%s, anchor=west] at (%.2f,%.2f) {%s};" % (col, c[0] + 0.45, c[1], label))


def main():
    A = qubit_A()
    preps = [(ONE, "prepared\\\\ in $\\ket1$"), (X0, "prepared\\\\ in $\\vec x_0$")]
    meas = [(CX, "measure $\\sigma_x$", (1.0, 0.0), "ndBlue"), (CN, "measure $\\sigma_n$", (np.sqrt(0.5), np.sqrt(0.5)), "ndGreen")]
    notes = {(0, 0): "nothing that is excited is seen:\\\\ the data vanish",
             (0, 1): "the precession is\\\\ never excited",
             (1, 0): "the relaxation is\\\\ never seen",
             (1, 1): "all four modes appear"}
    expect = {(0, 0): 0, (0, 1): 2, (1, 0): 2, (1, 1): 4}
    t = np.linspace(0, 12, 361)
    b = []
    for c, (C, mlab, d, col) in enumerate(meas):
        cx = 2.4 + 7.4 * c
        axis_icon(b, (cx + 1.2, 6.15), d, mlab, col)
    for r, (x0, plab) in enumerate(preps):
        cy = 3.35 * (1 - r)
        state_icon(b, (0.75, cy + 1.25), x0, plab)
        for c, (C, mlab, d, col) in enumerate(meas):
            cx = 2.4 + 7.4 * c
            y = outputs(A, C, x0, t)[:, 0]
            lam, seen, exc = classify(A, C, x0)
            assert int(np.sum(seen & exc)) == expect[(r, c)]
            if expect[(r, c)] == 0:
                assert np.abs(y).max() < 1e-12                  # the data vanish identically
            Pl = N.Plot((0, 12), (-1.0, 1.0), 3.9, 2.1, origin=(cx, cy + 0.2))
            b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$t$};" % (Pl.pt(0, 0), Pl.pt(12.6, 0)))
            b.append("\\draw[faint] %s -- %s;" % (Pl.pt(0, -1.0), Pl.pt(0, 1.0)))
            for v in (-1, 1):
                b.append("\\draw[faint] %s -- %s node[left, font=\\tiny, text=ndInk] {$%d$};" % (Pl.pt(0, v), Pl.pt(-0.25, v), v))
            b.append(Pl.curve(t, y, "line width=1.1pt, %s" % col))
            pole_inset(b, lam, seen, exc, (cx + 5.85, cy + 1.25))
            b.append("\\node[note, anchor=north] at (%.2f,%.2f) {%s};" % (cx + 2.6, cy + 0.05, notes[(r, c)]))
    b.append("\\node[note, anchor=north] at (%.2f,%.2f) {in each small $\\lambda$-plane: \\tikz[baseline=-0.6ex]\\fill[ndRed] (0,0) circle (2.2pt);\\ a pole present in the data,"
             " \\ \\tikz[baseline=-0.6ex]\\draw[line width=0.7pt, ndInk!60] (0,0) circle (2.4pt);\\ not excited ($\\vec w_i^\\dagger\\vec x_0=0$),"
             " \\ \\tikz[baseline=-0.6ex]\\draw[line width=0.6pt, ndInk!60] (0,0) circle (2.4pt) (-0.06,-0.06) -- (0.06,0.06) (-0.06,0.06) -- (0.06,-0.06);\\ not seen ($\\mathbf C\\vec v_i=0$)};"
             % (8.9, -0.75))
    N.write("ch05_excite", b)
    print("written figures/ch05_excite.tex")


if __name__ == "__main__":
    main()
