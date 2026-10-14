"""ch04_kalman.py -- what the Kalman rank conditions mean geometrically.

A linear system x' = A x + B u, y = C x on R^3 whose controllability matrix
has rank 2 and whose observability matrix has rank 2.  It is obtained from a
damped rotation in the (e1, e2)-plane with a decoupled decay along e3, with
B = e1 and C = e1^T, by an oblique change of coordinates S, so that neither
subspace is aligned with the axes.
  (a) From x(0) = 0, every input keeps the state in the reachable plane
      ran(controllability matrix) = S span{e1, e2}.
  (b) Two initial states that differ along the unobservable line
      ker(observability matrix) = S span{e3} give identical outputs.
Writes figures/ch04_kalman.tex.
"""
import numpy as np
from scipy.linalg import expm, null_space
import needham as N

A0 = np.array([[-0.15, -1.0, 0.0], [1.0, -0.15, 0.0], [0.0, 0.0, -0.35]])
B0 = np.array([[1.0], [0.0], [0.0]])
C0 = np.array([[1.0, 0.0, 0.0]])
th = 0.5
ROT = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
SHEAR = np.array([[1.0, 0.0, 0.55], [0.0, 1.0, -0.35], [0.0, 0.0, 1.0]])
S = ROT @ SHEAR
A, B, C = S @ A0 @ np.linalg.inv(S), S @ B0, C0 @ np.linalg.inv(S)
V = N.View(az=35.0, el=20.0, R=2.1, shift=(0.0, 0.0))
DT, TEND = 0.02, 14.0


def simulate(x0, u_of_t):
    """Exact zero-order-hold steps of x' = A x + B u."""
    Ad = expm(A * DT)
    Bd = np.linalg.solve(A, (Ad - np.eye(3)) @ B)
    xs, x = [x0], x0.copy()
    for k in range(int(TEND / DT)):
        x = Ad @ x + (Bd * u_of_t(k * DT)).ravel()
        xs.append(x)
    return np.array(xs)


def main():
    CM = np.hstack([B, A @ B, A @ A @ B])
    OM = np.vstack([C, C @ A, C @ A @ A])
    assert np.linalg.matrix_rank(CM) == 2 and np.linalg.matrix_rank(OM) == 2
    normal = np.cross(S[:, 0], S[:, 1]); normal /= np.linalg.norm(normal)
    kvec = S[:, 2] / np.linalg.norm(S[:, 2])
    ker = null_space(OM)[:, 0]
    assert abs(abs(ker @ kvec) - 1) < 1e-10                   # the unobservable line is S e3
    assert abs(kvec @ normal) > 0.3                            # oblique to the plane, not orthogonal

    rng = np.random.default_rng(3)
    driven = []
    for _ in range(2):
        levels = rng.uniform(-0.45, 0.45, size=8)
        driven.append(simulate(np.zeros(3), lambda t, lv=levels: lv[min(int(t / (TEND / 8)), 7)]))
    assert max(np.abs(d @ normal).max() for d in driven) < 1e-10   # inputs never leave the plane

    x0 = S @ np.array([1.0, 0.25, 0.0])
    x1 = x0 + 0.9 * S[:, 2]
    free0, free1 = simulate(x0, lambda t: 0.0), simulate(x1, lambda t: 0.0)
    y0, y1 = free0 @ C.ravel(), free1 @ C.ravel()
    assert np.max(np.abs(y0 - y1)) < 1e-10 and np.max(np.abs(free0 - free1)) > 0.5
    print("max output difference %.1e; reachable plane, unobservable line checked" % np.max(np.abs(y0 - y1)))

    b = []
    # (a)/(b) the state space
    L = 1.3
    corners = [S @ np.array([sx * L, sy * L, 0.0]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    b.append("\\filldraw[fill=ndSky, fill opacity=0.55, draw=ndBlue!60, line width=0.6pt] %s -- cycle;"
             % " -- ".join(V.pt(c) for c in corners))
    line = [S @ np.array([0, 0, t]) for t in (-1.5, 1.5)]
    b.append("\\draw[line width=1.0pt, ndRed, dash pattern=on 4pt off 2pt] %s -- %s;" % (V.pt(line[0]), V.pt(line[1])))
    b.append("\\node[dot] at %s {};" % V.pt(np.zeros(3)))
    for d, col in zip(driven, ["ndBlue", "ndGreen"]):
        b.append(N.polyline([V.P(p) for p in d[::3]], "line width=1.0pt, %s" % col))
    b.append(N.polyline([V.P(p) for p in free0[::3]], "line width=1.4pt, ndGold"))
    b.append(N.polyline([V.P(p) for p in free1[::3]], "line width=1.4pt, ndRed"))
    b.append("\\node[dot, fill=ndGold, minimum size=4.4pt] at %s {};" % V.pt(x0))
    b.append("\\node[dot, fill=ndRed, minimum size=4.4pt] at %s {};" % V.pt(x1))
    b.append("\\draw[pointer, ndRed!70, shorten <=3pt] %s -- %s;" % (V.pt(x0), V.pt(x1)))
    b.append("\\node[note, anchor=west, text=ndBlue] (pl) at (3.0,2.0) {reachable from $\\vec x(0)=0$:\\\\ the plane $\\ran\\CM$ (rank $2$);\\\\ inputs never leave it};")
    b.append("\\draw[pointer] (pl.west) to[bend right=15] %s;" % V.pt(S @ np.array([0.9, 1.2, 0.0])))
    b.append("\\node[note, anchor=west, text=ndRed] (ln) at (2.7,-1.75) {unobservable: the line $\\ker\\OM$};")
    b.append("\\draw[pointer] (ln.west) to[bend left=15] %s;" % V.pt(line[0] * 0.85))
    b.append("\\node[note, anchor=east, text=ndGold!70!black] (st) at (-2.3,2.0) {two initial states that\\\\ differ along $\\ker\\OM$};")
    b.append("\\draw[pointer] (st.east) to[bend left=15] %s;" % V.pt(x1))
    b.append("\\node[font=\\small] at (0,-2.75) {(a) the state space $\\R^3$};")
    # (c) the outputs
    t = np.arange(len(y0)) * DT
    Pl = N.Plot((0, TEND), (-1.0, 1.1), 5.6, 3.4, origin=(7.6, -1.6))
    b += Pl.axes("$t$", "$y(t)$", yticks=[(0, "$0$")], yzero=0.0)
    b.append(Pl.curve(t[::3], y0[::3], "line width=2.2pt, ndGold"))
    b.append(Pl.curve(t[::3], y1[::3], "line width=1.0pt, ndRed, dash pattern=on 3pt off 2pt"))
    b.append("\\node[note, anchor=south west] at %s {identical outputs:\\\\ the data cannot tell\\\\ the two states apart};" % Pl.pt(5.2, 0.35))
    b.append("\\node[font=\\small] at %s {(b) the outputs $\\mathbf C\\vec x(t)$};" % N.pt(Pl.P(7, -1.0) + np.array([0, -0.55])))
    N.write("ch04_kalman", b)
    print("written figures/ch04_kalman.tex")


if __name__ == "__main__":
    main()
