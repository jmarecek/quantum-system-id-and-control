"""ch02_jacobi.py -- the Jacobi-Lie bracket as the gap in a loop of flows.

Vector fields on the plane: V = d/dx (move right) and W = x d/dy (move up,
faster further to the right).  Their flows do not commute.  Following V, W,
-V, -W for time eps each misses the starting point by exactly
eps^2 [V, W] = eps^2 d/dy, in agreement with [V, W] = (DW) V - (DV) W.
Writes figures/ch02_jacobi.tex.
"""
import numpy as np
import needham as N

EPS = 1.1
X0, Y0 = 0.7, 0.4
S = 1.45                                   # page units per coordinate unit


def V(p):
    return np.array([1.0, 0.0])


def W(p):
    return np.array([0.0, p[0]])


def flow(field, p, t):
    """Exact flows of the two fields (both are integrable in closed form)."""
    x, y = p
    if field is V:
        return np.array([x + t, y])
    return np.array([x, y + t * x])


def bracket(p, h=1e-6):
    """(DW) V - (DV) W by central differences."""
    def D(F):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2); e[j] = h
            J[:, j] = (F(p + e) - F(p - e)) / (2 * h)
        return J
    return D(W) @ V(p) - D(V) @ W(p)


def page(p):
    return (S * p[0], S * p[1])


def main():
    start = np.array([X0, Y0])
    legs = [(V, EPS), (W, EPS), (V, -EPS), (W, -EPS)]
    pts = [start]
    for f, t in legs:
        pts.append(flow(f, pts[-1], t))
    gap = pts[-1] - start
    br = bracket(start)
    assert np.allclose(gap, EPS ** 2 * br), (gap, br)

    b = []
    # the field W, drawn faintly on a grid: arrows grow to the right
    for gx in np.arange(0.3, 2.1, 0.4):
        if min(abs(gx - X0), abs(gx - X0 - EPS)) < 0.15:     # keep clear of the loop
            continue
        for gy in np.arange(0.0, 2.6, 0.45):
            p = np.array([gx, gy])
            w = 0.18 * W(p)
            if abs(w[1]) > 1e-3:
                b.append("\\draw[->, >=stealth, line width=0.5pt, ndGreen!45] %s -- %s;"
                         % (N.pt(page(p)), N.pt(page(p + w))))
    b.append("\\draw[faint, ->, >=stealth] (%.2f,0) -- (%.2f,0) node[below] {$x$};" % (-0.5, S * 2.5))
    b.append("\\draw[faint, ->, >=stealth] (0,%.2f) -- (0,%.2f) node[left] {$y$};" % (-0.4, S * 2.3))
    # the four legs
    cols = ["ndBlue", "ndGreen", "ndBlue", "ndGreen"]
    labels = ["$V$ for $\\epsilon$", "$W$ for $\\epsilon$", "$V$ back", "$W$ back"]
    anchors = ["below", "right", "above", "left"]
    for k in range(4):
        a, c = page(pts[k]), page(pts[k + 1])
        b.append("\\draw[vec, %s] %s -- %s;" % (cols[k], N.pt(a), N.pt(c)))
        f = 0.3 if k == 3 else 0.5                     # keep "W back" clear of the gap
        m = (a[0] + f * (c[0] - a[0]), a[1] + f * (c[1] - a[1]))
        b.append("\\node[%s, font=\\footnotesize, text=%s] at %s {%s};" % (anchors[k], cols[k], N.pt(m), labels[k]))
    # the gap
    b.append("\\draw[line width=2.2pt, ndRed] %s -- %s;" % (N.pt(page(start)), N.pt(page(pts[-1]))))
    b.append("\\node[dot, label={[font=\\footnotesize]below left:start}] at %s {};" % N.pt(page(start)))
    b.append("\\node[dot, fill=ndRed] at %s {};" % N.pt(page(pts[-1])))
    b.append("\\node[note, anchor=east] (g) at (%.2f,%.2f) {the loop does not close:\\\\"
             "the gap is $\\epsilon^2\\,[V,W]$};" % (-0.35, S * (Y0 + 0.2)))
    b.append("\\draw[pointer] (g.east) to[bend right=25] %s;"
             % N.pt(page(start + np.array([-0.03, 0.5 * EPS ** 2]))))
    b.append("\\node[note, anchor=west] (w) at (%.2f,%.2f) {$W=x\\,\\partial_y$ pushes up,\\\\"
             "harder further right};" % (S * 2.45, S * 1.75))
    b.append("\\node[note, anchor=west] at (%.2f,%.2f) {$V=\\partial_x$, $\\;[V,W]=\\partial_y$};"
             % (S * 2.45, S * 0.9))
    N.write("ch02_jacobi", b)
    print("written figures/ch02_jacobi.tex; gap", gap, "= eps^2 *", br)


if __name__ == "__main__":
    main()
