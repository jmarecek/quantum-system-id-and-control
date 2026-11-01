"""ch09_kinematic.py -- the kinematic landscape of U(2) has no traps.

J(U) = Re Tr(V^dag U)/2 on U(2).  Write W = V^dag U = Q diag(e^{i alpha}, e^{i beta}) Q^dag;
then J = (cos alpha + cos beta)/2 depends only on the eigenphases, and we draw it
on the torus of eigenphases, the square [-pi, pi]^2 with opposite sides
identified.  Its critical points are the Hermitian W: the maximum W = I at
(0, 0), the minimum W = -I at (pi, pi), and the saddles at (0, pi) and (pi, 0),
where W has the eigenvalues 1 and -1 (Theorem 9.x).  On U(2) the gradient of J
at U is U i G with G = i(W - W^dag)/4 (Hermitian), which commutes with W, so the
steepest-ascent flow U -> U exp(i eps G) keeps the eigenvectors and moves the
eigenphases by alpha' = -sin(alpha)/2, beta' = -sin(beta)/2: the projection of
the flow onto the torus is the gradient flow of J there.
(a) Contours of J, its critical points, and streamlines of the flow.
(b) Steepest ascent on the whole group U(2) from 40 Haar-random starts,
    computed with finite steps and drawn through the eigenphases of W (in the
    eigenbasis of the start): every run ends at the maximum W = I.  The saddles
    attract only the lines alpha = pi or beta = pi, of measure zero.
(c) The phase-insensitive fidelity F = |Tr W|^2/4 = cos^2((alpha - beta)/2) on
    the same torus: its maxima fill the diagonal alpha = beta, the gates equal
    to V up to a global phase, its minima the lines alpha - beta = +-pi, and it
    has no other critical points; ascent from the same 40 starts ends on the
    diagonal.
Writes figures/ch09_kinematic.tex.
"""
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import needham as N

PI = np.pi


def haar(rng, n=2):
    Z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(Z)
    d = np.diag(r)
    return q * (d / abs(d))


def J(W):
    return np.real(np.trace(W)) / 2


def Fpi(W):
    return abs(np.trace(W)) ** 2 / 4


def grad_J(W):
    return 1j * (W - W.conj().T) / 4                                 # dJ(W e^{i eps K})/d eps = Tr(G K)


def grad_F(W):
    t = np.trace(W)
    return 1j * (np.conj(t) * W - t * W.conj().T) / 4


def ascend(W0, grad, steps=400, eps=0.15):
    """Steepest ascent W -> W exp(i eps G) on U(2); returns the eigenphases in the eigenbasis of W0."""
    _, Q = np.linalg.eig(W0)
    Q, _ = np.linalg.qr(Q)
    W, out = W0.copy(), []
    for _ in range(steps + 1):
        D = Q.conj().T @ W @ Q
        out.append(np.angle(np.diag(D)))
        W = W @ expm(1j * eps * grad(W))
    return np.array(out), W


def hessian_signs(alpha, beta, h=1e-4):
    """Second derivatives of J along W e^{i eps |j><j|} and W e^{i eps sigma_x} at diag(e^{i alpha}, e^{i beta})."""
    W = np.diag([np.exp(1j * alpha), np.exp(1j * beta)])
    out = []
    for K in (np.diag([1.0, 0.0]), np.diag([0.0, 1.0]), np.array([[0, 1.0], [1.0, 0]])):
        f = lambda e: J(W @ expm(1j * e * K))
        out.append((f(h) - 2 * f(0) + f(-h)) / h ** 2)
    return np.array(out)


def checks(rng):
    # J depends only on the eigenphases
    for _ in range(20):
        W = haar(rng)
        a = np.angle(np.linalg.eigvals(W))
        assert abs(J(W) - np.cos(a).sum() / 2) < 1e-12
        assert np.allclose(grad_J(W) @ W, W @ grad_J(W))            # the gradient commutes with W
    # critical points: the gradient vanishes at the four Hermitian diagonal points, and the Hessian signs
    kinds = {}
    for a, b in [(0, 0), (PI, PI), (0, PI), (PI, 0)]:
        W = np.diag([np.exp(1j * a), np.exp(1j * b)])
        assert np.linalg.norm(grad_J(W)) < 1e-12
        hs = hessian_signs(a, b)
        kinds[(a, b)] = "max" if hs.max() < -1e-6 else ("min" if hs.min() > 1e-6 else "saddle")
    assert kinds == {(0, 0): "max", (PI, PI): "min", (0, PI): "saddle", (PI, 0): "saddle"}
    # off the diagonal unitaries, the gradient does not vanish: sample
    for _ in range(200):
        W = haar(rng)
        if np.linalg.norm(W - W.conj().T) > 1e-3:
            assert np.linalg.norm(grad_J(W)) > 1e-4


def streamline(a, b, dt=0.02, n=600):
    pts = [(a, b)]
    for _ in range(n):
        a, b = a - dt * np.sin(a) / 2, b - dt * np.sin(b) / 2
        pts.append((a, b))
        if abs(a) + abs(b) < 0.08:
            break
    return np.array(pts)


def torus_panel(b, P, Z, g, levels, label_txt):
    b.append("\\fill[ndPaper] %s rectangle %s;" % (N.pt(P(-PI, -PI)), N.pt(P(PI, PI))))
    cs = plt.contour(g, g, Z, levels=levels)
    for lev, segs in zip(levels, cs.allsegs):
        shade = int(15 + 75 * (lev - min(levels)) / (max(levels) - min(levels)))
        for seg in segs:
            if len(seg) > 2:
                b.append(N.polyline([P(*q) for q in seg[::2]] + [P(*seg[-1])], "line width=0.45pt, ndBlue!%d" % shade))
    plt.close("all")


def frame(b, P, xlab=True):
    b.append("\\draw[ink] %s rectangle %s;" % (N.pt(P(-PI, -PI)), N.pt(P(PI, PI))))
    for v, lab in [(-PI, "$-\\pi$"), (0, "$0$"), (PI, "$\\pi$")]:
        b.append("\\node[font=\\scriptsize, anchor=north] at %s {%s};" % (N.pt(P(v, -PI) + np.array([0, -0.03])), lab))
        b.append("\\node[font=\\scriptsize, anchor=east] at %s {%s};" % (N.pt(P(-PI, v) + np.array([-0.03, 0])), lab))
    b.append("\\node[font=\\scriptsize] at %s {$\\alpha$};" % N.pt(P(PI, -PI) + np.array([0.3, -0.25])))
    b.append("\\node[font=\\scriptsize] at %s {$\\beta$};" % N.pt(P(-PI, PI) + np.array([-0.3, 0.22])))


def marker(b, q, kind):
    if kind == "max":
        b.append("\\node[dot, fill=ndGold, minimum size=4.6pt] at %s {};" % N.pt(q))
    elif kind == "min":
        b.append("\\node[dot, fill=ndInk, minimum size=4.0pt] at %s {};" % N.pt(q))
    else:
        b.append("\\draw[ndInk, fill=white, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- cycle;"
                 % (q[0] - 0.09, q[1], q[0], q[1] + 0.09, q[0] + 0.09, q[1], q[0], q[1] - 0.09))


def main():
    rng = np.random.default_rng(3)
    checks(rng)
    S = 0.62
    g = np.linspace(-PI, PI, 241)
    A, B = np.meshgrid(g, g)
    b = []
    # ---- (a) J on the torus, critical points and streamlines
    oa = np.array([2.2, 2.3])
    P = lambda x, y: oa + S * np.array([x, y])
    torus_panel(b, P, (np.cos(A) + np.cos(B)) / 2, g, list(np.linspace(-0.9, 0.9, 10)), "")
    starts = [(a0, b0) for a0 in (-3.0, -1.6, 1.6, 3.0) for b0 in (-3.0, 3.0)] + [(-3.0, -0.8), (3.0, 0.8), (0.8, -3.0), (-0.8, 3.0),
                                                                                     (-2.2, 1.2), (2.2, -1.2), (1.3, 2.4), (-1.3, -2.4)]
    for a0, b0 in starts:
        sl = streamline(a0, b0)
        b.append(N.polyline([P(*q) for q in sl[::4]], "line width=0.6pt, ndInk!65"))
        k = len(sl) // 3
        d = sl[k + 1] - sl[k]
        q0 = P(*sl[k])
        b.append("\\draw[->, >=stealth, line width=0.6pt, ndInk!65] %s -- %s;" % (N.pt(q0), N.pt(q0 + 0.01 * d / np.linalg.norm(d))))
    for (x, y), kind in [((0, 0), "max"), ((PI, PI), "min"), ((-PI, PI), "min"), ((PI, -PI), "min"), ((-PI, -PI), "min"),
                         ((0, PI), "saddle"), ((0, -PI), "saddle"), ((PI, 0), "saddle"), ((-PI, 0), "saddle")]:
        marker(b, P(x, y), kind)
    frame(b, P)
    b.append("\\node[note, anchor=south, text=ndGold!55!ndInk] at %s {gold: $W=\\Id$, the only maximum};" % N.pt(P(0, PI) + np.array([0, 0.25])))
    b.append("\\node[note, anchor=north] at %s {diamonds: saddles, eigenvalues $1$, $-1$;\\\\ black: the minimum $W=-\\Id$};" % N.pt(P(0, -PI) + np.array([0, -0.38])))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) $\\operatorname{Re}\\Tr(V^\\dagger U)/2$};" % N.pt(P(0, -PI) + np.array([0, -1.1])))

    # ---- (b) steepest ascent on U(2) from Haar-random starts
    ob = np.array([7.15, 2.3])
    Q = lambda x, y: ob + S * np.array([x, y])
    torus_panel(b, Q, (np.cos(A) + np.cos(B)) / 2, g, list(np.linspace(-0.9, 0.9, 10)), "")
    ends = []
    rs = np.random.default_rng(17)
    for _ in range(40):
        W0 = haar(rs)
        ph, Wend = ascend(W0, grad_J)
        assert abs(J(Wend) - 1) < 1e-6
        ends.append(Wend)
        ph = np.unwrap(ph, axis=0)
        seg = [Q(*q) for q in ph[::3]]
        b.append(N.polyline(seg, "line width=0.55pt, ndRed!85"))
        b.append("\\fill[ndRed] %s circle (0.9pt);" % N.pt(Q(*ph[0])))
    marker(b, Q(0, 0), "max")
    for (x, y) in [(0, PI), (0, -PI), (PI, 0), (-PI, 0)]:
        marker(b, Q(x, y), "saddle")
    frame(b, Q)
    b.append("\\node[note, anchor=south] at %s {40 random starts on $\\mathrm U(2)$:\\\\ all reach $W=\\Id$};" % N.pt(Q(0, PI) + np.array([0, 0.18])))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) steepest ascent};" % N.pt(Q(0, -PI) + np.array([0, -1.1])))

    # ---- (c) the phase-insensitive fidelity
    oc = np.array([12.1, 2.3])
    R = lambda x, y: oc + S * np.array([x, y])
    torus_panel(b, R, np.cos((A - B) / 2) ** 2, g, [0.1, 0.3, 0.5, 0.7, 0.9], "")
    b.append("\\draw[line width=1.6pt, ndGold] %s -- %s;" % (N.pt(R(-PI, -PI)), N.pt(R(PI, PI))))
    b.append("\\draw[line width=1.0pt, ndInk, dash pattern=on 3pt off 2pt] %s -- %s;" % (N.pt(R(0, -PI)), N.pt(R(PI, 0))))
    b.append("\\draw[line width=1.0pt, ndInk, dash pattern=on 3pt off 2pt] %s -- %s;" % (N.pt(R(-PI, 0)), N.pt(R(0, PI))))
    rs = np.random.default_rng(17)
    for _ in range(40):
        W0 = haar(rs)
        ph, Wend = ascend(W0, grad_F)
        assert abs(Fpi(Wend) - 1) < 1e-6 and np.allclose(grad_F(Wend) @ Wend, Wend @ grad_F(Wend))
        ph = np.unwrap(ph, axis=0)[::3]
        dif = ph[-1, 0] - ph[-1, 1]
        assert abs((dif + PI) % (2 * PI) - PI) < 1e-6                    # ends on the diagonal of the torus
        wr = (ph + PI) % (2 * PI) - PI                                   # back onto the square, split at the seams
        cut = np.where(np.any(np.abs(np.diff(wr, axis=0)) > PI, axis=1))[0] + 1
        for sg in np.split(np.arange(len(wr)), cut):
            if len(sg) > 1:
                b.append(N.polyline([R(*q) for q in wr[sg]], "line width=0.55pt, ndRed!85"))
        b.append("\\fill[ndRed] %s circle (0.9pt);" % N.pt(R(*wr[0])))
    frame(b, R)
    b.append("\\node[note, anchor=south, text=ndGold!55!ndInk] at %s {maxima: $W=\\ee^{\\imath\\varphi}\\Id$};" % N.pt(R(0, PI) + np.array([0, 0.18])))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) $\\abs{\\Tr(V^\\dagger U)}^2/4$};" % N.pt(R(0, -PI) + np.array([0, -1.1])))
    b.append("\\node[note, anchor=north] at %s {dashed: the minima};" % N.pt(R(0, -PI) + np.array([0, -0.38])))
    N.write("ch09_kinematic", b)


if __name__ == "__main__":
    main()
    print("written figures/ch09_kinematic.tex")
