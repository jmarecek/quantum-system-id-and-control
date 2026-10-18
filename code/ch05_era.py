"""ch05_era.py -- Ho--Kalman / ERA as a pipeline of seven small pictures.

From K = 40 exact samples, tau = 0.25, of the readout sigma_n of the
running example (ch05_spectrum), with R = 5 block rows:
  1. the data y_k = C G^k x_0;
  2. the Hankel matrices H(0) and H(1) (H(1) is H(0) shifted by one sample);
  3. the SVD H(0) = P S Q^T: four singular values stand out, the fifth is zero;
  4. the balanced factors  O = P_1 S^{1/2},  C = S^{1/2} Q_1^T;
  5. the step matrix  A_d = S^{-1/2} P_1^T H(1) Q_1 S^{-1/2}  (4 x 4);
  6. its eigenvalues z_i = e^{lambda_i tau}, inside or on the unit circle;
  7. lambda_i = log(z_i) / tau, the poles, which coincide with the true ones.
Every picture is computed from ho_kalman of ch05_spectrum.
Writes figures/ch05_era.tex.
"""
import numpy as np
from scipy.linalg import expm
import needham as N
from ch05_spectrum import qubit_A, hankel, ho_kalman, X0, CN, TAU, POLES

K, R, NORD = 40, 5, 4


def grid(M, x0, y0, cw, vmax=None, outline="ink", ncols=None):
    """Matrix M as coloured cells, top-left corner at (x0, y0)."""
    M = np.asarray(M)
    if ncols:
        M = M[:, :ncols]
    vmax = vmax or np.abs(M).max()
    out = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j] / vmax
            col = "ndBlue" if v > 0 else "ndRed"
            out.append("\\fill[%s!%d] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                       % (col, int(8 + 72 * min(abs(v), 1)), x0 + j * cw, y0 - i * cw, x0 + (j + 1) * cw, y0 - (i + 1) * cw))
    out.append("\\draw[%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (outline, x0, y0, x0 + M.shape[1] * cw, y0 - M.shape[0] * cw))
    return out


def main():
    A = qubit_A()
    G = expm(A * TAU)
    y = np.array([CN @ np.linalg.matrix_power(G, k) @ X0 for k in range(K)])
    Ad, Cd, x0d = ho_kalman(y, NORD, r=R)
    s = K - R
    H0, H1 = hankel(y, R, s), hankel(y[1:], R, s)
    P, S, Qt = np.linalg.svd(H0)
    P1, S1, Q1 = P[:, :NORD], S[:NORD], Qt[:NORD]
    O, Cs = P1 * np.sqrt(S1), np.sqrt(S1)[:, None] * Q1
    assert np.allclose(O @ Cs, H0) and S[NORD] / S[0] < 1e-12
    assert np.allclose(np.linalg.inv(np.diag(np.sqrt(S1))) @ P1.T @ H1 @ Q1.T @ np.linalg.inv(np.diag(np.sqrt(S1))), Ad)
    z = np.linalg.eigvals(Ad)
    lam = np.log(z.astype(complex)) / TAU
    assert np.allclose(np.sort_complex(lam), np.sort_complex(POLES), atol=1e-6)
    assert np.allclose(np.sort_complex(z), np.sort_complex(np.exp(POLES * TAU)), atol=1e-8)

    b = []
    top, bot = 2.05, -0.15                               # pictograms live between these heights
    stages = [(0.0, 2.0), (2.5, 4.35), (4.85, 6.25), (6.75, 8.75), (9.25, 10.55), (11.05, 12.95), (13.45, 15.35)]
    titles = ["the data", "arrange in\\\\ Hankel matrices", "SVD: count\\\\ large values", "split into\\\\ two factors",
              "solve for the\\\\ one-step matrix", "its\\\\ eigenvalues", "back to rates\\\\ and frequencies"]
    forms = ["$y_k=\\mathbf C\\mathbf G_\\tau^k\\vec x_0$", "$\\Hankel(0)$, and $\\Hankel(1)$\\\\ shifted by one step",
             "$\\Hankel(0)=\\mathbf P\\mathbf S\\mathbf Q^{\\top}$\\\\ keep $n=%d$" % NORD,
             "$\\Obsv=\\mathbf P_1\\mathbf S^{1/2}$\\\\ $\\Ctrb=\\mathbf S^{1/2}\\mathbf Q_1^{\\top}$",
             "$\\mathbf A_d=\\mathbf S^{-\\frac12}\\mathbf P_1^{\\top}$\\\\ $\\cdot\\Hankel(1)\\mathbf Q_1\\mathbf S^{-\\frac12}$",
             "$z_i=\\ee^{\\lambda_i\\tau}$", "$\\lambda_i=\\frac1\\tau\\log z_i$"]
    for k, ((xa, xb), ti, fo) in enumerate(zip(stages, titles, forms)):
        xm = (xa + xb) / 2
        b.append("\\node[note, anchor=south] at (%.3f,%.3f) {\\textup{\\textbf{%d}}\\\\ %s};" % (xm, top + 0.3, k + 1, ti))
        b.append("\\node[font=\\scriptsize, align=center, anchor=north] at (%.3f,%.3f) {%s};" % (xm, bot - 0.15, fo))
        if k:
            b.append("\\draw[->, >=stealth, line width=1.0pt, ndInk!70] (%.3f,%.3f) -- (%.3f,%.3f);"
                     % (stages[k - 1][1] + 0.08, 0.95, xa - 0.08, 0.95))
    # 1. stems
    xa, xb = stages[0]
    kk = np.arange(14)
    sx, sy, y0 = (xb - xa) / 14, 1.0 / np.abs(y[:14]).max(), 0.95
    b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (xa, y0, xb, y0))
    for k in kk:
        px, py = xa + sx * (k + 0.5), y0 + sy * y[k, 0]
        b.append("\\draw[line width=0.8pt, ndInk!70] (%.3f,%.3f) -- (%.3f,%.3f);" % (px, y0, px, py))
        b.append("\\fill[ndInk] (%.3f,%.3f) circle (1.2pt);" % (px, py))
    # 2. two Hankel matrices, H(1) behind H(0)
    xa, xb = stages[1]
    cw = 0.19
    vmax = np.abs(H0).max()
    b += grid(H1, xa + 0.42, top - 0.05, cw, vmax, ncols=7)
    b.append("\\node[font=\\scriptsize, anchor=east] at (%.3f,%.3f) {$\\Hankel(1)$};" % (xa + 0.42, top - 0.32))
    b += grid(H0, xa, top - 0.7, cw, vmax, ncols=7)
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$\\Hankel(0)$};" % (xa + 3.5 * cw, top - 0.7 - R * cw))
    # 3. singular values as bars, log scale, cut after n
    xa, xb = stages[2]
    lo = -16.0
    hb = lambda v: 1.9 * (np.log10(max(v, 10 ** lo)) - lo) / (-lo)
    for i, v in enumerate(S / S[0]):
        x = xa + 0.15 + 0.25 * i
        col = "ndGreen" if i < NORD else "ndInk!40"
        b.append("\\fill[%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, x, bot + 0.05, x + 0.17, bot + 0.05 + max(hb(v), 0.04)))
    xc = xa + 0.15 + 0.25 * NORD - 0.04
    b.append("\\draw[dashedink, ndRed] (%.3f,%.3f) -- (%.3f,%.3f);" % (xc, bot, xc, top))
    b.append("\\node[font=\\scriptsize, text=ndRed, anchor=north west, align=left] at (%.3f,%.3f) {cut};" % (xc + 0.02, top))
    b.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (xa + 0.05, bot + 0.05, xb, bot + 0.05))
    # 4. the balanced factors
    xa, xb = stages[3]
    cw4 = 0.19
    vm = max(np.abs(O).max(), np.abs(Cs).max())
    b += grid(O, xa, top - 0.25, cw4, vm)
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$\\Obsv$};" % (xa + 2 * cw4, top - 0.25 - R * cw4))
    b += grid(Cs, xa + 4 * cw4 + 0.12, top - 0.25, cw4, vm, ncols=6)
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$\\Ctrb$};" % (xa + 4 * cw4 + 0.12 + 3 * cw4, top - 0.25 - NORD * cw4))
    # 5. A_d
    xa, xb = stages[4]
    cw5 = 0.3
    b += grid(Ad, xa + 0.05, top - 0.35, cw5)
    b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$\\mathbf A_d$, $4\\times4$};" % (xa + 0.05 + 2 * cw5, top - 0.35 - 4 * cw5))
    # 6. z-plane
    xa, xb = stages[5]
    c, Rz = np.array([(xa + xb) / 2, 0.95]), 0.85
    Z = lambda w: c + Rz * np.array([w.real, w.imag])
    b.append("\\draw[faint] %s -- %s;" % (N.pt(Z(-1.12)), N.pt(Z(1.15))))
    b.append("\\draw[faint] %s -- %s;" % (N.pt(Z(-1.12j)), N.pt(Z(1.15j))))
    b.append("\\draw[ink] %s circle (%.3f);" % (N.pt(c), Rz))
    for w in z:
        b.append("\\fill[ndGreen] %s circle (1.8pt);" % N.pt(Z(complex(w))))
    b.append("\\node[font=\\tiny, anchor=north east] at %s {$1$};" % N.pt(Z(1.0)))
    # 7. lambda-plane, Re stretched
    xa, xb = stages[6]
    o = np.array([xb - 0.45, 0.95])
    L = lambda w: o + np.array([1.85 * w.real, 0.27 * w.imag])
    b.append("\\fill[ndSky, fill opacity=0.45] %s rectangle %s;" % (N.pt(L(-0.75 - 3.6j)), N.pt(L(0 + 3.6j))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (N.pt(L(-0.8 + 0j)), N.pt(L(0.25 + 0j))))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s;" % (N.pt(L(-3.8j)), N.pt(L(3.9j))))
    for w in POLES:
        b.append("\\fill[ndInk] %s circle (1.6pt);" % N.pt(L(w)))
    for w in lam:
        b.append("\\draw[line width=0.7pt, ndGreen] %s circle (3.2pt);" % N.pt(L(w)))
    b.append("\\node[note, anchor=north] at (7.7,-1.2) {all seven panels computed from $%d$ exact samples of $\\langle\\sigma_n\\rangle$;"
             " in 7, rings: recovered poles; dots: true poles};" % K)
    N.write("ch05_era", b)
    print("written figures/ch05_era.tex; recovered poles", np.round(np.sort_complex(lam), 6))


if __name__ == "__main__":
    main()
