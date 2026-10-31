"""ch13_twoqubit_id.py -- identification of a two-qubit GKSL generator from shot-noisy data.

Qualitative re-creation of the numerical demonstration in the identifiability paper:
 * simulate the coherence vector x(t) = (Tr F_j rho(t))_j of the two-qubit model
   (Hamiltonian omega_1, omega_2, delta; local dephasing / decay / pump rates),
 * sample Pauli expectation values with S shots per time point,
 * recover the discrete propagator by least squares, the generator by a matrix log,
 * fit the 3 Hamiltonian parameters and 6 rates on their known support,
 * plot the relative errors against S (expected S^{-1/2}).
Writes figures/ch13_twoqubit_err.tex and figures/ch13_twoqubit_bars.tex.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import logm
import tikzexport as T

I2, X, Y, Z = T.pauli()
sp, sm = 0.5 * (X + 1j * Y), 0.5 * (X - 1j * Y)
paulis = [I2, X, Y, Z]
# orthonormal basis F_j of traceless Hermitian 4x4 matrices: Pauli strings / 2
F = [np.kron(a, b) / 2 for a in paulis for b in paulis][1:]
n = len(F)                                                     # 15

TRUE = dict(w1=1.0, w2=1.3, delta=0.2, gz1=0.05, gz2=0.03, gm1=0.08, gm2=0.06, gp1=0.02, gp2=0.01)
NAMES = ["w1", "w2", "delta", "gz1", "gz2", "gm1", "gm2", "gp1", "gp2"]


def op(A, k):
    return np.kron(A, I2) if k == 0 else np.kron(I2, A)


def lindblad(p):
    """Return the map rho -> L(rho) for parameter dict p."""
    H = 0.5 * p["w1"] * op(Z, 0) + 0.5 * p["w2"] * op(Z, 1) + p["delta"] * (op(sp, 0) @ op(sm, 1) + op(sm, 0) @ op(sp, 1))
    jumps = []
    for k in range(2):
        jumps += [(0.5 * p["gz%d" % (k + 1)], op(Z, k)), (2 * p["gm%d" % (k + 1)], op(sm, k)), (2 * p["gp%d" % (k + 1)], op(sp, k))]

    def L(rho):
        out = -1j * (H @ rho - rho @ H)
        for g, J in jumps:
            out = out + g * (J @ rho @ J.conj().T - 0.5 * (J.conj().T @ J @ rho + rho @ J.conj().T @ J))
        return out
    return L


def affine_form(L):
    """Coherence-vector form  x' = A x + b  of a Lindbladian L."""
    b = np.array([np.real(np.trace(Fj @ L(np.eye(4) / 4))) for Fj in F])
    A = np.array([[np.real(np.trace(Fj @ L(Fk))) for Fk in F] for Fj in F])
    return A, b


def structure_matrix():
    """Linear map from the 9 physical parameters to vec([A | b])."""
    cols = []
    for name in NAMES:
        p = dict((k, 0.0) for k in NAMES); p[name] = 1.0
        A, b = affine_form(lindblad(p))
        cols.append(np.concatenate([A.ravel(), b]))
    return np.array(cols).T


def identify(S, tau=0.3, K=20, rng=None):
    A, b = affine_form(lindblad(TRUE))
    M = np.zeros((n + 1, n + 1)); M[:n, :n] = A; M[:n, n] = b
    Md = np.real(np.array(_expm(M * tau)))
    rows_in, rows_out = [], []
    for j in range(n):                                          # 15 initial states Id/4 + F_j/2
        x = np.zeros(n); x[j] = 0.5
        for k in range(K):
            xa = np.append(x, 1.0)
            xn = Md @ xa
            # shot noise on the Pauli expectations m_j = 2 x_j, variance (1-m^2)/S
            noise = rng.randn(n) * np.sqrt(np.maximum(1 - (2 * xn[:n]) ** 2, 0) / S) / 2
            rows_in.append(xa); rows_out.append(xn[:n] + noise)
            x = xn[:n]
    Xin, Xout = np.array(rows_in), np.array(rows_out)
    Md_hat = np.linalg.lstsq(Xin, Xout, rcond=None)[0].T        # 15 x 16
    Mfull = np.vstack([Md_hat, np.append(np.zeros(n), 1.0)])
    M_hat = np.real(logm(Mfull)) / tau
    A_hat, b_hat = M_hat[:n, :n], M_hat[:n, n]
    theta_hat = np.linalg.lstsq(structure_matrix(), np.concatenate([A_hat.ravel(), b_hat]), rcond=None)[0]
    return theta_hat


def _expm(M):
    from scipy.linalg import expm
    return expm(M)


def main():
    rng = np.random.RandomState(0)
    truth = np.array([TRUE[k] for k in NAMES])
    shots = np.logspace(2, 6, 9)
    errH, errD = [], []
    for S in shots:
        eh, ed = [], []
        for rep in range(5):
            th = identify(S, rng=rng)
            eh.append(np.linalg.norm(th[:3] - truth[:3]) / np.linalg.norm(truth[:3]))
            ed.append(np.linalg.norm(th[3:] - truth[3:]) / np.linalg.norm(truth[3:]))
        errH.append(np.mean(eh)); errD.append(np.mean(ed))
        print("S=%.0e  eps_theta=%.3e  eps_gamma=%.3e" % (S, errH[-1], errD[-1]))
    guide = errD[0] * np.sqrt(shots[0] / shots)
    T.write_tikz("ch13_twoqubit_err", T.pgfplots_axis(
        [dict(x=shots, y=errH, label="$\\varepsilon_\\theta$ (Hamiltonian)", style="mark=*"),
         dict(x=shots, y=errD, label="$\\varepsilon_\\gamma$ (rates)", style="mark=square*"),
         dict(x=shots, y=guide, label="$S^{-1/2}$ guide", style="dashed, gray")],
        xlabel="shots per time point $S$", ylabel="relative error", legend_pos="south west",
        axis_options="xmode=log, ymode=log"))
    th = identify(1e4, rng=rng)
    labels = ["$\\omega_1$", "$\\omega_2$", "$\\delta$", "$g^z_1$", "$g^z_2$", "$g^-_1$", "$g^-_2$", "$g^+_1$", "$g^+_2$"]
    bars = ["\\begin{axis}[ybar, bar width=5pt, width=0.85\\linewidth, height=0.45\\linewidth, ylabel={value},",
            "  symbolic x coords={%s}, xtick=data, x tick label style={font=\\footnotesize}," % ",".join(labels),
            "  legend style={font=\\footnotesize}, legend pos=north east, ymin=0, enlarge x limits=0.08]",
            "\\addplot[fill=qocA, draw=none] coordinates {%s};" % " ".join("(%s,%.4f)" % (l, v) for l, v in zip(labels, truth)),
            "\\addplot[fill=qocB, draw=none] coordinates {%s};" % " ".join("(%s,%.4f)" % (l, v) for l, v in zip(labels, th)),
            "\\legend{exact, recovered ($S=10^4$)}", "\\end{axis}"]
    T.write_tikz("ch13_twoqubit_bars", "\n".join(bars))


if __name__ == "__main__":
    main()
