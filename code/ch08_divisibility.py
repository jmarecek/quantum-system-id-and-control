"""ch08_divisibility.py -- trace distance under Markovian vs non-Markovian dephasing.

Pure dephasing multiplies the off-diagonal element of rho by c(t), a CPTP map
for every t as long as |c(t)| <= 1.  For the pair |+>, |-> the trace distance is
D(t) = |c(t)|.  Markovian: c = exp(-gamma t) (monotone).  Toy non-Markovian
model: c = exp(-gamma t) cos(omega t) (revivals: information flows back).
Output: figures/ch08_divisibility.tex
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T


def dephase(rho, c):
    out = rho.copy(); out[0, 1] *= c; out[1, 0] *= np.conj(c)
    return out


def trace_distance(rho, sigma):
    return 0.5 * np.abs(np.linalg.eigvalsh(rho - sigma)).sum()


if __name__ == "__main__":
    plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    minus = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)
    t = np.linspace(0, 6, 400)
    gamma, omega = 0.4, 2.0
    c_m = np.exp(-gamma * t)
    c_nm = np.exp(-gamma * t) * np.cos(omega * t)
    D_m = [trace_distance(dephase(plus, c), dephase(minus, c)) for c in c_m]
    D_nm = [trace_distance(dephase(plus, c), dephase(minus, c)) for c in c_nm]
    body = T.pgfplots_axis([
        dict(x=t, y=D_m, label="Markovian: $c(t)=\\ee^{-\\gamma t}$"),
        dict(x=t, y=D_nm, label="non-Markovian: $c(t)=\\ee^{-\\gamma t}\\cos\\omega t$"),
    ], xlabel="$t$", ylabel="$D(\\mathcal{E}_t(\\rho_+),\\mathcal{E}_t(\\rho_-))$",
        width="0.8\\linewidth", height="0.42\\linewidth")
    T.write_tikz("ch08_divisibility", body)
    print("wrote figures/ch08_divisibility.tex")
