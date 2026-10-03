"""ch03_bath.py -- a system and its bath.

A qubit (left) coupled to a bath of harmonic oscillators.  The bath modes have
frequencies omega_k on a grid, and couplings g_k with g_k^2 = J(omega_k) d omega
for the Ohmic spectral density J(omega) = eta omega exp(-omega / omega_c):
the line widths show g_k.  The reduced state rho_S = Tr_B rho_full keeps only
what is inside the dashed window.  The inset shows J and the sampled modes.
Writes figures/ch03_bath.tex.
"""
import numpy as np
import needham as N

ETA, WC = 0.3, 1.5


def J(w):
    return ETA * w * np.exp(-w / WC)


def main():
    w = np.linspace(0.25, 6.0, 14)
    dw = w[1] - w[0]
    g = np.sqrt(J(w) * dw)
    wf = np.linspace(w[0] - dw / 2, w[-1] + dw / 2, 20001)    # the band covered by the modes
    assert abs((g ** 2).sum() - np.trapz(J(wf), wf)) / np.trapz(J(wf), wf) < 0.02
    b = []
    q = np.array([0.0, 0.0])
    # the bath: a pale region with the modes on a curved grid
    b.append("\\filldraw[fill=ndPaper, draw=ndInk!40, line width=0.6pt, rounded corners=18pt] (3.0,-2.6) rectangle (10.2,2.6);")
    rng = np.random.default_rng(2)
    pos = [np.array([3.9 + 0.95 * (k % 7), 1.5 - 1.5 * (k // 7) + 0.18 * rng.normal()]) for k in range(len(w))]
    for p, gk, wk in zip(pos, g, w):
        b.append("\\draw[line width=%.2fpt, ndBlue!60, opacity=0.8] %s -- %s;" % (0.3 + 4.5 * gk, N.pt(q + (0.6, 0)), N.pt(p - (0.25, 0))))
    for p, gk, wk in zip(pos, g, w):
        s = np.linspace(-0.22, 0.22, 40)                       # a little spring: frequency ~ wavelength
        b.append(N.polyline([p + (x, 0.11 * np.sin(2 * np.pi * (0.6 + wk / 3) * x / 0.22)) for x in s], "line width=0.8pt, ndInk"))
        b.append("\\fill[ndGold] %s circle (2.2pt);" % N.pt(p + (0.26, 0)))
    b.append("\\shade[ball color=ndSky!80!white] %s circle (0.55);" % N.pt(q))
    b.append("\\draw[ink] %s circle (0.55);" % N.pt(q))
    b.append("\\node at %s {$S$};" % N.pt(q))
    b.append("\\draw[line width=0.9pt, ndRed, dash pattern=on 4pt off 2pt, rounded corners=6pt] (-1.0,-1.0) rectangle (1.0,1.0);")
    b.append("\\node[note, anchor=north, text=ndRed] at (0,-1.1) {$\\rhoS=\\TrB\\,\\rho_{\\rm full}$:\\\\ we see only\\\\ what is in here};")
    b.append("\\node[note, anchor=south] at (6.6,2.65) {bath $B$: many oscillators $\\omega_k$, coupled with strengths $g_k$ (line widths)};")
    b.append("\\node[note, anchor=south] at (0,1.05) {system $S$};")
    # inset: the spectral density and the sampled modes
    Pl = N.Plot((0, 6.5), (0, 0.42), 4.2, 1.5, origin=(4.6, -2.35))
    b += Pl.axes("$\\omega$", "$J(\\omega)$")
    for wk in w:
        b.append("\\draw[line width=0.8pt, ndGold] %s -- %s;" % (Pl.pt(wk, 0), Pl.pt(wk, J(wk))))
    wf2 = np.linspace(0, 6.5, 200)
    b.append(Pl.curve(wf2, J(wf2), "line width=1.2pt, ndBlue"))
    b.append("\\node[note, anchor=west] at %s {Ohmic: $g_k^2=J(\\omega_k)\\,\\Delta\\omega$};" % Pl.pt(3.2, 0.33))
    N.write("ch03_bath", b)
    print("written figures/ch03_bath.tex")


if __name__ == "__main__":
    main()
