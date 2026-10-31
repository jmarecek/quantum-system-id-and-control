"""ch01_tomography_scaling.py -- the cost of full state tomography with the number
of qubits: real parameters of a density matrix (4^n - 1), Pauli strings to measure,
and a rough count of shots for a fixed precision.  Writes figures/ch01_tomography_scaling.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

n = np.arange(1, 13)
params = 4.0 ** n - 1
eps = 0.01
shots = params / eps ** 2          # order of magnitude: each of the 4^n-1 expectation values to precision eps
BUDGET = 1e9                       # about a day at 10 kHz


def main():
    first_over = int(n[np.argmax(shots > BUDGET)])
    assert first_over == 9 and abs(BUDGET / (1e4 * 86400) - 1) < 0.2
    b = []
    Pl = N.Plot((0.5, 12.5), (0.0, 12.0), 10.0, 5.6)
    b += Pl.axes("number of qubits $n$", "", xticks=[(k, "$%d$" % k) for k in range(1, 13)],
                 yticks=[(k, "$10^{%d}$" % k) for k in range(0, 13, 3)])
    # the region beyond the budget
    b.append("\\fill[ndRose!55] %s rectangle %s;" % (Pl.pt(0.5, 9.0), Pl.pt(12.5, 12.0)))
    b.append("\\draw[ndRed, line width=0.8pt, dash pattern=on 4pt off 2pt] %s -- %s;" % (Pl.pt(0.5, 9.0), Pl.pt(12.5, 9.0)))
    b.append("\\node[note, anchor=south west, text=ndRed, align=left] at %s {beyond $10^{9}$ shots, about a day\\\\ of measurement at $10\\,$kHz};"
             % Pl.pt(0.75, 10.4))
    b.append(Pl.curve(n, np.log10(params), "line width=1.0pt, ndBlue"))
    b.append(Pl.curve(n, np.log10(shots), "line width=1.0pt, ndRed"))
    b += ["\\fill[ndBlue] %s circle (1.7pt);" % Pl.pt(k, np.log10(p)) for k, p in zip(n, params)]
    b += ["\\fill[ndRed] %s rectangle %s;" % (N.pt(Pl.P(k, np.log10(s)) - 0.055), N.pt(Pl.P(k, np.log10(s)) + 0.055))
          for k, s in zip(n, shots)]
    b.append("\\node[note, anchor=west, text=ndBlue, align=left] at %s {parameters\\\\ of $\\rho$: $4^n-1$};"
             % Pl.pt(12.55, np.log10(params[-1])))
    b.append("\\node[note, anchor=north west, text=ndRed, align=left] at %s {shots,\\\\ $(4^n-1)/\\epsilon^2$\\\\"
             " for $\\epsilon=10^{-2}$};" % Pl.pt(12.55, 12.2))
    b.append("\\node[note, anchor=south, text=ndRed] (o) at %s {$n=%d$: over budget};" % (Pl.pt(first_over + 1.4, 7.0), first_over))
    b.append("\\draw[pointer] (o.west) to[bend left=20] %s;" % Pl.pt(first_over + 0.12, np.log10(shots[first_over - 1]) - 0.1))
    N.write("ch01_tomography_scaling", b)
    print("written figures/ch01_tomography_scaling.tex; first n over 1e9 shots: %d" % first_over)


if __name__ == "__main__":
    main()
