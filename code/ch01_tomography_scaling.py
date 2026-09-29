"""ch01_tomography_scaling.py -- the cost of full state tomography with the number
of qubits: real parameters of a density matrix (4^n - 1), Pauli strings to measure,
and a rough count of shots for a fixed precision.  Writes figures/ch01_tomography_scaling.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

n = np.arange(1, 13)
params = 4.0 ** n - 1
eps = 0.01
shots = params / eps ** 2          # order of magnitude: each of the 4^n-1 expectation values to precision eps
series = [dict(x=n, y=params, label=r"parameters $4^n-1$", style="mark=*"),
          dict(x=n, y=shots, label=r"shots for precision $\epsilon=10^{-2}$ (order of magnitude)", style="mark=square*"),
          dict(x=n, y=np.full_like(n, 1e9, dtype=float), label=r"$10^{9}$ shots ($\approx$ a day at $10\,$kHz)", style="gray, dashed")]
body = T.pgfplots_axis(series, xlabel="number of qubits $n$", ylabel="", width="0.85\\linewidth", height="0.45\\linewidth",
                       legend_pos="north west", axis_options="ymode=log, xtick=data")
T.write_tikz("ch01_tomography_scaling", body)
print("written figures/ch01_tomography_scaling.tex")
