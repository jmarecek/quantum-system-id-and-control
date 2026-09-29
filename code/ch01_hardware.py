"""ch01_hardware.py -- composite picture of a superconducting quantum computer:
cryostat photograph as background, the wiring stages seen inside the can, and two
nested zoom-ins: the printed circuit board at the centre of the cryostat and the
qubit chip at the centre of the board; room-temperature electronics to the right.
The photographs (figures/hw_*.{png,jpg}, credit IBM) come from the quantum
computing course slides.  Writes figures/ch01_hardware.tex.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

W = 8.0   # width of the cryostat photograph in cm (the picture is roughly square)
body = r"""
%% background: the cryostat
\node[anchor=south west, inner sep=0] (cryo) at (0,0) {\includegraphics[width=%(W).2fcm]{figures/hw_cryostat.png}};
%% wiring stages, seen inside the can (left-centre of the photograph)
\node[anchor=center, inner sep=0, draw=white, line width=0.6pt] (wires) at (%(wx).2f,%(wy).2f) {\includegraphics[height=%(wh).2fcm]{figures/hw_wires.jpg}};
%% zoom 1: the printed circuit board at the bottom of the cryostat
\node[anchor=center, inner sep=0, draw=qocB, line width=1pt] (pcb) at (%(px).2f,%(py).2f) {\includegraphics[width=%(pw).2fcm]{figures/hw_board.jpg}};
%% zoom 2: the qubit chip at the centre of the board
\node[anchor=center, inner sep=0, draw=qocA, line width=1pt] (chip) at (pcb.center) {\includegraphics[width=%(cw).2fcm]{figures/hw_chip.png}};
%% zoom lines
\draw[qocB, thick, dashed] (wires.south west) -- (pcb.north west);
\draw[qocB, thick, dashed] (wires.south east) -- (pcb.north east);
\draw[qocA, thick, dashed] ($(pcb.center)+(-0.12,-0.10)$) -- (chip.south west);
\draw[qocA, thick, dashed] ($(pcb.center)+(0.12,-0.10)$) -- (chip.south east);
%% room-temperature electronics, placed over the right half of the cabinet
\node[anchor=south, inner sep=0, draw=white, line width=0.6pt] (rack) at (%(rx).2f,%(ry).2f) {\includegraphics[height=%(rh).2fcm]{figures/hw_electronics.png}};
%% labels
\small
\node[anchor=north west, align=left, font=\small\bfseries, fill=white, fill opacity=0.8, text opacity=1, inner sep=1.5pt] at (0.05,%(top).2f) {dilution cryostat (Bluefors),\\ about 10\,mK};
\node[anchor=west, align=left, fill=white, fill opacity=0.8, text opacity=1, inner sep=1pt, font=\footnotesize] at ($(wires.east)+(0.06,0.5)$) {wiring and\\ filtering stages\\ (fragile below 1\,K)};
\node[anchor=north, align=center, fill=white, fill opacity=0.8, text opacity=1, inner sep=1pt, text=qocB] at ($(pcb.south)+(0,-0.05)$) {circuit board (zoom)};
\node[anchor=south, align=center, fill=white, fill opacity=0.8, text opacity=1, inner sep=1pt, text=qocA] at ($(chip.north)+(0,0.03)$) {qubit chip (zoom)};
\node[anchor=north, align=center, font=\small, text width=%(rw).2fcm, fill=white, fill opacity=0.85, text opacity=1, inner sep=2pt] at ($(rack.south)+(0,-0.08)$) {room-temperature electronics: signal sources, FPGA pulse shaping, digitisers -- \alert{this is where $u(t)$ is made}};
\node[anchor=north east, font=\tiny, fill=white, fill opacity=0.8, text opacity=1, inner sep=1pt] at (%(W).2f-0.05,%(top).2f) {Image credit: IBM.};
""" % dict(W=W, wx=0.30*W, wy=0.63*W, wh=0.34*W,
           px=0.34*W, py=0.26*W, pw=0.40*W, cw=0.17*W,
           rx=0.76*W, ry=0.30*W, rh=0.60*W, rw=0.36*W, top=W-0.05)
T.write_tikz("ch01_hardware", body, options="x=1cm,y=1cm")
print("written figures/ch01_hardware.tex")
