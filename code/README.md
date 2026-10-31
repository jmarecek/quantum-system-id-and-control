# code/

Python scripts that generate the figures of the lecture notes as TikZ/pgfplots,
and short simulations shown as code snippets in the notes.

Conventions
* `chNN_<name>.py` writes `../figures/chNN_<name>.tex` (a single `tikzpicture`)
  through the helpers in `tikzexport.py`; the chapter includes it with
  `\input{figures/chNN_<name>.tex}`.
* Only `numpy`/`scipy` are needed (Python >= 3.7). Run from the course root:
  `python3 code/ch02_bloch.py`, or `make figures` to regenerate all.
* Snippets are shown in the notes with `\lstinputlisting[style=python,firstline=..,lastline=..]{code/chNN_<name>.py}`
  inside a `\begin{frame}[fragile]`.
