# Contributing to the Quantum Optimal Control course materials

See [README.md](README.md) for the syllabus and the layout of the repository.

## Writing a chapter

Every chapter file starts with

```latex
%\documentclass[overheadsbeamer.tex]{subfiles}
\documentclass[handoutsbeamer.tex]{subfiles}
\begin{document}
\ona{ \setcounter{chapter}{N-1} }
\lecture{Title}{LectNN}
\chapter{Title}\label{C.Label}
```

and consists of `\begin{frame}\ft{Title} ... \end{frame}` blocks separated by
`\section{}`/`\subsection{}`:

* `\ona{...}` — appears **only in the handouts** (prose paragraphs, proofs, figure environments);
* `\onp{...}` — appears **only on the slides** (bullet lists, resized figures);
* everything else (equations, `defn`/`thm`/`propn`/`lem`/`cor`/`exm`/`exe`/`rmk`/`probl` environments) appears in both.

Frames containing code listings need `\begin{frame}[fragile]`.

Everything a chapter shows must sit inside a frame.  The full slide deck uses beamer's
`ignorenonframetext`, which skips text *and assignments* outside frames; a chapter's
closing `\biblio` is handled by the drivers (empty in the full builds, a reference list
or a References frame in single-lecture builds).

There is no fixed page budget; `make pages` reports the handout page count of each
chapter.

## Figures and code

Figures are never pasted as PDFs (the only exceptions are the team portraits and the hardware photographs `figures/team_*.jpg`, `figures/hw_*` taken over from the quantum computing course slides, with their original credits).  Each figure is produced by
`code/chNN_<name>.py`, which uses the helpers in `code/tikzexport.py`
(`pgfplots_axis`, `pgfplots_matrix`, `bloch_sphere`, `write_tikz`) or, for the
geometric figures, `code/needham.py` to write `figures/chNN_<name>.tex`; the chapter includes it with
`\input{figures/chNN_<name>.tex}`.  Snippets of the same scripts are shown with
`\lstinputlisting[style=python,firstline=..,lastline=..]{code/chNN_<name>.py}`.
Only `numpy`/`scipy` are needed.  `make figures` regenerates everything.

The geometric figures share one visual style, modelled on T. Needham, *Visual
Differential Geometry and Forms* (only the style is borrowed, no figure is
reproduced): ink line art, pale colour washes, bold labelled vectors, and short
italic notes with thin pointer arrows.  `code/needham.py` provides the colours,
the TikZ styles (`ink`, `faint`, `vec`, `note`, `pointer`, `dot`, `wash`) and an
orthographic `View` for spheres; write figures with `needham.write(name, body)`.
Each script should compute or assert the geometry it draws.

## Building

```sh
make               # handouts + slides (latexmk)
./build.sh         # same targets when make is unavailable: ./build.sh handout 03
make handout-03    # notes for lecture 3 only
make slides-03     # slides for lecture 3 only
make labels        # refresh labels_*.tex from the last full build (done automatically by make handouts/slides)
make pages         # page budget check
```

Requires TeX Live (tested with 2026: beamer, beamerarticle, subfiles, pgfplots,
algorithm2e, natbib, ...) and Python 3.7+ with numpy and scipy.

## Single-lecture builds

`make handout-NN` / `make slides-NN` compile one chapter on its own.  Cross-chapter
references to chapter numbers always resolve (fallbacks in `coursemacros.tex`);
references to theorems, equations and figures of other chapters resolve to the
numbers of the last full build through `labels_handouts.tex` / `labels_slides.tex`
(printed without hyperlinks).  Run `make handouts` once after cloning to create them.

## Checks

* `code/test_standalone.sh` compiles every chapter on its own, as handout and as slides, and
  reports exit codes, page counts and unresolved references (this is what "rendering the
  handouts for one lecture only" must pass without errors).
* `make pages` / `./build.sh pages` prints the handout page count of each chapter (the
  standalone PDFs include the chapter's own reference list, which the full build does not).

## Bibliography

`refs.bib` merges the bibliographies of all source papers.  The reference lists that
existed only as `thebibliography` items (the Ansel et al. tutorial, Control_Technical2)
were resolved to proper entries through Crossref / DataCite by `code/resolve_bib.py`;
items it could not match with confidence remain as `@misc` entries whose `title`
holds the original text (`python3 code/resolve_bib.py` then `python3 code/merge_resolved_bib.py`
to rerun the resolution).  The optional textbooks are `dalessandro2021introduction` and `borzi2017formulation`.  The six source papers of the course are `bondar2025globally`,
`lawrence2026geometric`, `parvaiz2025identifiability`, `lawrence2026penalised`,
`parvaiz2026survey`, and the tutorial `ansel2024introduction`.

## Sources

The chapters draw on the following material (kept in sibling folders, not part
of the build):

| folder | content | reuse |
| --- | --- | --- |
| `SystemID_Perspective/` | survey of identifiability and identification of open quantum systems | verbatim (own work) |
| `SystemID_Technical/` | identifiability of autonomous and controlled open quantum systems | verbatim (own work) |
| `Control_Technical/` | globally optimal control via polynomial optimization (QCPOP) | verbatim (own work) |
| `Control_Technical2/` | geometric quantum control and the random Schrödinger equation | verbatim (own work) |
| `Control_Technical3/` | penalised and constrained geodesics in geometric control | verbatim (own work) |
| `Control_Tutorial/` | Ansel et al., tutorial on quantum optimal control | **rephrased only**, no figures |
| `conic_optimization/` | the conic optimization course | verbatim (own work), lecture 1 |
| `Quantum_Computing_via_Randomized_Algorithms/` | the quantum computing course | LaTeX set-up; `ch01_QM101.tex` adapted in Lecture 2, Section 1 (own work) |
