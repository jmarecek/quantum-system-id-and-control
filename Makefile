# Quantum Optimal Control -- build system
#
#   make               handouts + slides
#   make handouts      handoutsbeamer.pdf   (lecture notes, amsbook + beamerarticle)
#   make slides        overheadsbeamer.pdf  (beamer slides)
#   make book          book.pdf             (Springer svmono rendering; optional)
#   make handout-05    lecture notes for chapter 05 alone (ch05_*.tex compiled standalone)
#   make slides-05     slides for chapter 05 alone
#   make figures       regenerate all Python-exported TikZ figures (code/chNN_*.py -> figures/chNN_*.tex)
#   make pages         print the handout page count of every chapter (budget: <= 15, <= 10 without figures/code)
#   make clean         remove build artefacts (keeps PDFs)
#   make distclean     also remove PDFs and generated single-lecture drivers

LATEXMK  = latexmk -pdf -interaction=nonstopmode -halt-on-error
PYTHON   = python3
CHAPTERS = $(sort $(wildcard ch??_*.tex))
SCRIPTS  = $(sort $(wildcard code/ch??_*.py))
FIGS     = $(patsubst code/%.py,figures/%.tex,$(SCRIPTS))

.PHONY: all handouts slides book figures pages clean distclean

all: handouts slides

handouts: handoutsbeamer.pdf
slides:   overheadsbeamer.pdf
book:     book.pdf

handoutsbeamer.pdf overheadsbeamer.pdf book.pdf: %.pdf: %.tex $(CHAPTERS) $(FIGS) refs.bib coursemacros.tex $(wildcard macros_ch??.tex)
	$(LATEXMK) $<
	@$(MAKE) --no-print-directory labels

# labels_handouts.tex / labels_slides.tex let single-lecture builds resolve
# cross-chapter references (theorems, figures) to the numbers of the full build.
.PHONY: labels
labels:
	@[ -f handoutsbeamer.aux ] && grep '^\\newlabel' handoutsbeamer.aux > labels_handouts.tex && echo "labels_handouts.tex updated" || true
	@[ -f overheadsbeamer.aux ] && grep '^\\newlabel' overheadsbeamer.aux > labels_slides.tex && echo "labels_slides.tex updated" || true

# ---- single lectures ---------------------------------------------------
# make handout-05 / make slides-05 pick the unique ch05_*.tex
handout-%:
	@f=$$(ls ch$*_*.tex | grep -v _slides.tex | head -1); test -n "$$f" || { echo "no chapter $*"; exit 1; }; \
	echo "== $$f (handout)"; $(LATEXMK) $$f

slides-%:
	@f=$$(ls ch$*_*.tex | grep -v _slides.tex | head -1); test -n "$$f" || { echo "no chapter $*"; exit 1; }; \
	s=$${f%.tex}_slides.tex; \
	sed 's/^\\documentclass\[handoutsbeamer.tex\]{subfiles}/\\documentclass[overheadsbeamer.tex]{subfiles}/' $$f > $$s; \
	echo "== $$s (slides)"; $(LATEXMK) $$s

# ---- figures -----------------------------------------------------------
figures: $(FIGS)

figures/%.tex: code/%.py code/tikzexport.py
	$(PYTHON) $<

# ---- page budget -------------------------------------------------------
pages:
	@for f in $(CHAPTERS); do \
	  case $$f in *_slides.tex) continue;; esac; \
	  p=$${f%.tex}.pdf; \
	  if [ -f $$p ]; then n=$$(pdfinfo $$p 2>/dev/null | awk '/^Pages:/{print $$2}'); \
	    flag=""; [ "$$n" -gt 15 ] && flag="  <-- OVER 15"; [ "$$n" -gt 10 ] && [ -z "$$flag" ] && flag="  (>10: needs figures/code)"; \
	    printf "%-45s %3s pages%s\n" $$f $$n "$$flag"; \
	  else printf "%-45s  (not built: make handout-NN)\n" $$f; fi; \
	done

# ---- cleaning ----------------------------------------------------------
clean:
	latexmk -c -silent $(wildcard *.tex) 2>/dev/null || true
	rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.bbl *.blg *.idx *.ilg *.ind *.lof *.lot *.loa *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f *.pdf ch??_*_slides.tex
