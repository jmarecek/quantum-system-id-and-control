#!/bin/sh
# build.sh -- shell equivalent of the Makefile targets, for machines where
# `make` is unavailable (e.g. a broken xcrun shim on macOS).
#
#   ./build.sh                 handouts + slides
#   ./build.sh handouts|slides|book
#   ./build.sh handout 05      lecture notes of chapter 05 alone
#   ./build.sh slides 05       slides of chapter 05 alone
#   ./build.sh figures         regenerate all Python-exported TikZ figures
#   ./build.sh pages           handout page count per chapter (budget <= 15; <= 10 without figures/code)
#   ./build.sh clean           remove build artefacts (keeps PDFs)
set -e
cd "$(dirname "$0")"
LATEXMK="latexmk -pdf -interaction=nonstopmode -halt-on-error"

chapter_file() { ls ch$1_*.tex 2>/dev/null | grep -v _slides.tex | head -1; }
# labels_handouts.tex / labels_slides.tex let single-lecture builds resolve cross-chapter references
labels() {
  [ -f handoutsbeamer.aux ] && grep '^\\newlabel' handoutsbeamer.aux > labels_handouts.tex
  [ -f overheadsbeamer.aux ] && grep '^\\newlabel' overheadsbeamer.aux > labels_slides.tex
  return 0
}

case "${1:-all}" in
  all)      $LATEXMK handoutsbeamer.tex; $LATEXMK overheadsbeamer.tex; labels ;;
  handouts) $LATEXMK handoutsbeamer.tex; labels ;;
  slides)
    if [ -n "$2" ]; then
      f=$(chapter_file "$2"); [ -n "$f" ] || { echo "no chapter $2"; exit 1; }
      s=${f%.tex}_slides.tex
      sed 's/^\\documentclass\[handoutsbeamer.tex\]{subfiles}/\\documentclass[overheadsbeamer.tex]{subfiles}/' "$f" > "$s"
      $LATEXMK "$s"
    else
      $LATEXMK overheadsbeamer.tex; labels
    fi ;;
  book)     $LATEXMK book.tex ;;
  labels)   labels ;;
  handout)
    f=$(chapter_file "$2"); [ -n "$f" ] || { echo "no chapter $2"; exit 1; }
    $LATEXMK "$f" ;;
  figures)
    for s in code/ch??_*.py; do echo "python3 $s"; python3 "$s"; done ;;
  pages)
    for f in ch??_*.tex; do
      case $f in *_slides.tex) continue;; esac
      p=${f%.tex}.pdf
      if [ -f "$p" ]; then
        n=$(pdfinfo "$p" 2>/dev/null | awk '/^Pages:/{print $2}')
        flag=""; [ "$n" -gt 15 ] && flag="  <-- OVER 15"
        [ "$n" -gt 10 ] && [ -z "$flag" ] && flag="  (>10: needs figures/code)"
        printf "%-45s %3s pages%s\n" "$f" "$n" "$flag"
      else printf "%-45s  (not built)\n" "$f"; fi
    done ;;
  clean)
    rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.bbl *.blg *.idx *.ilg *.ind *.lof *.lot *.loa *.fls *.fdb_latexmk *.synctex.gz ;;
  *) echo "unknown target $1"; exit 1 ;;
esac
