#!/bin/sh
# Compile every chapter on its own (handout and slides) with latexmk and report
# exit codes, page counts and unresolved references.  Run from the course root.
cd "$(dirname "$0")/.."
LATEXMK="latexmk -pdf -interaction=nonstopmode -halt-on-error"
status=0
for f in ch??_*.tex; do
  case $f in *_slides.tex) continue;; esac
  b=${f%.tex}
  $LATEXMK "$f" >/dev/null 2>&1; h=$?
  hq=$(pdftotext "$b.pdf" - 2>/dev/null | grep -c "??")
  hp=$(pdfinfo "$b.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
  sed 's/^\\documentclass\[handoutsbeamer.tex\]{subfiles}/\\documentclass[overheadsbeamer.tex]{subfiles}/' "$f" > "${b}_slides.tex"
  $LATEXMK "${b}_slides.tex" >/dev/null 2>&1; s=$?
  sq=$(pdftotext "${b}_slides.pdf" - 2>/dev/null | grep -c "??")
  sp=$(pdfinfo "${b}_slides.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
  uc=$(grep -c "Citation.*undefined" "$b.log" 2>/dev/null)
  printf "%-42s handout exit %s (%3s pages, %s ??, %s undefined cites)   slides exit %s (%3s slides, %s ??)\n" "$f" "$h" "$hp" "$hq" "$uc" "$s" "$sp" "$sq"
  [ "$h" -ne 0 ] && { grep -n "^!" -A3 "$b.log" | head -8; status=1; }
  [ "$s" -ne 0 ] && { grep -n "^!" -A3 "${b}_slides.log" | head -8; status=1; }
done
exit $status
