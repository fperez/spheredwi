# Simple makefile to build a pdf file out of tex sources
.PHONY: clean

# Title of final pdf
title = whitepaper

# Specify all dependencies that can change for this pdf
$(title).pdf: $(title).tex $(title).bbl

# Rule declarations, don't change this unless you know what you're doing
%.aux : %.tex
	pdflatex $<

%.bbl : %.aux
	bibtex $<

%.pdf : %.tex %.bbl
	pdflatex $<
	pdflatex $<

clean:
	-rm -f *.aux *.log *.out *.bbl *.blg *~

clean-all: clean
	-rm -f $(title).pdf
