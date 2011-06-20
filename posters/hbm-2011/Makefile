.PHONY: all

all:
	latex poster.tex && echo "\n\nTurning to PDF...\n\n" && dvipdf poster.dvi
	git update-index --assume-unchanged poster.pdf

