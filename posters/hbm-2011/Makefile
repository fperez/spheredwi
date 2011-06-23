.PHONY: all clean clenall

all:
	latex poster.tex && bibtex poster.aux && latex poster.tex && latex poster.tex  && echo "\n\nTurning to PDF...\n\n" && dvipdf poster.dvi
	git update-index --assume-unchanged poster.pdf


clean:
	rm -f *~ *bak poster.aux poster.bbl poster.blg poster.log poster.dvi

cleanall: clean
	rm -f poster.pdf