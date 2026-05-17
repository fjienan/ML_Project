.PHONY: all clean pdf view

LATEX := xelatex
MAIN  := main
OUT   := $(MAIN).pdf

all: $(OUT)

$(OUT): $(MAIN).tex sections/*.tex
	@mkdir -p _build
	$(LATEX) -output-directory=_build $(MAIN)
	@cp _build/$(OUT) .

clean:
	rm -rf _build *.pdf *.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk

view: $(OUT)
	xdg-open $(OUT)
