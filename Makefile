TEX=paper.latex
PDF=$(TEX:.latex=.pdf)

.PHONY: build clean view

build:
	@echo "Building $(TEX) -> $(PDF) using latexmk"
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode -file-line-error" -use-make $(TEX)

clean:
	@echo "Cleaning auxiliary files for $(TEX)"
	latexmk -C $(TEX)

view: build
	@echo "Opening $(PDF)"
	if command -v xdg-open >/dev/null 2>&1; then xdg-open $(PDF) || true; fi
