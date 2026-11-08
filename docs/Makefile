.PHONY: pdf clean

pdf: ## build main.pdf
	latexmk -pdf -interaction=nonstopmode main.tex

clean: ## remove aux files, keep the PDF
	latexmk -c main.tex
