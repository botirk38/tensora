# AGENTS.md — Paper

## Scope

LaTeX sources for the academic paper. Do NOT modify without explicit instruction.

## Build

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Conventions

- Style: PRIMEarxiv (arXiv preprint format)
- Bibliography: `library.bib`
- Sections in `sections/` directory
- Figures in `figures/` directory

## Do NOT

- Edit paper content without user instruction
- Change the document class or style
- Remove or rename section files
