# Paper

LaTeX sources for *Load by Design: Adaptive Heuristics for LLM Checkpoint Loading*.

## Build

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Structure

| File | Content |
|------|---------|
| `main.tex` | Document root |
| `template.tex` | Style selection (arXiv/MLSys) |
| `library.bib` | Bibliography |
| `sections/` | Individual paper sections |
| `figures/` | Diagrams and plots |
| `PRIMEarxiv.sty` | arXiv style package |

## arXiv Submission

- `00README.json` — arXiv packaging metadata
- `arxiv_abstract_plain.txt` — Plain-text abstract for submission form
- Build produces `arxiv.pdf` for upload
