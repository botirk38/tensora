# Paper (LaTeX)

Sources for *Load by Design: Adaptive Heuristics for LLM Checkpoint Loading*: **`main.tex`** (standard `article`, `natbib` + `plainnat`). Sections live under `sections/`; bibliography is `bib/references.bib`.

**Commands to reproduce measurements** (Rust matrices, benchmarks, vLLM, environment capture) are in the **[repository root `README.md`](../README.md)**—that file is the paper-facing entry point for replication.

## Build

From this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or: `latexmk -pdf -interaction=nonstopmode main.tex` if you use `latexmk`.

## arXiv packaging

Create a `.zip` or `.tar.gz` whose **root** contains (no extra `paper/` wrapper):

- `main.tex`
- `00README.json` (declares `pdflatex` + `main.tex` as the top-level source for Submission System 1.5)
- `sections/` (all `.tex` fragments)
- `bib/references.bib`
- `figures/` if you add binary graphics later (currently figures are TikZ/pgfplots in source)

**Omit** generated files (`main.aux`, `main.log`, `main.pdf`, `main.out`, …). You may optionally include a freshly built `main.bbl` to pin bibliography output; otherwise arXiv runs BibTeX when `references.bib` is present.

**Plain abstract for the web form:** see `arxiv_abstract_plain.txt`.

### arXiv checklist

- **`bib`:** `plainnat.bst` is in TeX Live on arXiv; no custom `.bst` to ship.
- **Metadata:** Title and abstract in the web form should match the PDF; `\hypersetup` in `main.tex` only sets PDF viewer metadata.
- **License:** Choose in the submission UI.
- **Processor:** Select **pdfLaTeX** if the UI does not auto-detect (consistent with `00README.json`).
- **Preview:** Always open the auto-generated PDF and scan the bibliography and first pages before submitting.
