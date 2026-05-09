# Paper build

Default build uses arXiv style (PRIMEarxiv). To switch template:

1. Edit `template.tex`
2. Comment/uncomment the relevant `\XXtrue` / `\XXfalse` pair
3. Ensure the matching `.cls` file is available
4. Rebuild: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
