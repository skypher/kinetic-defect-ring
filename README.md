# A single kinetic defect in a run-and-tumble ring

This repository contains a theorem-centered analysis of a single lattice
run-and-tumble particle on an `n`-site ring.  The hopping rate is one, the
bulk tumbling rate is `omega`, and the tumbling rate at site zero is
`omega_0`.

The compiled manuscript is [`paper/main.pdf`](paper/main.pdf), with source in
[`paper/main.tex`](paper/main.tex). Its principal results are:

1. three equivalent exact characteristic equations (resolvent,
   Chebyshev, and `2 x 2` transfer-matrix forms);
2. a parity factorization and a Jordan-block classification, including the
   flat-band point `omega = 1`;
3. an infinite-volume criterion for defect-localized modes, including
   the compactly supported mode at the singular transfer point;
4. fixed-rate and critical large-ring expansions of the spectral gap; and
5. a term-by-term comparison with the telegraph and
   telegraph-with-lattice-diffusion approximations.

The characteristic, transfer, Jordan, gap, and compact-localization
formulas can be replayed against direct matrices with
[`scripts/verify_formulas.py`](scripts/verify_formulas.py).  The script is a
short audit, not a substitute for the proofs in the manuscript.

## Build the manuscript

A TeX installation providing `pdflatex`, BibTeX, AMS-LaTeX, `mathtools`,
`geometry`, `hyperref`, and Latin Modern is required.

```sh
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Replay the formula audit

The audit requires Python 3 and NumPy.

```sh
python3 -u scripts/verify_formulas.py
```
