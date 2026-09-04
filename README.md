# A single kinetic defect in a run-and-tumble ring

This repository contains a theorem-centered analysis of a single lattice
run-and-tumble particle on an `n`-site ring.  The hopping rate is one, the
bulk tumbling rate is `omega`, and the tumbling rate at site zero is
`omega_0`.

The compiled manuscript is [`paper/main.pdf`](paper/main.pdf), with source in
[`paper/main.tex`](paper/main.tex).  The annotated tag
`paper-2026-09-04-referee-revised` identifies the exact manuscript and audit
revision described here.  Its principal results are:

1. three equivalent exact characteristic equations (resolvent,
   Chebyshev, and `2 x 2` transfer-matrix forms);
2. a parity factorization and a Jordan-block classification, including the
   flat-band point `omega = 1`;
3. an infinite-volume criterion for defect-localized modes outside the bulk
   spectrum, an exclusion of dispersive embedded eigenvalues, and exact
   descriptions of the compact singular mode and embedded flat-band eigenspace;
4. fixed-rate and critical large-ring expansions of the spectral gap; and
5. a term-by-term comparison with the telegraph and
   telegraph-with-lattice-diffusion approximations.

The characteristic, transfer, flat-band Jordan, fixed-rate and critical
gap, and localization formulas can be replayed against direct matrices
and their defining equations with
[`scripts/verify_formulas.py`](scripts/verify_formulas.py).  The script is a
short assertion-based audit, including all ring sizes from 2 through 12
for the flat-band nullity profiles and a multi-size convergence check for the
critical-gap side expansions.  The data in the manuscript's numerical
summary figure are reproduced by
[`scripts/generate_figure_data.py`](scripts/generate_figure_data.py).  These
computations are cross-checks rather than substitutes for the proofs in the
manuscript.

## Build the manuscript

A TeX installation providing `pdflatex`, BibTeX, AMS-LaTeX, `mathtools`,
`geometry`, `hyperref`, TikZ, PGFPlots, and Latin Modern is required.

```sh
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

## Replay the formula audit

The audit requires Python 3 and NumPy.  The reviewed formula audit was
tested with Python 3.12.3 and NumPy 1.26.4.  Install the pinned Python
dependency with

```sh
python3 -m pip install -r requirements.txt
```

```sh
python3 -u scripts/verify_formulas.py
```

## Regenerate the figure data

The figure-data script uses the same Python and NumPy environment as the
formula audit.

```sh
python3 -u scripts/generate_figure_data.py
```
