# bayes_market_external

Public-facing materials for the `bayes_market` project, prepared for course submission and sharing without proprietary WRDS raw data.

## What is included

- `slides/wrds_slides_submission.tex`
- `slides/wrds_slides_submission.pdf`
- WRDS figures used in the slide deck under `assets/figures/wrds/`
- Amazon-demo figure used in the slide deck under `assets/figures/demo/`
- Coverage table used in the appendix under `assets/tables/wrds/`
- Supporting code used for the presentation workflow:
  - `code/notebooks/05_wrds_analysis.ipynb`
  - `code/src/wrds/generate_summary_tables.py`
  - `code/src/demo/bayes_brand.py`
  - `code/src/demo/plot_cr4_posteriors.py`
  - `code/src/demo/prep_amazon.py`

## What is excluded

- Proprietary WRDS raw data
- Proprietary WRDS processed extracts
- Any local credentials, API tokens, or account-specific files

This repository is intentionally presentation-oriented. The slide deck and exported visuals are included directly, but the underlying WRDS data files needed to rerun the full analysis are not distributed here.

## Directory structure

- `slides/`: submission-ready Beamer slides
- `assets/figures/`: figures referenced by the slides
- `assets/tables/`: LaTeX table inputs referenced by the slides
- `code/`: supporting notebook and scripts

## Notes on reproducibility

The included code shows the analysis workflow, but the WRDS-specific portions cannot be executed without licensed WRDS access and the omitted proprietary extracts. The Amazon demo code is included because it is referenced in the appendix as a non-WRDS illustration of the same Bayesian allocation logic.

## Compile the slides

From this repository root:

```bash
cd slides
pdflatex wrds_slides_submission.tex
```
