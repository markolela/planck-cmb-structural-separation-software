# planck-cmb-structural-separation-software

Software for the paper:

**Robust Structural Separation of Planck 2018 CMB Temperature Patches from Phase-Randomized Power-Preserving Surrogates**

## What this repository contains

This repository contains the Python analysis pipeline and helper scripts used for the CMB1 structural-separation study.
It provides the software used to build patch datasets, run the comparison pipeline, and generate numerical summaries for the paper.
Large artifact bundles are not stored in this repository.

## Scope

This repository is the software companion to the paper:

Robust Structural Separation of Planck 2018 CMB Temperature Patches from Phase-Randomized Power-Preserving Surrogates

Its role is to provide the analysis code.
The released artifact bundles and numerical outputs are linked separately from the software repository.

## Software DOI

Concept DOI:
https://doi.org/10.5281/zenodo.18972934

Version DOI (v1.0.0):
https://doi.org/10.5281/zenodo.18972935

## Reproducibility

The paper defines dataset IDs and an artifact layout under:

`data/processed/astro/suite/<dataset_id>/`

Run outputs include JSON summaries and CSV metrics.
The numerical values reported in the paper are traced through these generated artifacts, as described in the paper appendix.

## Main entry points

- `scripts/run_t3_on_patches.py`  
  Runs the T3 pipeline on an existing patch-stack dataset directory and produces summaries and figures.

- `scripts/build_headline_patches.py`  
  Builds the headline patch stacks used for the main Planck analyses.

- `scripts/build_hm_diff_patches.py`  
  Builds the half-mission difference-map patch stacks used for the predefined negative control.

## Installation

Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

## License

Apache-2.0. See `LICENSE`.

## Citation

See `CITATION.cff`.

Concept DOI:
https://doi.org/10.5281/zenodo.18972934

Version DOI (v1.0.0):
https://doi.org/10.5281/zenodo.18972935
