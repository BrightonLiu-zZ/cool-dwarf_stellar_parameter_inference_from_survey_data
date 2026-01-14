
## 1) Project title

**ASTR199: Cool-dwarf stellar parameter inference from survey data (photometry-first baseline)**

## 2) One-sentence purpose

Build a reproducible, end-to-end data pipeline and baseline ML models to infer **effective temperature (Teff)** for K and M dwarfs using **survey photometry and cross-matched catalogs**, starting with a version-1 dataset built primarily from **LAMOST + SDSS** (and commonly Gaia / 2MASS / WISE via crossmatch).

## 3) Scope and priorities

### Version 1 (current scope)

1. **Target first:** Teff (regression)
2. **Data core:** LAMOST objects + SDSS where applicable; add multi-band photometry through crossmatch (Gaia is already present in my extracted table; I still need 2MASS/WISE and any relevant LAMOST/SDSS photometric bands if available).
3. **Features first:** color indices (e.g., Gaia colors; then extend to 2MASS/WISE colors once added), plus basic quality flags if available.
4. **Deliverable focus:** working dataset builder + baseline model + evaluation + documented schema.

### Later iterations (after Teff baseline is working)

* Add **log g** and **[Fe/H] / metallicity** as additional targets (either multi-output regression or separate models).
* Improve crossmatch completeness; add more surveys or additional feature engineering as needed.

## Data sources (working assumptions)

* **LAMOST catalog**: primary object list and (potentially) parameters/labels depending on what fields are available in the release used.
* **SDSS**: used for version-1 overlap and/or supplemental metadata (depending on availability).
* **Gaia photometry**: currently available in my extracted table (Gaia magnitudes).
* **2MASS / WISE photometry**: expected to be added via crossmatch (currently missing from my extracted table).
* Note: I will not assume any specific column names unless confirmed by the dataset schema I print from my files.

## Labels / targets (to be finalized with advisor guidance)

### Primary target (Phase 1)

* **Teff** (effective temperature)
* Source of truth: to be confirmed (e.g., LAMOST pipeline parameter, or a curated label set in a reference catalog/paper sample).

### Secondary targets (Phase 2)

* **log g** (surface gravity)
* **[Fe/H]** (metallicity proxy)

## Features (version-1 baseline)

* **Photometric colors** derived from available magnitudes:

  * Start: Gaia-only colors (since Gaia magnitudes are present)
  * Extend: Gaia+2MASS, Gaia+WISE once crossmatch is in place
* Optional: simple quality filters/flags when available (e.g., photometric uncertainty thresholds)

## Pipeline components (definition of “end-to-end”)

1. **Ingest**: read base catalogs / tables
2. **Select sample**: apply basic cuts (object class / quality)
3. **Crossmatch / enrich**: join external photometry (2MASS/WISE and/or SDSS metadata)
4. **Feature build**: compute colors; handle missing data; standardize features
5. **Split strategy**: train/val/test (rule to be documented; avoid leakage)
6. **Train baseline model**: simple regressors first (interpretable baseline)
7. **Evaluate**: metrics + residual diagnostics + sanity checks
8. **Package outputs**: save dataset (parquet/csv), trained model artifact, and a run log

## Success criteria (Version 1)

A run is “successful” when:

* I can rebuild the dataset from raw inputs reproducibly.
* The final dataset has a clear **data dictionary** (column meaning + units).
* Teff model trains end-to-end and produces evaluation metrics on a held-out set.
* I can explain (briefly) which colors/features drive performance (e.g., feature importance or permutation importance).

---

## Introduction to source papers

1) A neural network approach to determining photometric metallicities of M-type dwarf stars.pdf

Role: A methods reference for predicting metallicity from photometry using a neural network.
How you’ll use it: Later-phase guidance for the [Fe/H]/[M/H] target (model design, feature choices, evaluation mindset). Not required for your first Teff-only baseline, but good to keep as a reference.

2) Exploring the Age-dependent Properties of M and L Dwarfs Using Gaia and SDSS.pdf

Role: A data/selection + feature reference for Gaia × SDSS cool dwarfs, including quality cuts and relationships between Gaia colors/absolute magnitudes and spectral type.
How you’ll use it: Helps justify color features, provides sanity-check expectations (what colors correlate with type), and informs sensible Gaia quality filtering.

3) Half a Million M Dwarf Stars Characterized Using Domain-adapted Spectral Analysis.pdf

Role: A core pipeline/labels reference for deriving Teff, log g, metallicity for a very large LAMOST M dwarf sample (LAMOST DR10) using a robust spectral-analysis approach.
How you’ll use it: This is one of your most important “ground truth / label source” and dataset-definition references (even if you only start with Teff).

4) Machine learning methods for the search for L&T brown dwarfs in the data of modern sky surveys.pdf

Role: A ML classification reference for identifying L/T dwarfs using multi-survey photometric colors (e.g., 2MASS/WISE + optical surveys) and comparing classical color cuts vs ML.
How you’ll use it: Feature engineering ideas (color combinations), model baselines (RF/XGBoost/SVM), and how to communicate performance vs simple decision rules.

5) Theissen et al 2017.pdf

Role: A catalog + crossmatch/photometry reference (LaTE-MoVeRS) built from SDSS + 2MASS + WISE with proper motions, photometric distances, and Teff estimates for late-type objects.
How you’ll use it: Very relevant for crossmatch thinking (SDSS/2MASS/WISE combination), practical photometric Teff estimation context, and “what fields/surveys matter” for cool dwarfs.