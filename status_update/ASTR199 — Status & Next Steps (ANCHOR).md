
## Last updated

2026-01-13 (update this date each time you edit)

## Current phase

* Assembled the training set (N=41,362) by cross-matching LAMOST M-dwarfs with the Wang et al. (2022) VAC.

* The dataset includes photometry from Pan-STARRS, SDSS, 2MASS, and WISE, along with cleaned temperature labels.

## What is done

* Downloaded and inspected dr8_final_vac_flag_parameter.fits and LAMOST_dM0-7.csv.

* Performed left join on obsid (LAMOST Observation ID).

* Removed sentinel values (-9999).

* Applied quality flags (uqflag=1) to ensure unique source.

* Enforced strict completeness: dropped rows missing any critical magnitude (Pan-STARRS/2MASS/WISE/SDSS).

## What is in progress

* Identifying already cross-matched catalogs (LAMOST × Gaia × IR surveys) to reduce custom cross-matching overhead.

## Immediate next actions

* Move from data collection to feature construction (colors, absolute magnitudes) and begin baseline regression/anomaly-detection models.