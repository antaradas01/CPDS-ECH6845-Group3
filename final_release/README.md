# Smart College Predictor (Final Release)

This folder contains cleaned, GitHub-ready notebooks for training and serving the "high-value college" models and generating user-weighted rankings.

## Contents
- `01_value_model_institution.ipynb` — Institution-level model only. Trains multiple classifiers (LogReg, RF, GB, KNN, SVC) with cross-validated ROC AUC selection, optional regressors for `QUALITY_SCORE`, and a ranking function with user weights + admission buckets.
- `02_value_model_institution_course_stable.ipynb` — Institution + course-level aggregates (program counts, debt stats) with stable min/max scaling for ranking. Uses cross-validated selection, location radius, sector prefs, debt/program-count scoring, and admission buckets.
- `03_location_filter.ipynb` — Simple haversine-based location radius filter example.

## Data inputs
Place these CSVs alongside the notebooks (already in the repo root):
- `institute_most_recent_processed.csv` (institution-level features, includes `ADM_RATE`, `QUALITY_SCORE`, etc.)
- `merged_course_level.csv` (course-level engineering data used by notebook 02)

## Running
1) Open the notebook in Jupyter/VS Code. 
2) Run cells sequentially. Each model uses `GridSearchCV` with 5-fold CV; expect a few minutes depending on hardware.
3) After training, use the example weight dictionaries to generate a ranked table. The output includes `USER_SCORE` and an `ADM_BUCKET` column: `High admit (>=70%)`, `Medium admit (30-70%)`, `Low admit (<30%)` (or `Unknown` if missing).

## User weights (ranking)
You can set weights 0–5 for:
- `value_model` (model probability of being high-value)
- `cost` (lower cost better)
- `engineering` (ENG_RATIO)
- `earnings` (MD_EARN_WNE_P10)
- `research` (QUALITY_SCORE or proxy)
- `location` (state match or radius score)
- `program_count` (only notebook 02)
- `debt` (only notebook 02; lower debt better)
- `prefer_public` / `prefer_private` (sector preference)

Notebook 02 also applies stable min/max scaling from the training set (`norm_stats`) so scores stay consistent across user queries.

## Notes
- Test sets are used only for final reporting; model choice is based on cross-validated scores.
- Missing values are imputed inside pipelines; rows are not dropped before scoring.
- Location radius uses haversine distance (km).
