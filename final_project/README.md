# Smart College Predictor — Structured Project

This `final_project/` folder restructures the work into a clean, GitHub-ready layout with data, notebooks, and reusable code modules.

## Layout
```
final_project/
  data/                       # place input CSVs here (see Data)
  notebooks/
    01_data_preprocessing.ipynb
    02_feature_engineering.ipynb
    03_model_training.ipynb    # institution + course features, stable scaling
    04_evaluation_visualization.ipynb (same as 03 for now; use for reporting)
  src/
    preprocessing.py           # load/clean/merge institution + course data
    feature_engineering.py     # VALUE_LABEL creation + feature selection
    modeling.py                # model training, ranking helper
    utils.py                   # normalization + admission buckets
  README.md
  requirements.txt
  .gitignore
```

## Data
Expected files (do not commit if sensitive):
- `data/institute_most_recent_processed.csv` — institution-level features (includes COSTT4_A, QUALITY_SCORE, ADM_RATE, etc.).
- `data/merged_course_level.csv` — course-level engineering data (for program counts and debt aggregates).
If you keep the CSVs at repo root, update notebook paths accordingly.

## Notebooks
- `03_model_training.ipynb` is a cleaned copy of the stable training notebook (institution + course). It trains multiple classifiers (LogReg, RF, GB, KNN, SVC) with cross-validated ROC AUC selection, stable min/max scaling for ranking, sector/location/debt/program-count scoring, and admission buckets.
- `04_evaluation_visualization.ipynb` currently mirrors `03`—use it for plots/analysis or extend as needed.
- `01` and `02` are placeholders to document your preprocessing/feature steps if you want to separate them.

## Scripts (src/)
- `preprocessing.py` — helpers to load data, coerce UNITID, aggregate course-level features, and merge with institution data.
- `feature_engineering.py` — defines base and course feature lists and adds the VALUE_LABEL (affordable + high quality).
- `modeling.py` — builds/trains classifiers and regressors via GridSearchCV, selects by CV score, and provides a `rank_colleges` helper that applies user weights with stable normalization and admission buckets.
- `utils.py` — normalization helpers (`minmax_with_stats`, `compute_norm_stats`) and admission bucket logic.

## Requirements
See `requirements.txt` for minimal dependencies (pandas, numpy, scikit-learn, matplotlib, seaborn).

## How to run
1. Ensure data CSVs are in `data/` or adjust notebook paths.
2. Install deps: `pip install -r requirements.txt`.
3. Open `notebooks/03_model_training.ipynb` and run cells. Model choice is based on cross-validated ROC AUC; test set is used only for reporting.
4. Use the ranking examples to generate `USER_SCORE` tables with user-defined weights. Admission buckets are included when `ADM_RATE` is available.
