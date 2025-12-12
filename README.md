# CPDS-ECH6845-Group3
This repository hosts notebooks for the final group project for the course Chemical Process Data Science, instructed by Sumant Shreedhar Patankar in Fall 2025 for Group 3.

Group members are: Prathamesh Khandakar, Sirsha Ganguly, Antara Das and Abdulfatai Faro.

## 📊 Data Processing

### 1. Download the Raw Data
First download the full dataset from the **College Scorecard** website:

👉 https://collegescorecard.ed.gov/data/

Save the raw files into the `data/raw/` directory  
(or any location you prefer — just update the paths inside the notebooks).

---

### 2. Run the Preprocessing Notebooks
After downloading the dataset, run the preprocessing notebooks in order:

1. **`institute_level_data_processing.ipynb`** — performs initial cleaning and feature engineering for institute level data.  
2. **`course_level_data_processing.ipynb`** — performs initial cleaning for field of study data.

📌 Both notebooks are located in the [`notebooks/data_processing/`](./notebooks/data_processing/).

---

### 3. Output - processed data
The cleaned and merged dataset will be saved to: [`data/processed/`](./data/processed)

## 🧠 Modeling
Full model builds (classification/regression) and visual diagnostics that feed the final app/ranker.

📌 notebooks are located in the [`notebooks/modeling/`](./notebooks/modeling/).

### 1. [`ML_value_&_quality_level_classification_&_regression.ipynb`](./notebooks/modeling/ML_value_%26_quality_level_classification_%26_regression.ipynb)
Value-label classifiers with institution + course aggregates, plus regression on `QUALITY_SCORE` with and without the GRAD_RATE/RET_FT4/MD_EARN_WNE_P10 ingredients to remove leakage so new schools can be scored; notes on parallel earnings regression for student-facing salary predictions.

### 2. [`ML_institute_only_earnings_slider_no_leak_course_features_added.ipynb`](./notebooks/modeling/ML_institute_only_earnings_slider_no_leak_course_features_added.ipynb)
No-leak target predictions with institute + course feature sets for VALUE_LABEL classification (drops the label ingredients) and MD_EARN_WNE_P10 regression, with interactive sliders to re-rank colleges by earnings vs cost/quality/location/sector preferences.

### 3. [`PCA and others.ipynb`](./notebooks/modeling/PCA%20and%20others.ipynb)
Exploratory PCA, mutual information, and UMAP on the VALUE_LABEL using leak-free features to see which variables separate affordable/high-quality schools.

### 4. [`Visualising correlation in the data.ipynb`](./notebooks/modeling/Visualising%20correlation%20in%20the%20data.ipynb)
Correlation heatmaps and target-wise correlations (QUALITY_SCORE, COSTT4_A, MD_EARN_WNE_P10) to spot groups of features that move together.

## 🔍 Exploration
Drafts, baselines, and feature/location/ROI experiments that informed the final modeling choices.

📌 notebooks are located in the [`notebooks/exploration/`](./notebooks/exploration/).

### 1. [`ML_models_institute_level_only.ipynb`](./notebooks/exploration/ML_models_institute_level_only.ipynb)
Initial leak-free VALUE_LABEL classifiers and regressors (dropping COSTT4_A/QUALITY_SCORE from features), plus a user-weighted ranking that mixes value probability, cost, engineering intensity, earnings, and location/sector preferences.

### 2. [`ML_models_institute_with_field_&_location_filter.ipynb`](./notebooks/exploration/ML_models_institute_with_field_%26_location_filter.ipynb)
Adds course-level aggregates and introduces a location preference weight/radius so rankings can respect nearby schools while comparing classifiers.

### 3. [`ML_models_institute_with_field_location_filter_with_regression.ipynb`](./notebooks/exploration/ML_models_institute_with_field_location_filter_with_regression.ipynb)
Extends the prior notebook with regression: QUALITY_SCORE prediction shows leakage when GRAD_RATE/RET_FT4/MD_EARN_WNE_P10 stay in, then re-runs without them to generalize to new schools; also sketches earnings regression for student guidance.

### 4. [`ML_models_stable_ranking.ipynb`](./notebooks/exploration/ML_models_stable_ranking.ipynb)
Design notes and code for merging course aggregates, adding distance-based filters, and plotting CV grids to keep rankings stable while honoring user location weights.

### 5. [`ML_institute_only_earnings_slider.ipynb`](./notebooks/exploration/ML_institute_only_earnings_slider.ipynb)
Institute-only VALUE_LABEL classifiers (with quality/grad features still present) plus MD_EARN_WNE_P10 regression and ranking sliders to trade off cost, quality, location, and public/private preferences.

### 6. [`ML_institute_only_earnings_slider_no_leak.ipynb`](./notebooks/exploration/ML_institute_only_earnings_slider_no_leak.ipynb)
Same slider workflow but removes GRAD_RATE, RET_FT4, QUALITY_SCORE, and earnings from the classifier features to avoid target leakage; shows the expected drop in ROC AUC/recall.

### 7. [`ML_institute_only_earnings-prediction.ipynb`](./notebooks/exploration/ML_institute_only_earnings-prediction.ipynb)
Baseline institute-only experiment where VALUE_LABEL classification still leaks (keeps GRAD_RATE/RET_FT4/QUALITY_SCORE) while MD_EARN_WNE_P10 regression and ranking functions are prototyped.

### 8. [`ML model.ipynb`](./notebooks/exploration/ML%20model.ipynb)
Early “high-value engineering college” classifier and admission-rate model that hit perfect scores because the label-defining features stayed in X; points to later notebooks for leak-free setups.

### 9. [`College Choice.ipynb`](./notebooks/exploration/College%20Choice.ipynb)
Quick MinMax scoring (no ML) that normalizes affordability, access, and outcome metrics to rank institutes with simple user weights.

### 10. [`location_filter.ipynb`](./notebooks/exploration/location_filter.ipynb) / [`location_filter.backup.ipynb`](./notebooks/exploration/location_filter.backup.ipynb)
Haversine helper that filters schools within a user-specified radius of a latitude/longitude before scoring.

### 11. [`ROI.ipynb`](./notebooks/exploration/ROI.ipynb) / [`ROI (1).ipynb`](./notebooks/exploration/ROI%20(1).ipynb)
Return-on-investment and belonging calculations comparing public vs private schools (e.g., average ROI and diversity-adjusted belonging gaps).
