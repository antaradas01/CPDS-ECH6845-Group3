from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    r2_score,
)
from utils import minmax_with_stats, admission_bucket


def build_classifiers():
    classifiers = {}

    log_reg_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    log_reg_param_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__class_weight": [None, "balanced"]
    }
    classifiers["LogisticRegression"] = (log_reg_pipe, log_reg_param_grid)

    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    rf_param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__class_weight": [None, "balanced"],
    }
    classifiers["RandomForest"] = (rf_pipe, rf_param_grid)

    gb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])
    gb_param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.01, 0.1],
        "clf__max_depth": [3, 5],
    }
    classifiers["GradientBoosting"] = (gb_pipe, gb_param_grid)

    knn_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
    knn_param_grid = {
        "clf__n_neighbors": [5, 15, 25],
        "clf__weights": ["uniform", "distance"],
    }
    classifiers["KNN"] = (knn_pipe, knn_param_grid)

    svc_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ])
    svc_param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", "auto"],
        "clf__kernel": ["rbf", "linear"],
        "clf__class_weight": [None, "balanced"],
    }
    classifiers["SVC"] = (svc_pipe, svc_param_grid)

    return classifiers


def train_classifiers(X_train, y_train, X_test, y_test):
    classifiers = build_classifiers()
    best_classifiers = {}
    cv_best_scores = {}
    test_metrics = {}

    for name, (pipe, param_grid) in classifiers.items():
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        cv_best_scores[name] = grid.best_score_
        best_model = grid.best_estimator_
        best_classifiers[name] = best_model

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        test_metrics[name] = {
            "roc_auc": auc,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion": confusion_matrix(y_test, y_pred).tolist(),
        }
    best_model_name = max(cv_best_scores, key=cv_best_scores.get)
    return best_model_name, best_classifiers[best_model_name], cv_best_scores, test_metrics


def build_regressors():
    regressors = {}

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestRegressor(random_state=42))
    ])
    rf_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    regressors["RandomForestRegressor"] = (rf, rf_grid)

    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingRegressor(random_state=42))
    ])
    gb_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.01, 0.1],
        "clf__max_depth": [3, 5],
    }
    regressors["GradientBoostingRegressor"] = (gb, gb_grid)

    en = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", ElasticNet(max_iter=5000))
    ])
    en_grid = {
        "clf__alpha": [0.01, 0.1, 1.0, 10.0],
        "clf__l1_ratio": [0.1, 0.5, 0.9],
    }
    regressors["ElasticNet"] = (en, en_grid)

    svr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVR())
    ])
    svr_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", "auto"],
        "clf__epsilon": [0.01, 0.1, 0.2],
    }
    regressors["SVR"] = (svr, svr_grid)

    return regressors


def train_regressors(X_train, y_train, X_test, y_test):
    regressors = build_regressors()
    best_regressors = {}
    cv_best_scores = {}
    test_metrics = {}

    for name, (pipe, param_grid) in regressors.items():
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="r2",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        cv_best_scores[name] = grid.best_score_
        best_model = grid.best_estimator_
        best_regressors[name] = best_model

        preds = best_model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        test_metrics[name] = {"r2": r2, "rmse": rmse}

    best_name = max(cv_best_scores, key=cv_best_scores.get)
    return best_name, best_regressors[best_name], cv_best_scores, test_metrics


def rank_colleges(df: pd.DataFrame, feature_cols: list[str], value_model, user_weights: dict, norm_stats: dict, user_lat=None, user_lon=None, radius_km=None, preferred_state=None) -> pd.DataFrame:
    df_scored = df.copy()

    # Model-based probability
    X_all = df_scored[feature_cols]
    p_value = value_model.predict_proba(X_all)[:, 1]
    df_scored["p_value_model"] = p_value

    def score_with_stats(col_name, out_col, invert=False):
        if col_name in df_scored.columns and col_name in norm_stats:
            vals = minmax_with_stats(df_scored[col_name].astype(float), norm_stats[col_name])
            df_scored[out_col] = 1 - vals if invert else vals
        else:
            df_scored[out_col] = 0.5

    # Components
    score_with_stats("COSTT4_A", "cost_score", invert=True)
    score_with_stats("ENG_RATIO", "eng_score")
    score_with_stats("MD_EARN_WNE_P10", "earnings_score")
    score_with_stats("QUALITY_SCORE", "research_score")
    score_with_stats("ENG_PROG_COUNT", "prog_count_score")

    debt_cols_present = [c for c in df_scored.columns if "DEBT_ALL_STGP_ANY_MDN" in c or "DEBT_ALL_PP_ANY_MDN" in c]
    if debt_cols_present:
        debt_raw = df_scored[debt_cols_present].mean(axis=1)
        if "DEBT_AGG_MEAN" in norm_stats:
            vals = minmax_with_stats(debt_raw, norm_stats["DEBT_AGG_MEAN"])
            df_scored["debt_score"] = 1 - vals
        else:
            df_scored["debt_score"] = 1 - minmax_with_stats(debt_raw, (debt_raw.min(), debt_raw.max()))
    else:
        df_scored["debt_score"] = 0.5

    if preferred_state is not None and "STABBR" in df_scored.columns:
        df_scored["location_score"] = (df_scored["STABBR"] == preferred_state).astype(float)
    else:
        df_scored["location_score"] = 0.5

    if all(c in df_scored.columns for c in ["IS_PUBLIC", "IS_PRIVATE", "IS_FORPROFIT"]):
        sector_indicator = np.where(
            df_scored["IS_PUBLIC"] == 1, 1.0,
            np.where(df_scored["IS_PRIVATE"] == 1, 0.0, 0.5)
        )
    else:
        sector_indicator = np.full(len(df_scored), 0.5)
    df_scored["sector_indicator"] = sector_indicator

    # Weights
    w_value   = user_weights.get("value_model", 0)
    w_cost    = user_weights.get("cost", 0)
    w_eng     = user_weights.get("engineering", 0)
    w_earn    = user_weights.get("earnings", 0)
    w_loc     = user_weights.get("location", 0)
    w_research = user_weights.get("research", 0)
    w_prog    = user_weights.get("program_count", 0)
    w_debt    = user_weights.get("debt", 0)
    w_public  = user_weights.get("prefer_public", 0)
    w_private = user_weights.get("prefer_private", 0)

    sector_component = (
        w_public  * df_scored["sector_indicator"] +
        w_private * (1 - df_scored["sector_indicator"])
    )

    total_w = (
        w_value + w_cost + w_eng + w_earn + w_loc +
        w_research + w_prog + w_debt + w_public + w_private
    )
    if total_w == 0:
        raise ValueError("All user weights are zero!")

    df_scored["USER_SCORE"] = (
        w_value * df_scored["p_value_model"] +
        w_cost  * df_scored["cost_score"] +
        w_eng   * df_scored["eng_score"] +
        w_earn  * df_scored["earnings_score"] +
        w_loc   * df_scored["location_score"] +
        w_research * df_scored["research_score"] +
        w_prog * df_scored["prog_count_score"] +
        w_debt * df_scored["debt_score"] +
        sector_component
    ) / total_w

    ranked = df_scored.sort_values("USER_SCORE", ascending=False)
    if "ADM_RATE" in df_scored.columns:
        ranked["ADM_BUCKET"] = df_scored["ADM_RATE"].apply(admission_bucket)
    return ranked
