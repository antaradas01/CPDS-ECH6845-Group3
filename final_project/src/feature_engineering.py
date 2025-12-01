import pandas as pd


BASE_FEATURES = [
    "ENG_RATIO", "STEM_RATIO",
    "ENG_HEAVY", "STEM_HEAVY",
    "UGDS", "UGDS_MEN", "UGDS_WOMEN",
    "DIVERSITY_SCORE",
    "TUITIONFEE_IN", "TUITIONFEE_OUT", "TUITION_GAP",
    "RET_FT4", "GRAD_RATE", "ADM_RATE",
    "MD_EARN_WNE_P10",
    "CONTROL", "IS_PUBLIC", "IS_PRIVATE", "IS_FORPROFIT",
]

COURSE_FEATURES = [
    "ENG_PROG_COUNT", "ENG_CRED_LEVELS",
    "DEBT_ALL_STGP_ANY_MDN_mean",
    "DEBT_ALL_STGP_ANY_MDN_min",
    "DEBT_ALL_STGP_ANY_MDN_max",
    "DEBT_ALL_PP_ANY_MDN_mean",
    "DEBT_ALL_PP_ANY_MDN_min",
    "DEBT_ALL_PP_ANY_MDN_max",
]


def add_value_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cost_med = df["COSTT4_A"].median()
    quality_med = df["QUALITY_SCORE"].median()
    df["VALUE_LABEL"] = ((df["COSTT4_A"] <= cost_med) & (df["QUALITY_SCORE"] >= quality_med)).astype(int)
    return df


def select_features(df: pd.DataFrame, include_course: bool = False) -> list[str]:
    feats = BASE_FEATURES.copy()
    if include_course:
        feats += COURSE_FEATURES
    return [c for c in feats if c in df.columns]
