import pandas as pd


def load_institution_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_course_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def coerce_unitid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["UNITID"] = pd.to_numeric(df["UNITID"], errors="coerce")
    df = df.dropna(subset=["UNITID"])
    df["UNITID"] = df["UNITID"].astype(int)
    return df


def aggregate_course_features(course_df: pd.DataFrame) -> pd.DataFrame:
    course_df = coerce_unitid(course_df)

    agg_cols = [
        "DEBT_ALL_STGP_ANY_MDN",
        "DEBT_ALL_PP_ANY_MDN",
    ]
    for col in agg_cols:
        if col in course_df.columns:
            course_df[col] = pd.to_numeric(course_df[col], errors="coerce")

    base_cols = ["UNITID", "CIPCODE", "CREDLEV"]
    subset_cols = base_cols + [c for c in agg_cols if c in course_df.columns]
    subset = course_df[subset_cols].copy()

    agg_dict: dict = {
        "CIPCODE": "nunique",
        "CREDLEV": "nunique",
    }
    if "DEBT_ALL_STGP_ANY_MDN" in subset.columns:
        agg_dict["DEBT_ALL_STGP_ANY_MDN"] = ["mean", "min", "max"]
    if "DEBT_ALL_PP_ANY_MDN" in subset.columns:
        agg_dict["DEBT_ALL_PP_ANY_MDN"] = ["mean", "min", "max"]

    grouped = subset.groupby("UNITID").agg(agg_dict)
    grouped.columns = ["_".join([str(c) for c in col if c]) if isinstance(col, tuple) else col for col in grouped.columns]
    grouped = grouped.rename(columns={
        "CIPCODE_nunique": "ENG_PROG_COUNT",
        "CREDLEV_nunique": "ENG_CRED_LEVELS",
    })
    return grouped.reset_index()


def merge_institution_course(inst_df: pd.DataFrame, course_agg: pd.DataFrame) -> pd.DataFrame:
    inst_df = coerce_unitid(inst_df)
    return inst_df.merge(course_agg, on="UNITID", how="left")
