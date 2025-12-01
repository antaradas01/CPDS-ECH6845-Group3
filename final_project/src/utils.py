import numpy as np
import pandas as pd


def minmax_with_stats(series: pd.Series, stats: tuple[float, float]) -> pd.Series:
    mn, mx = stats
    denom = mx - mn
    return (series - mn) / (denom if denom > 0 else 1e-9)


def admission_bucket(adm_rate: float) -> str:
    if pd.isna(adm_rate):
        return "Unknown"
    if adm_rate >= 0.70:
        return "High admit (>=70%)"
    if adm_rate >= 0.30:
        return "Medium admit (30-70%)"
    return "Low admit (<30%)"


def compute_norm_stats(df: pd.DataFrame, cols: list[str]) -> dict:
    stats = {}
    for col in cols:
        if col in df.columns:
            mn = df[col].min()
            mx = df[col].max()
            if mx == mn:
                mx = mn + 1e-9
            stats[col] = (mn, mx)
    return stats
