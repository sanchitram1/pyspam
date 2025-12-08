import numpy as np
import pandas as pd


def create_legit_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Create a boolean mask identifying legitimate (non-spam) packages.

    :param df: DataFrame containing package data with optional 'is_spam' column
    :type df: pd.DataFrame
    :return: Boolean numpy array where True indicates legitimate packages
    :rtype: np.ndarray
    """
    if "is_spam" in df.columns:
        df["is_spam"] = pd.to_numeric(df["is_spam"], errors="coerce")
        legit_mask = df["is_spam"] == 0
    else:
        legit_mask = pd.Series([True] * len(df))

    return legit_mask.to_numpy()
