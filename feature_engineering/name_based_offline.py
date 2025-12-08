import logging

import numpy as np
import pandas as pd
from feature_engineering.helper import min_levenshtein_to_set
from feature_engineering.settings import BRAND_ALIASES, TOP_LEGIT_PACKAGES
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def add_name_based(df: pd.DataFrame, legit_mask_np: np.ndarray):
    """
    Add name-based features to the dataframe.

    Computes three features based on package names:
    1. n_lev_dist_to_top1: Minimum Levenshtein distance to top legitimate packages
    2. n_lev_dist_to_alias: Minimum Levenshtein distance to brand aliases
    3. sim_tfidf_to_legit_centroid: TF-IDF cosine similarity to legitimate package name centroid

    :param df: DataFrame containing package data with 'pkg_name' column
    :type df: pd.DataFrame
    :param legit_mask_np: Boolean numpy array indicating legitimate packages (True = legit, False = spam)
    :type legit_mask_np: np.ndarray
    :return: DataFrame with added name-based features
    :rtype: pd.DataFrame
    """
    logger.info("Starting name-based feature engineering")
    logger.debug(f"Processing {len(df)} packages, {legit_mask_np.sum()} legitimate")

    # Extract package names as strings, handling nulls
    name_text = df["pkg_name"].astype(str).fillna("")
    logger.debug(f"Extracted {len(name_text)} package names")

    # Create TF-IDF vectorizer for character n-grams (2-5 characters)
    # This captures typosquatting patterns and name similarities
    name_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
    X_name = name_vectorizer.fit_transform(name_text)
    logger.debug(f"Created TF-IDF matrix with shape {X_name.shape}")

    # Compute centroid of legitimate package names
    # If no legitimate packages exist, use overall mean as fallback
    if legit_mask_np.sum() > 0:
        legit_centroid = X_name[legit_mask_np].mean(axis=0)
        logger.debug(
            f"Computed centroid from {legit_mask_np.sum()} legitimate packages"
        )
    else:
        legit_centroid = X_name.mean(axis=0)
        logger.warning("No legitimate packages found, using overall mean as centroid")

    legit_centroid = np.asarray(legit_centroid)

    # Compute cosine similarity to legitimate centroid
    sim_to_legit_name = cosine_similarity(X_name, legit_centroid).ravel()
    logger.debug(
        f"Computed similarity scores (min={sim_to_legit_name.min():.3f}, max={sim_to_legit_name.max():.3f})"
    )

    # Compute Levenshtein distance to top legitimate packages
    logger.debug("Computing Levenshtein distances to top legitimate packages")
    df["n_lev_dist_to_top1"] = [
        min_levenshtein_to_set(name, TOP_LEGIT_PACKAGES)
        for name in df["pkg_name"].astype(str)
    ]

    # Compute Levenshtein distance to brand aliases
    logger.debug("Computing Levenshtein distances to brand aliases")
    df["n_lev_dist_to_alias"] = [
        min_levenshtein_to_set(name, BRAND_ALIASES)
        for name in df["pkg_name"].astype(str)
    ]

    # Add TF-IDF similarity feature
    df["sim_tfidf_to_legit_centroid"] = sim_to_legit_name

    logger.info("Completed name-based feature engineering")
    logger.debug(
        "Added features: n_lev_dist_to_top1, n_lev_dist_to_alias, sim_tfidf_to_legit_centroid"
    )

    return df
