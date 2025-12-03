
import pandas as pd
import numpy as np
from settings import TOP_BRAND_PKGS, TOP_LEGIT_PACKAGES, BRAND_ALIASES
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from helper import min_levenshtein_to_set


def add_name_based(df: pd.DataFrame, legit_mask_np: pd.Series):
    name_text = df["pkg_name"].astype(str).fillna("")

    name_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
    X_name = name_vectorizer.fit_transform(name_text)

    if legit_mask_np.sum() > 0:
        legit_centroid = X_name[legit_mask_np].mean(axis=0)
    else:
        legit_centroid = X_name.mean(axis=0)

    legit_centroid = np.asarray(legit_centroid)

    sim_to_legit_name = cosine_similarity(X_name, legit_centroid).ravel()

    df["n_lev_dist_to_top1"] = [
        min_levenshtein_to_set(name, TOP_LEGIT_PACKAGES)
        for name in df["pkg_name"].astype(str)
    ]

    df["n_lev_dist_to_alias"] = [
        min_levenshtein_to_set(name, BRAND_ALIASES)
        for name in df["pkg_name"].astype(str)
    ]

    df["sim_tfidf_to_legit_centroid"] = sim_to_legit_name
    
    return df