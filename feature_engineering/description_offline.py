import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------
# Step 3: B.* description embedding distance offline
# -------------------------------------------------------


def handle_description(df, legit_mask_np):
    """
    Add description-based features to the dataframe.

    Computes TF-IDF embeddings of package descriptions and calculates
    distance to the centroid of legitimate package descriptions.

    :param df: DataFrame containing 'latest_description' column
    :type df: pd.DataFrame
    :param legit_mask_np: Boolean numpy array indicating legitimate packages
    :type legit_mask_np: np.ndarray
    :return: DataFrame with added 'dist_embed_to_legit_desc' column
    :rtype: pd.DataFrame
    """
    # Extract descriptions, handling nulls
    desc_text = df["latest_description"].fillna("").astype(str)

    # Create TF-IDF vectorizer for word-level n-grams (1-2 words)
    desc_vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english"
    )
    X_desc = desc_vectorizer.fit_transform(desc_text)

    # Compute centroid of legitimate package descriptions
    # If no legitimate packages exist, use overall mean as fallback
    if legit_mask_np.sum() > 0:
        legit_desc_centroid = X_desc[legit_mask_np].mean(axis=0)
    else:
        legit_desc_centroid = X_desc.mean(axis=0)

    legit_desc_centroid = np.asarray(legit_desc_centroid)

    # Compute cosine similarity and convert to distance (1 - similarity)
    sim_to_legit_desc = cosine_similarity(X_desc, legit_desc_centroid).ravel()
    df["dist_embed_to_legit_desc"] = 1.0 - sim_to_legit_desc

    return df
