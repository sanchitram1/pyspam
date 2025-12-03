import numpy as np
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------
# Step 3: B.* description embedding distance offline
# -------------------------------------------------------


def handle_description(df, legit_mask_np):
    
    desc_text = df["latest_description"].fillna("").astype(str)

    desc_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X_desc = desc_vectorizer.fit_transform(desc_text)

    if legit_mask_np.sum() > 0:
        legit_desc_centroid = X_desc[legit_mask_np].mean(axis=0)
    else:
        legit_desc_centroid = X_desc.mean(axis=0)

    legit_desc_centroid = np.asarray(legit_desc_centroid)
    sim_to_legit_desc = cosine_similarity(X_desc, legit_desc_centroid).ravel()
    df["dist_embed_to_legit_desc"] = 1.0 - sim_to_legit_desc
    
    return df
    
    