# -------------------------------------------------------
# Step 5: H.* dependency-based offline features
#   - has_dependency_to_top_brand
#   - has_dependency_lev_close_to_brand
# -------------------------------------------------------
import math
from typing import List

import numpy as np
import pandas as pd
from feature_engineering.helper import extract_pkg_name_from_requirement, min_levenshtein_to_set
from feature_engineering.settings import BRAND_ALIASES, LEV_THRESHOLD, TOP_BRAND_PKGS


def deps_base_names(deps: List[str]) -> List[str]:
    """
    Extract base package names from dependency requirement strings.

    Parses dependency strings (e.g., "requests>=2.0") to extract
    just the package name (e.g., "requests").

    :param deps: List of dependency requirement strings
    :type deps: List[str]
    :return: List of base package names
    :rtype: List[str]
    """
    return [extract_pkg_name_from_requirement(d) for d in deps if isinstance(d, str)]


def handle_dependency(df: pd.DataFrame):
    """
    Add dependency-based features to the dataframe.

    Computes two features:
    1. has_dependency_to_top_brand: Whether package depends on any top brand package
    2. has_dependency_lev_close_to_brand: Whether any dependency is typosquatting a brand
    3. min_dep_lev_to_brand: Minimum Levenshtein distance from any dependency to brands

    :param df: DataFrame containing 'latest_dependencies' column
    :type df: pd.DataFrame
    :return: DataFrame with added dependency-based features
    :rtype: pd.DataFrame
    """
    # Extract base package names from dependency requirement strings
    dep_base_names = df["latest_dependencies"].apply(deps_base_names)

    # Feature 1: Check if package has dependency on any top brand package
    top_brand_set_lower = {b.lower() for b in TOP_BRAND_PKGS}
    df["has_dependency_to_top_brand"] = [
        int(any(base.lower() in top_brand_set_lower for base in bases))
        for bases in dep_base_names
    ]

    # Feature 2 & 3: Check for typosquatting in dependencies
    # Combine brand packages and aliases for comprehensive checking
    brand_set_lower = [b.lower() for b in TOP_BRAND_PKGS + BRAND_ALIASES]

    close_flags = []
    min_dists = []
    for bases in dep_base_names:
        # Handle empty dependency lists
        if not bases:
            min_dists.append(math.inf)
            close_flags.append(0)
            continue

        # Compute minimum Levenshtein distance across all dependencies to any brand
        # Filter out empty strings from base names
        dists = [
            min_levenshtein_to_set(base, brand_set_lower) for base in bases if base
        ]

        # Handle case where all base names were empty strings
        if not dists:
            min_dists.append(math.inf)
            close_flags.append(0)
        else:
            min_dist = min(dists)
            min_dists.append(min_dist)
            # Flag as close if within threshold (typosquatting detection)
            close_flags.append(int(min_dist <= LEV_THRESHOLD))

    # Convert infinity to NaN for proper handling in downstream processing
    df["min_dep_lev_to_brand"] = [d if d != math.inf else np.nan for d in min_dists]
    df["has_dependency_lev_close_to_brand"] = close_flags
    return df
