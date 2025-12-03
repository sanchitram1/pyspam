# -------------------------------------------------------
# Step 5: H.* dependency-based offline features
#   - has_dependency_to_top_brand
#   - has_dependency_lev_close_to_brand
# -------------------------------------------------------
import math
import numpy as np
import pandas as pd
from typing import List
from settings import TOP_BRAND_PKGS, LEV_THRESHOLD, BRAND_ALIASES
from helper import extract_pkg_name_from_requirement, min_levenshtein_to_set





def deps_base_names(deps: List[str]) -> List[str]:
    return [extract_pkg_name_from_requirement(d) for d in deps if isinstance(d, str)]


def handle_dependency(df: pd.DataFrame):

    dep_base_names = df["latest_dependencies"].apply(deps_base_names)

    # has_dependency_to_top_brand
    df["has_dependency_to_top_brand"] = [
        int(any(base.lower() in {b.lower() for b in TOP_BRAND_PKGS} for base in bases))
        for bases in dep_base_names
    ]

    # has_dependency_lev_close_to_brand
    brand_set_lower = [b.lower() for b in TOP_BRAND_PKGS + BRAND_ALIASES]

    close_flags = []
    min_dists = []
    for bases in dep_base_names:
        if not bases:
            min_dists.append(math.inf)
            close_flags.append(0)
            continue
        # compute minimum distance across all deps to any brand
        dists = [
            min_levenshtein_to_set(base, brand_set_lower)
            for base in bases
            if base
        ]
        if not dists:
            min_dists.append(math.inf)
            close_flags.append(0)
        else:
            min_dist = min(dists)
            min_dists.append(min_dist)
            close_flags.append(int(min_dist <= LEV_THRESHOLD))

    df["min_dep_lev_to_brand"] = [d if d != math.inf else np.nan for d in min_dists]
    df["has_dependency_lev_close_to_brand"] = close_flags
    return df
