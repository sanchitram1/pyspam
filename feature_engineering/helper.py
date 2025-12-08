import math
from typing import Iterable, Tuple


# -------------------------------------------------------
# Utility: extract base package name from requires_dist
# -------------------------------------------------------
def extract_pkg_name_from_requirement(req: str) -> str:
    """
    Roughly extract the package name from a requirement string.
    Examples:
      'requests>=2.0' -> 'requests'
      'google-cloud-storage==1.0; python_version<"3.10"' -> 'google-cloud-storage'
    """
    if not isinstance(req, str):
        return ""
    # Split on semicolon to drop markers, then on space and version operators
    base = req.split(";")[0].strip()
    # Remove version specifiers like <, >, =, !, ~
    # We just take the token until we hit one of those chars.
    for i, ch in enumerate(base):
        if ch in "<>!=~ ":
            base = base[:i]
            break
    return base.strip()


# -------------------------------------------------------
# Utility: Levenshtein distance (pure Python)
# -------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """
    Simple dynamic programming Levenshtein distance implementation.
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr_row
    return prev_row[-1]


def min_levenshtein_to_set(s: str, ref: Iterable[str]) -> int:
    s = s or ""
    if not ref:
        return math.inf
    s_lower = s.lower()
    return min(levenshtein(s_lower, r.lower()) for r in ref)


def min_levenshtein_to_set_with_threshold(
    s: str, ref: Iterable[str], threshold: int
) -> Tuple[int, bool]:
    """Return (distance, is_close_flag)."""
    dist = min_levenshtein_to_set(s, ref)
    return dist, dist <= threshold
