import numpy as np


# Convert numpy arrays to lists, cuz fastapi doesn't like numpy arrays in json responses


def make_json_safe(record: dict) -> dict:
    safe = {}
    for k, v in record.items():
        # Convert numpy arrays (like your latest_project_urls, distinct_authors, etc.)
        if isinstance(v, np.ndarray):
            safe[k] = v.tolist()

        # Convert numpy scalar types (if any)
        elif isinstance(v, np.generic):
            safe[k] = v.item()

        else:
            safe[k] = v
    return safe
