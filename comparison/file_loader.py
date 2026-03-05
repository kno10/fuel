from pathlib import Path

import numpy as np


def load_numeric_data(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    first_non_empty = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                first_non_empty = stripped
                break

    if first_non_empty is None:
        raise ValueError("CSV has no data rows")

    delimiter = "," if "," in first_non_empty else None

    try:
        data = np.loadtxt(path, dtype=np.float64, delimiter=delimiter, ndmin=2)
    except Exception as exc:
        raise ValueError(f"failed to parse numeric data from {path}: {exc}") from exc

    if data.size == 0:
        raise ValueError("CSV has no data rows")

    return data