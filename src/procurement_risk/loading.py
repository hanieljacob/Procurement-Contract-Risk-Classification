"""Raw data loading with a Parquet cache.

The source CSV is 107 MB / 288,237 rows. Parsing it takes ~10 s; the Parquet
cache brings reloads under a second, which matters when iterating on features.
The cache is keyed on the source file's mtime+size, so an updated extract
invalidates it automatically rather than silently serving stale data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config


def _fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"name": path.name, "size": st.st_size, "mtime": int(st.st_mtime)}


def _meta_path(cache: Path) -> Path:
    return cache.with_suffix(".meta.json")


def load_raw(
    csv_path: Path | None = None,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load the contract-awards extract with source column names preserved.

    Returns the frame exactly as published (renamed to snake_case, nothing
    cleaned). All cleaning decisions live in `cleaning.py` so that the raw
    baseline stays inspectable -- we want to be able to show what we changed.
    """
    csv_path = csv_path or config.RAW_CSV
    cache_path = cache_path or config.PARQUET_CACHE

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Source extract not found at {csv_path}. Download DS00005 from "
            "https://financesone.worldbank.org/ and place it in data/."
        )

    fp = _fingerprint(csv_path)
    meta = _meta_path(cache_path)
    if use_cache and cache_path.exists() and meta.exists():
        if json.loads(meta.read_text()) == fp:
            return pd.read_parquet(cache_path)

    # Read everything as string except the two numerics. Deliberately NOT using
    # pandas' date parsing: an unparseable date must surface as a quality flag,
    # not be silently coerced to NaT during load.
    dtypes = {c: "string" for c in config.COLUMN_RENAMES}
    dtypes["Supplier Contract Amount (USD)"] = "float64"
    dtypes["WB Contract Number"] = "int64"

    df = pd.read_csv(csv_path, dtype=dtypes, low_memory=False)

    unexpected = set(df.columns) - set(config.COLUMN_RENAMES)
    if unexpected:
        raise ValueError(f"Extract has unexpected columns: {sorted(unexpected)}")
    missing = set(config.COLUMN_RENAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Extract is missing expected columns: {sorted(missing)}")

    df = df.rename(columns=config.COLUMN_RENAMES)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        meta.write_text(json.dumps(fp))

    return df
