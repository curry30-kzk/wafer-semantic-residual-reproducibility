"""Compatibility loader for the Python 2-era WM-811K pickle."""

from __future__ import annotations

import pickle
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
import pandas.core.indexes.base as pandas_indexes_base
import pandas.core.indexes.range as pandas_indexes_range


REQUIRED_COLUMNS = ("waferMap", "failureType", "trianTestLabel")


@contextmanager
def _legacy_pandas_index_aliases() -> Iterator[None]:
    """Temporarily expose module names embedded by old pandas versions."""

    alias_names = (
        "pandas.indexes",
        "pandas.indexes.base",
        "pandas.indexes.range",
    )
    previous = {name: sys.modules.get(name) for name in alias_names}

    legacy_package = types.ModuleType("pandas.indexes")
    legacy_package.__path__ = []
    legacy_package.base = pandas_indexes_base
    legacy_package.range = pandas_indexes_range
    sys.modules["pandas.indexes"] = legacy_package
    sys.modules["pandas.indexes.base"] = pandas_indexes_base
    sys.modules["pandas.indexes.range"] = pandas_indexes_range
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_wm811k_pickle(path: str | Path) -> pd.DataFrame:
    """Load WM-811K with old pandas aliases and validate its core schema."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"找不到 WM-811K 文件: {source}")

    with source.open("rb") as handle, _legacy_pandas_index_aliases():
        loaded = pickle.load(handle, encoding="latin1")

    if not isinstance(loaded, pd.DataFrame):
        raise TypeError(
            "LSWMD.pkl 读取结果不是 pandas.DataFrame，"
            f"而是 {type(loaded).__name__}"
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in loaded.columns]
    if missing:
        raise ValueError("WM-811K 数据缺少必需列: " + ", ".join(missing))
    return loaded
