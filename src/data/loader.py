"""
File loading utilities – supports CSV, Excel, and in-memory uploads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, IO

import pandas as pd

from utils import get_logger

log = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_file(source: Union[str, Path, IO]) -> pd.DataFrame:
    """
    Load a DataFrame from a file path or file-like object.
    Supports CSV and Excel formats.

    Args:
        source: File path (str/Path) or a file-like object (e.g. Streamlit UploadedFile).

    Returns:
        Raw DataFrame.

    Raises:
        ValueError: If the file format is not supported.
        IOError: If the file cannot be read.
    """
    try:
        # File-like object (Streamlit UploadedFile, BytesIO, etc.)
        if hasattr(source, "read"):
            name = getattr(source, "name", "")
            ext = Path(name).suffix.lower()
            if ext in (".xlsx", ".xls"):
                df = pd.read_excel(source)
            else:
                df = pd.read_csv(source)
            log.info("Loaded uploaded file '%s': %d rows", name, len(df))
            return df

        # Path-based loading
        path = Path(source)
        if not path.exists():
            raise IOError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {_SUPPORTED_EXTENSIONS}"
            )

        df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
        log.info("Loaded '%s': %d rows, %d cols", path.name, *df.shape)
        return df

    except (pd.errors.ParserError, Exception) as exc:
        log.exception("Failed to load file: %s", exc)
        raise
