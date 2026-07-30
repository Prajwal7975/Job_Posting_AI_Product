"""
Stateless helper functions for the Common Feature Engineering stage.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException


_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z]+")


def generate_dataset_id(prefix: str = "fs") -> str:
    """Timestamp-based, sortable dataset id (UTC)."""
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def compute_schema_hash(df: pd.DataFrame) -> str:
    """
    Deterministic SHA-256 fingerprint of column names + dtypes.
    Order-sensitive: Column order shifts will produce a different hash, 
    matching standard production data warehouse behavior.
    """
    try:
        schema_repr = json.dumps(
            {col: str(dtype) for col, dtype in df.dtypes.astype(str).items()},
            sort_keys=False,
        )
        return hashlib.sha256(schema_repr.encode("utf-8")).hexdigest()
    except Exception as e:
        raise CustomException(e, sys)


def schema_dict(df: pd.DataFrame) -> Dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def ensure_dir(path: str | Path) -> Path:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        raise CustomException(e, sys)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    try:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=json_default)
        logging.info(f"Successfully saved JSON payload to {path}")
    except Exception as e:
        logging.error(f"Failed to save JSON payload to {path}: {e}")
        raise CustomException(e, sys)


def normalize_text(value: Any) -> str | None:
    # Handle iterable types first
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return None

    # Safe scalar null check
    if pd.isna(value):
        return None

    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text if text else None

def normalize_category(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    token = _NON_ALNUM_RE.sub("_", text).strip("_")
    return token.upper() if token else None


def normalize_list_value(
    value: Any,
    delimiter: str = "|",
    item_normalizer=normalize_category,
) -> str | None:

    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        items = value.ravel().tolist()
    else:
        if pd.isna(value):
            return None

        text = str(value).strip()

        if not text:
            return None

        items = re.split(r"[|,;/]", text)

    normalized = set()

    for item in items:

        if isinstance(item, np.ndarray):
            iterable = item.ravel().tolist()
        elif isinstance(item, pd.Series):
            iterable = item.tolist()
        elif isinstance(item, (list, tuple, set)):
            iterable = list(item)
        else:
            iterable = [item]

        for x in iterable:
            val = item_normalizer(x)
            if val is not None:
                normalized.add(val)

    if not normalized:
        return None

    return delimiter.join(sorted(normalized))
def list_length(value: str | None, delimiter: str = "|") -> int:
    if pd.isna(value):
        return 0
    return len([v for v in str(value).split(delimiter) if v])


def normalize_boolean(value: Any, truthy_values: Iterable[Any]) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    
    if isinstance(value, str):
        cleaned_val = value.strip().lower()
        normalized_truthy = {
            str(v).strip().lower() if isinstance(v, str) else v 
            for v in truthy_values
        }
        return cleaned_val in normalized_truthy

    return value in truthy_values


def to_datetime_safe(series: pd.Series, epoch_ms: bool = False) -> pd.Series:
    try:
        if epoch_ms:
            numeric = pd.to_numeric(series, errors="coerce")
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        return pd.to_datetime(series, errors="coerce", utc=False)
    except Exception as e:
        logging.error("Failed to parse datetime series: %s", str(e))
        raise CustomException(e, sys)


def infer_column_types(df: pd.DataFrame, config: Any) -> Dict[str, List[str]]:
    try:
        columns = set(df.columns.tolist())
        
        datetime_cols = [c for c in config.datetime_columns if c in columns]
        boolean_cols = [c for c in config.boolean_columns if c in columns]
        categorical_cols = [c for c in config.categorical_columns if c in columns]
        text_cols = [c for c in config.text_columns if c in columns]
        list_cols = [c for c in config.list_columns if c in columns]
        identifier_cols = [c for c in config.identifier_columns if c in columns]
        configured_numeric = [c for c in config.numeric_columns if c in columns]
        
        known_cols = set(
            datetime_cols + boolean_cols + categorical_cols + 
            text_cols + list_cols + configured_numeric + identifier_cols
        )
        
        auto_numeric = [
            c for c in columns 
            if c not in known_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        
        return {
            "datetime": datetime_cols,
            "boolean": boolean_cols,
            "categorical": categorical_cols,
            "text": text_cols,
            "list": list_cols,
            "numeric": configured_numeric + auto_numeric,
            "identifier": identifier_cols
        }
    except Exception as e:
        logging.error("Error inferring column types based on dataframe and config.")
        raise CustomException(e, sys)


def validate_feature_store_schema(df: pd.DataFrame, config: Any) -> None:
    logging.info("Starting strict schema validation for feature store output.")
    try:
        if config.fail_on_empty_output:
            if df.empty:
                raise ValueError("Validation failed: Feature store DataFrame has 0 rows.")
            if len(df.columns) == 0:
                raise ValueError("Validation failed: Feature store DataFrame has 0 columns.")
                
        if config.fail_on_duplicate_column_names:
            duplicates = df.columns[df.columns.duplicated()].tolist()
            if duplicates:
                raise ValueError(f"Validation failed: Duplicate column names detected: {duplicates}")

        expected_lists = [
            config.datetime_columns, config.boolean_columns, 
            config.categorical_columns, config.text_columns, config.list_columns
        ]
        
        expected_cols: Set[str] = {col for col_list in expected_lists for col in col_list}
        missing_cols = expected_cols - set(df.columns)
        
        if missing_cols:
            logging.warning(
                f"Validation warning: Missing expected columns (may be legitimately omitted): {missing_cols}"
            )
        
        expected_derived = {f for f, enabled in config.derived_feature_flags.items() if enabled}
        missing_derived = expected_derived - set(df.columns)
        
        if missing_derived:
            raise ValueError(f"Validation failed: Expected derived features were not created: {missing_derived}")

        logging.info("Schema validation passed successfully. No anomalies detected.")
    except Exception as e:
        logging.error(f"Schema validation error: {str(e)}")
        # If it's already a CustomException or ValueError, this wrapper will catch and wrap it properly.
        raise CustomException(e, sys)