"""
src/components/salary_predict/salary_preprocessor_builder.py

Salary Preprocessor Builder Component.

This component answers ONE question:
    "Given a SalaryExperimentConfig, how do we build the UNFITTED sklearn
     preprocessing object required by that experiment?"

Responsibilities & Constraints:
    - Constructs and returns an UNFITTED sklearn `ColumnTransformer` (or `None`).
    - Does NOT fit transformers or model estimators.
    - Does NOT load, read, or split datasets.
    - Does NOT evaluate metrics or choose winning configurations.
    - Does NOT call MLflow, joblib.dump, or mutate input configs/DataFrames.
    - Maintains strict separation of responsibilities.

Design & Compatibility:
    - All custom transformers are top-level module classes inheriting from
      `BaseEstimator` and `TransformerMixin` for clone/pickle/joblib compatibility.
    - Maintains sparse matrix representation end-to-end (sparse_threshold=1.0)
      to conserve memory.
    - Handles missing values, unseen categorical classes, and unknown skill
      tokens safely during inference.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.configs.salary_predict.salary_experiment_config import (
    ModelName,
    SalaryExperimentConfig,
    SkillEncoding,
    TEXT_FEATURE_SKILL_LIST,
    TEXT_FEATURE_TITLE,
    TfidfConfig,
)
from src.exception import CustomException
from src.logger import logging


# ======================================================================
# TOP-LEVEL CUSTOM TRANSFORMERS (SKLEARN / JOBLIB COMPATIBLE)
# ======================================================================


class SafeTextTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms 1D or 2D text inputs into a 1D numpy array of strings.

    Converts `None`, `NaN`, `pd.NA`, or non-string values into clean string
    representations (defaulting missing values to empty string `""`). This
    guarantees the 1D flat iterable sequence of string documents required by
    scikit-learn's `TfidfVectorizer`.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> SafeTextTransformer:
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            values = X.to_numpy()
        else:
            values = np.asarray(X)

        if values.ndim > 1:
            values = values.ravel()

        cleaned: List[str] = []
        for val in values:
            if val is None or pd.isna(val):
                cleaned.append("")
            else:
                cleaned.append(str(val))

        return np.array(cleaned, dtype=object)

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(list(input_features), dtype=object)
        return np.array(["text"], dtype=object)

class SafeCategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Normalizes categorical missing-value representations before
    sklearn imputation and encoding.

    Converts pd.NA, None, and other pandas-recognized missing
    values to np.nan while preserving valid categorical values.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> SafeCategoricalTransformer:
        return self

    def transform(self, X: Any) -> Any:
        # DataFrame input is expected from ColumnTransformer
        if isinstance(X, pd.DataFrame):
            X = X.copy()

            for col in X.columns:
                # Convert pandas extension/category/string dtype
                # to regular Python object dtype first.
                X[col] = X[col].astype(object)

                # Normalize pd.NA / None / NaN -> np.nan
                X[col] = X[col].where(
                    pd.notna(X[col]),
                    np.nan,
                )

            return X

        # Defensive support for numpy-like inputs
        X_array = np.asarray(X, dtype=object)

        return np.where(
            pd.isna(X_array),
            np.nan,
            X_array,
        )

    def get_feature_names_out(
        self,
        input_features: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        if input_features is None:
            return np.array([], dtype=object)

        return np.asarray(
            list(input_features),
            dtype=object,
        )



class SkillMultiHotTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms delimiter-separated skill strings into a sparse multi-hot matrix.

    Example:
        Input:  "python|sql|machine learning"
        Tokens: ["python", "sql", "machine learning"]

    Key Properties:
        - Learns skill vocabulary strictly during `fit()` on training data.
        - Unseen skill tokens encountered during `transform()` are safely ignored.
        - Duplicate tokens within a single row do not produce multi-count values.
        - Missing, empty, or whitespace-only inputs result in all-zero feature rows.
        - Outputs a scipy `csr_matrix` for sparse memory efficiency.
    """

    def __init__(
        self,
        delimiter: str = "|",
        lowercase: bool = True,
        strip_whitespace: bool = True,
    ) -> None:
        self.delimiter = delimiter
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace

    def _tokenize_row(self, row_val: Any) -> Set[str]:
        if row_val is None or pd.isna(row_val):
            return set()

        text = str(row_val)
        if not text.strip():
            return set()

        if self.lowercase:
            text = text.lower()

        raw_tokens = text.split(self.delimiter)
        tokens: Set[str] = set()
        for tok in raw_tokens:
            if self.strip_whitespace:
                tok = tok.strip()
            if tok:
                tokens.add(tok)
        return tokens

    def fit(self, X: Any, y: Any = None) -> SkillMultiHotTransformer:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            values = X.to_numpy()
        else:
            values = np.asarray(X)

        if values.ndim > 1:
            values = values.ravel()

        unique_vocab: Set[str] = set()
        for row in values:
            unique_vocab.update(self._tokenize_row(row))

        sorted_vocab = sorted(unique_vocab)
        self.vocabulary_: Dict[str, int] = {term: idx for idx, term in enumerate(sorted_vocab)}
        self.feature_names_in_: np.ndarray = np.array(sorted_vocab, dtype=object)
        return self

    def transform(self, X: Any) -> sp.csr_matrix:
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError("SkillMultiHotTransformer must be fitted before calling transform().")

        if isinstance(X, (pd.DataFrame, pd.Series)):
            values = X.to_numpy()
        else:
            values = np.asarray(X)

        if values.ndim > 1:
            values = values.ravel()

        n_samples = len(values)
        vocab_size = len(self.vocabulary_)

        if n_samples == 0 or vocab_size == 0:
            return sp.csr_matrix((n_samples, vocab_size), dtype=np.float64)

        rows: List[int] = []
        cols: List[int] = []

        for row_idx, row in enumerate(values):
            tokens = self._tokenize_row(row)
            for tok in tokens:
                col_idx = self.vocabulary_.get(tok)
                if col_idx is not None:
                    rows.append(row_idx)
                    cols.append(col_idx)

        data = np.ones(len(rows), dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(n_samples, vocab_size), dtype=np.float64)

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError("SkillMultiHotTransformer must be fitted before calling get_feature_names_out().")
        return np.array([f"skill_{name}" for name in self.feature_names_in_], dtype=object)


# ======================================================================
# HELPER UTILITIES
# ======================================================================


def _tfidf_from_config(tfidf_cfg: TfidfConfig) -> TfidfVectorizer:
    """Instantiates an unfitted TfidfVectorizer directly from a TfidfConfig."""
    return TfidfVectorizer(
        ngram_range=tfidf_cfg.ngram_range,
        min_df=tfidf_cfg.min_df,
        max_df=tfidf_cfg.max_df,
        sublinear_tf=tfidf_cfg.sublinear_tf,
        max_features=tfidf_cfg.max_features,
        strip_accents=tfidf_cfg.strip_accents,
        lowercase=tfidf_cfg.lowercase,
    )


def _build_one_hot_encoder() -> OneHotEncoder:
    """Builds OneHotEncoder supporting scikit-learn version differences for sparse output."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # Backward compatibility for scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)  # type: ignore


# ======================================================================
# MAIN PREPROCESSOR BUILDER
# ======================================================================


class SalaryPreprocessorBuilder:
    """
    Factory component that constructs unfitted scikit-learn `ColumnTransformer`
    objects based on declarative `SalaryExperimentConfig` parameters.
    """

    def build(self, config: SalaryExperimentConfig) -> Optional[ColumnTransformer]:
        """
        Constructs and returns an unfitted preprocessing object for an experiment.

        Args:
            config: Authoritative experiment configuration object.

        Returns:
            - `Optional[ColumnTransformer]`: Unfitted preprocessing object for
              feature-based experiments (E1-E6).
            - `None`: Exclusively for dummy baseline models (E0).

        Raises:
            TypeError: If `config` is not an instance of `SalaryExperimentConfig`.
            ValueError: If a non-dummy experiment contains no active predictor features
                        or fails to generate any preprocessor branches.
        """
        if not isinstance(config, SalaryExperimentConfig):
            raise TypeError(
                "config must be an instance of SalaryExperimentConfig, "
                f"got {type(config).__name__}."
            )

        try:
            logging.info(
                f"Building preprocessor for experiment_id='{config.experiment_id}' "
                f"({config.experiment_name})..."
            )

            # E0 / Dummy baseline models require no feature preprocessing
            if config.model_name == ModelName.DUMMY.value:
                logging.info(
                    "Dummy baseline model requires no feature preprocessor. Returning None."
                )
                return None

            # Non-dummy models MUST have active predictor features
            if not config.active_predictor_features:
                raise ValueError(
                    f"Non-dummy experiment '{config.experiment_id}' "
                    "must contain at least one active predictor feature."
                )

            transformers: List[Tuple[str, Any, Union[str, List[str]]]] = []

            # 1. Job Title TF-IDF Branch
            if config.use_title:
                if config.title_tfidf is None:
                    raise ValueError(
                        f"Experiment '{config.experiment_id}' specifies use_title=True "
                        "but title_tfidf configuration is missing."
                    )
                transformers.append(
                    ("title_tfidf", self._build_title_transformer(config.title_tfidf), TEXT_FEATURE_TITLE)
                )

            # 2. Skill List TF-IDF Branch
            if config.skill_encoding == SkillEncoding.TFIDF:
                if config.skill_tfidf is None:
                    raise ValueError(
                        f"Experiment '{config.experiment_id}' specifies skill_encoding=TFIDF "
                        "but skill_tfidf configuration is missing."
                    )
                transformers.append(
                    (
                        "skills_tfidf",
                        self._build_skill_tfidf_transformer(config.skill_tfidf),
                        TEXT_FEATURE_SKILL_LIST,
                    )
                )

            # 3. Skill List Multi-Hot Branch
            elif config.skill_encoding == SkillEncoding.MULTIHOT:
                transformers.append(
                    (
                        "skills_multihot",
                        self._build_skill_multihot_transformer(),
                        TEXT_FEATURE_SKILL_LIST,
                    )
                )

            # 4. Categorical Features Branch
            if config.categorical_features:
                transformers.append(
                    (
                        "categorical",
                        self._build_categorical_transformer(),
                        list(config.categorical_features),
                    )
                )

            # 5. Numeric Features Branch
            if config.numeric_features:
                transformers.append(
                    (
                        "numeric",
                        self._build_numeric_transformer(),
                        list(config.numeric_features),
                    )
                )

            if not transformers:
                raise ValueError(
                    f"No preprocessing branches were generated for non-dummy "
                    f"experiment '{config.experiment_id}'."
                )

            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                sparse_threshold=1.0,
            )

            logging.info(
                f"Successfully constructed unfitted ColumnTransformer for '{config.experiment_id}' "
                f"with {len(transformers)} branch(es): {[name for name, _, _ in transformers]}."
            )
            return preprocessor

        except (TypeError, ValueError):
            # Pass clean configuration errors through without wrapping
            raise
        except Exception as e:
            logging.error(
                f"Failed to build preprocessor for experiment_id='{config.experiment_id}': {e}"
            )
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------
    # BRANCH BUILDERS
    # ------------------------------------------------------------------

    def _build_title_transformer(self, tfidf_cfg: TfidfConfig) -> Pipeline:
        """Builds job title TF-IDF preprocessing pipeline."""
        return Pipeline(
            steps=[
                ("safe_text", SafeTextTransformer()),
                ("tfidf", _tfidf_from_config(tfidf_cfg)),
            ]
        )

    def _build_skill_tfidf_transformer(self, tfidf_cfg: TfidfConfig) -> Pipeline:
        """Builds skill list TF-IDF preprocessing pipeline."""
        return Pipeline(
            steps=[
                ("safe_text", SafeTextTransformer()),
                ("tfidf", _tfidf_from_config(tfidf_cfg)),
            ]
        )

    def _build_skill_multihot_transformer(self) -> Pipeline:
        """Builds skill list sparse multi-hot preprocessing pipeline."""
        return Pipeline(
            steps=[
                ("safe_text", SafeTextTransformer()),
                (
                    "multihot",
                    SkillMultiHotTransformer(
                        delimiter="|",
                        lowercase=True,
                        strip_whitespace=True,
                    ),
                ),
            ]
        )
        
    def _build_categorical_transformer(self) -> Pipeline:
        """Builds categorical imputation + one-hot encoding pipeline."""
        
        return Pipeline(steps=[("sanitizer",SafeCategoricalTransformer(),),("imputer",SimpleImputer(strategy="constant",fill_value="__MISSING__",),),("ohe",_build_one_hot_encoder(),),])


    def _build_numeric_transformer(self) -> Pipeline:
        """Builds numeric median-imputation + StandardScaler pipeline."""
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
            ]
        )

    # ------------------------------------------------------------------
    # SCHEMA VALIDATION HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def get_required_input_columns(config: SalaryExperimentConfig) -> List[str]:
        """
        Retrieves the list of raw input columns required by an experiment.
        Delegates to `config.active_predictor_features` as source of truth.
        """
        return list(config.active_predictor_features)

    @staticmethod
    def validate_input_columns(
        df_columns: Iterable[str], config: SalaryExperimentConfig
    ) -> List[str]:
        """
        Checks if a DataFrame's columns satisfy the experiment's feature requirements.

        Args:
            df_columns: Collection of column names present in the dataset.
            config: Authoritative experiment configuration.

        Returns:
            List of missing required column names (empty if all required columns exist).
        """
        present_set = set(df_columns)
        required_cols = SalaryPreprocessorBuilder.get_required_input_columns(config)
        missing_cols = [col for col in required_cols if col not in present_set]
        return missing_cols