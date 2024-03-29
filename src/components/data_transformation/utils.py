from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import power_transform

from src.config import *
from src.logger import log_message, logging
from src.utils import FeaturesInfo, log_feature_info_dict


def create_category(df_column: pd.Series, all_categories: List[str]) -> pd.Series:
    """
    Creates a modified Pandas Series with the specified category ordering of categories, retaining
    existing categories.

    The primary purpose of this function is to ensure there is no risk of data leakage. For instance,
    it safeguards against the scenario where categories learned in a larger training set do not exist
    in a smaller training set (e.g. cross-validation train set).

    Parameters
    ----------
        df_column : pd.Series
            Categorical data
        all_categories : List[str]
            A list of all categories in correct order

    Returns
    -------
    df_column_modified : pd.Series
        A new Pandas Series with categories reduced according to the provided criteria.
    """
    df_column_modified = df_column.copy()
    existing_categories = df_column.unique()

    reduced_categories = [c for c in all_categories if c in existing_categories]

    df_column_modified = df_column_modified.cat.set_categories(reduced_categories)

    return df_column_modified


class NumericColumnsTransformer(BaseEstimator, TransformerMixin):
    """Transforms numeric columns using various methods.

    This transformer applies different methods to clean numeric columns in a DataFrame.
    The available methods include outlier replacement, logarithmic transformation, square root transformation,
    the original values (no transformation) and the Yeo-Johnson transformation.

    Parameters
    ----------
    method : str, optional (default="outlier_replacement")
        The method to use for transforming numeric columns. Available options are:
        - "outlier_replacement": Replaces outliers in numeric columns with the nearest non-outlier values.
        - "log": Applies a logarithmic transformation to numeric columns. It handles negative values by adding
          the minimum value to make all values non-negative before applying the transformation.
        - "square": Applies a square root transformation to numeric columns. It handles negative values by adding
          the minimum value to make all values non-negative before applying the transformation.
        - "original": Keeps the original values in the columns without any transformation.
        - "yeo_johnson": Applies the Yeo-Johnson transformation to numeric columns. This transformation
          works with both positive and negative values.

    Attributes
    ----------
    columns : array-like of strings
        The names of the numeric columns to be transformed.

    methods : array-like of strings
        An array-like object of available transformation methods.

    Raises
    ------
    ValueError
        If an invalid transformation method is provided.
    """

    methods = ["outlier_replacement", "log", "original", "yeo_johnson", "square"]

    def __init__(self, method="outlier_replacement"):
        if method not in self.methods:
            raise ValueError(f"Transformation method '{method}' is invalid.")

        self.method = method

    def fit(self, X, columns):
        self.columns = columns
        return self

    def transform(self, X):
        X = X.copy()

        if self.method == "outlier_replacement":
            X[self.columns] = X[self.columns].apply(self._apply_boxplot_outlier_removal)
        elif self.method == "log":
            X[self.columns] = X[self.columns].apply(self._apply_log)
        elif self.method == "yeo_johnson":
            X[self.columns] = power_transform(X=X[self.columns], method="yeo-johnson")
        elif self.method == "square":
            X[self.columns] = X[self.columns].apply(self._apply_square)
        elif self.method == "original":
            pass

        return X

    def _apply_boxplot_outlier_removal(self, column):
        """Replaces outliers in a Pandas Series with the nearest non-outlier values.

        Outliers are defined as values that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR, where Q1 and Q3 are the
        first and third quartiles and IQR is the interquartile range. This function follows the same approach
        as handling outliers in a boxplot.

        Parameters
        ----------
        column : pd.Series
            A Pandas Series containing the numerical data with potential outliers.

        Returns
        -------
        pd.Series
            A Pandas Series with outliers replaced by the nearest non-outlier values within the range
            [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        """
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        IQR = q3 - q1
        lower = q1 - 1.5 * IQR
        upper = q3 + 1.5 * IQR
        # n_outliers = ((column > upper) | (column < lower)).sum()

        return np.clip(a=column, a_min=lower, a_max=upper)

    def _apply_log(self, column):
        return (
            np.log1p(column - np.min(column)) if any(column < 0) else np.log1p(column)
        )

    def _apply_square(self, column):
        return np.sqrt(column - np.min(column)) if any(column < 0) else np.sqrt(column)


class DropColumnsScheduledForDeletionTransformer(BaseEstimator, TransformerMixin):
    """Drops columns scheduled for deletion from the data frame and updates
    other columns list."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__()
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        log_message("Dropping columns scheduled for deletion...", self.verbose)

        # Iterate through the columns to delete
        # Note: features_info['features_to_delete'] is copied because
        # values for key 'features_to_delete' are altered in the loop and
        # otherwise it would mess the loop
        features_info = deepcopy(PipelineFeaturesInfo.pfea_transformer_fi)

        df = df.copy()

        for column in features_info["features_to_delete"]:
            for k in features_info:
                if k == "features_to_delete":
                    continue
                if column in features_info[k]:
                    features_info[k].remove(column)

        # Drop the columns from the DataFrame
        df = df.drop(
            columns=features_info["features_to_delete"], axis=1, errors="ignore"
        )

        log_feature_info_dict(
            features_info, "dropping columns scheduled for deletion", self.verbose
        )

        log_message(
            "Dropped columns scheduled for deletion successfully.", self.verbose
        )

        if not PipelineFeaturesInfo.drop_columns_scheduled_for_deletion_transformer_fi:
            PipelineFeaturesInfo.drop_columns_scheduled_for_deletion_transformer_fi = features_info  # type: ignore

        return df

    def set_output(*args, **kwargs):
        pass


class ColumnDtPrefixerTransformer(BaseEstimator, TransformerMixin):
    """Adds prefix to column names denoting its data type."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__()
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        features_info = deepcopy(
            PipelineFeaturesInfo.drop_columns_scheduled_for_deletion_transformer_fi
        )

        X.rename(
            columns={c: f"numerical__{c}" for c in features_info["numerical"]},
            inplace=True,
        )
        X.rename(
            columns={c: f"binary__{c}" for c in features_info["binary"]}, inplace=True
        )
        X.rename(
            columns={c: f"ordinal__{c}" for c in features_info["ordinal"]}, inplace=True
        )
        X.rename(
            columns={c: f"nominal__{c}" for c in features_info["nominal"]}, inplace=True
        )

        log_feature_info_dict(
            features_info, "adding data type prefix to columns", self.verbose
        )

        log_message("Added data type prefix to columns successfully.", self.verbose)

        if not PipelineFeaturesInfo.column_dt_prefixer_fi:
            PipelineFeaturesInfo.column_dt_prefixer_fi = features_info  # type: ignore

        return X

    def set_output(*args, **kwargs):
        pass


class LabelTransformer(FunctionTransformer):
    def __init__(self, verbose: int = 0, **kwargs):
        super().__init__(
            func=self.transform_func,
            inverse_func=self.inverse_transform_func,
            check_inverse=False,
            **kwargs,
        )
        self.verbose = verbose

    def transform(self, X):
        result = super().transform(X)
        log_message("Label transformed.", self.verbose)
        return result

    def inverse_transform(self, X):
        result = super().inverse_transform(X)
        log_message("Label transformed back to the original.", self.verbose)
        return result

    def transform_func(self, X, y=None):
        return np.log1p(X)

    def inverse_transform_func(self, X, y=None):
        return np.expm1(X)


class PipelineFeaturesInfo:
    ua_transformer_fi: FeaturesInfo = {}
    ma_transformer_fi: FeaturesInfo = {}
    fe_transformer_fi: FeaturesInfo = {}
    pfea_transformer_fi: FeaturesInfo = {}
    drop_columns_scheduled_for_deletion_transformer_fi: FeaturesInfo = {}
    column_dt_prefixer_fi: FeaturesInfo = {}
