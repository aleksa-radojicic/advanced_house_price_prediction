import copy
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import power_transform

from src.config import *
from src.logger import logging
from src.utils import FeaturesInfo


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