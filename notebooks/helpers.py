import copy
from importlib import reload
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import power_transform

from notebooks.config import (BINARY_INIT, BINARY_INIT_RENAME_MAP,
                              NOMINAL_INIT, NOMINAL_INIT_RENAME_MAP,
                              NUMERICAL_INIT, NUMERICAL_INIT_RENAME_MAP,
                              ORDINAL_INIT, ORDINAL_INIT_RENAME_MAP,
                              FeaturesInfo)


def downcast_numerical_dtypes(df: pd.DataFrame, numerical: List[str]):
    df = df.copy()

    for c in numerical:
        df[c] = pd.to_numeric(df[c], downcast="signed", dtype_backend="pyarrow")
    return df


def set_values_MSSubClass(df, x):
    if x != 20 and x != 60 and x != 50:
        return "Other"
    else:
        return str(x)


def set_values_Condition_1(df, x):
    if x != "Norm" and x != "Feedr" and x != "Artery":
        return "Other"
    else:
        return x


def set_values_HouseStyle(df, x):
    if x not in df.HouseStyle.value_counts()[0:3].index.tolist():
        return "Other"
    else:
        return x


def set_values_RoofStyle(df, x):
    if x not in df.RoofStyle.value_counts()[0:2].index.tolist():
        return "Other"
    else:
        return x


def set_values_Foundation(df, x):
    if x not in df.Foundation.value_counts()[0:2].index.tolist():
        return "Other"
    else:
        return x


def threshold_for_category(df, column, thr):
    a = pd.DataFrame(df[column].value_counts(normalize=True) * 100)
    return a.loc[a.proportion > thr,].index.tolist()  # type: ignore


def set_values_Neighborhood(df, x):
    if x not in threshold_for_category(df, "Neighborhood", 5):
        return "Other"
    else:
        return x


def set_values_Exterior1st(df, x):
    if x not in threshold_for_category(df, "Exterior1st", 7):
        return "Other"
    else:
        return x


def set_values_Exterior2nd(df, x):
    if x not in threshold_for_category(df, "Exterior2nd", 7):
        return "Other"
    else:
        return x


def drop_columns_scheduled_for_deletion(df: pd.DataFrame, features_info: FeaturesInfo):
    """Drops columns scheduled for deletion from the data frame and updates
    other columns list."""
    # Iterate through the columns to delete
    # Note: features_info['features_to_delete'] is copied because
    # values for key 'features_to_delete' are altered in the loop and
    # otherwise it would mess the loop
    df = df.copy()
    features_info = copy.deepcopy(features_info)

    for column in features_info["features_to_delete"]:
        for k in features_info:
            if k == "features_to_delete":
                continue
            if column in features_info[k]:
                features_info[k].remove(column)

    # Drop the columns from the DataFrame
    df = df.drop(columns=features_info["features_to_delete"], axis=1, errors="ignore")

    return df, features_info


def downcast_nonnumerical_dtypes(df, binary, ordinal, nominal):
    df = df.copy()

    for c in binary:
        df[c] = (
            df.loc[:, c]
            .apply(lambda x: True if x == "Y" else False)
            .astype("bool[pyarrow]")
        )

    for c in ordinal:
        df[c] = pd.Categorical(df.loc[:, c], ordered=True)

    for c in nominal:
        df[c] = pd.Categorical(df.loc[:, c], ordered=False)

    return df


def show_hist_box_numerical_col(df, numerical_col):
    fig, axs = plt.subplots(1, 2)

    df[numerical_col].hist(ax=axs[0])
    ax2 = df[numerical_col].plot.kde(ax=axs[0], secondary_y=True)
    ax2.set_ylim(0)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    df[numerical_col].plot.box(ax=axs[1])

    fig.tight_layout()

    print(f"Univariate analysis of '{numerical_col}' column")
    print("Histogram and box plot")
    plt.show()
    print("Descriptive statistics")
    display(df[numerical_col].describe())

    print(
        f"Variance: {stats.variation(df[numerical_col].astype('double'), ddof=1, nan_policy='omit')}"
    )
    print(
        f"Skewness: {stats.skew(df[numerical_col].astype('double'), nan_policy='omit')}"
    )
    print(
        f"Kurtosis: {stats.kurtosis(df[numerical_col].astype('double'), nan_policy='omit')}\n"
    )

    print("NA values")
    n_na_values = df[numerical_col].isna().sum()
    perc_na_values = n_na_values / df[numerical_col].shape[0] * 100
    print(f"Count [n]: {n_na_values}")
    print(f"Percentage [%]: {perc_na_values}%")


def rename_init_col_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, FeaturesInfo]:
    """Renames columns of a DataFrame based on predefined mapping dictionaries and returns
    a renamed DataFrame and a dictionary of renamed column names for different categories.

    Predefined mappings should be in the following format:
    {
        'old_column_name': 'new_column_name',
        ...
    }

    Predefined column names ending with -INIT look like this:
    List[str]

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to rename columns for.

    Returns
    -------
    renamed_df : pd.DataFrame
        A copy of the input DataFrame with columns renamed based on predefined mappings.
    features_info : Dict[str, List[str]]
        A dictionary containing lists of column names after renaming for different categories.
        - 'numerical': List of column names from the 'NUMERICAL_INIT' category after renaming.
        - 'binary': List of column names from the 'BINARY_INIT' category after renaming.
        - 'ordinal': List of column names from the 'ORDINAL_INIT' category after renaming.
        - 'nominal': List of column names from the 'NOMINAL_INIT' category after renaming.
        - 'derived_numerical': List of column names for derived numerical features.
        - 'derived_binary': List of column names for derived binary features.
        - 'derived_ordinal': List of column names for derived ordinal features.
        - 'derived_nominal': List of column names for derived nominal features.
        - 'other': List of other features.
        - 'features_to_delete': List of features to be deleted.

    Note
    ----
    This function assumes the existence of predefined mapping dictionaries and category
    lists for 'NUMERICAL_INIT', 'BINARY_INIT', 'ORDINAL_INIT', and 'NOMINAL_INIT'.
    """
    df = df.copy()

    df = df.rename(
        columns={
            **NUMERICAL_INIT_RENAME_MAP,
            **BINARY_INIT_RENAME_MAP,
            **ORDINAL_INIT_RENAME_MAP,
            **NOMINAL_INIT_RENAME_MAP,
        }
    )

    # Create lists of renamed columns for each category
    numerical = [NUMERICAL_INIT_RENAME_MAP.get(c, c) for c in NUMERICAL_INIT]
    binary = [BINARY_INIT_RENAME_MAP.get(c, c) for c in BINARY_INIT]
    ordinal = [ORDINAL_INIT_RENAME_MAP.get(c, c) for c in ORDINAL_INIT]
    nominal = [NOMINAL_INIT_RENAME_MAP.get(c, c) for c in NOMINAL_INIT]

    # Initializing other types of features
    derived_numerical = []
    derived_binary = []
    derived_ordinal = []
    derived_nominal = []
    other = []
    features_to_delete = []

    features_info: FeaturesInfo = {
        "numerical": numerical,
        "binary": binary,
        "ordinal": ordinal,
        "nominal": nominal,
        "derived_numerical": derived_numerical,
        "derived_binary": derived_binary,
        "derived_ordinal": derived_ordinal,
        "derived_nominal": derived_nominal,
        "other": other,
        "features_to_delete": features_to_delete,
    }

    return df, features_info


def convert_sf_to_m2(
    df_sf, numerical_with_sf, column_suffix_sf="SF", column_suffix_m2="M2"
):
    # Conversion from square feet to square meters
    single_sf_in_m2 = 0.09290304

    df_m2 = df_sf.copy()

    # Identify columns with the specified suffix (squared feets and squared meters)
    columns_sf = [c for c in numerical_with_sf if c.endswith(column_suffix_sf)]
    columns_m2 = [
        f"{c.removesuffix(column_suffix_sf)}{column_suffix_m2}" for c in columns_sf
    ]

    # Create a dictionary to map SF column names to M2 column names
    sf_to_m2_rename_map = dict(zip(columns_sf, columns_m2))
    # Rename columns in the DataFrame
    df_m2.rename(columns=sf_to_m2_rename_map, inplace=True)
    # Convert SF values to M2
    df_m2[columns_m2] = df_sf[columns_sf] * single_sf_in_m2

    # Update the list of numerical columns with M2 suffix
    numerical_with_m2 = [sf_to_m2_rename_map.get(c, c) for c in numerical_with_sf]

    return df_m2, numerical_with_m2


def get_value_counts_freq_with_perc(df, column):
    value_counts_freq = df.loc[:, column].value_counts(dropna=False)
    value_counts_perc = value_counts_freq / df.shape[0] * 100

    result = pd.concat([value_counts_freq, value_counts_perc], axis=1)
    result.columns.values[1] = "percentage [%]"
    return result


def get_correlations_and_pvals(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    corr_with_derived = (
        df[columns]
        .corr()
        .unstack()
        .sort_values(key=lambda x: np.abs(x), ascending=False)  # type: ignore
    )
    corr_with_derived = pd.DataFrame(
        {
            "r": corr_with_derived[
                corr_with_derived.index.get_level_values(0)
                != corr_with_derived.index.get_level_values(1)
            ]
        }
    )

    col_pairs = []
    idx = []
    pvalues = []

    # Iterate on index pairs to remove redundant pairs
    for i, tuple_pair in enumerate(corr_with_derived.index.values):
        set_pair = set(tuple_pair)
        if set_pair not in col_pairs:
            col_pairs.append(set_pair)
            idx.append(i)
            pvalues.append(
                stats.pearsonr(x=df[tuple_pair[0]], y=df[tuple_pair[1]]).pvalue  # type: ignore
            )

    corr_with_derived = corr_with_derived.iloc[idx,]  # type: ignore
    corr_with_derived["pvalues"] = pvalues

    return corr_with_derived


def display_feature_name_heading(feature):
    display(Markdown(f"<h3>'{feature}' feature</h3>"))


def get_outliers_idx_using_boxplot(df, column):
    Q3 = np.quantile(df[column], 0.75)
    Q1 = np.quantile(df[column], 0.25)
    IQR = Q3 - Q1

    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR

    result = {
        "top": df.loc[df[column] > upper_range, :].index,
        "bottom": df.loc[df[column] < lower_range, :].index,
    }
    return result


def show_positive_vals_percs(df, column):
    n_rows = df.shape[0]

    positives_n = (df[column] > 0).sum()
    positives_perc = positives_n / n_rows * 100

    print(f"Positive values '{column}'")
    print(f"Count [n]: {positives_n}")
    print(f"Percentage [%]: {positives_perc}%")


def show_zero_vals_percs(df, column):
    n_rows = df.shape[0]

    zeros_n = (df[column] == 0).sum()
    zeros_perc = zeros_n / n_rows * 100

    print(f"Zero values '{column}'")
    print(f"Count [n]: {zeros_n}")
    print(f"Percentage [%]: {zeros_perc}%")


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


def category_influence_mean_value(df, categ):
    # df_new = df[categ].cat.add_categories(["Unknown"]).fillna("Unknown").copy()
    display(f"{categ}")
    if df[categ].isna().sum() > 0:
        display(df[categ].value_counts(dropna=False))
        display(df[categ].value_counts(dropna=False, normalize=True) * 100)

    else:
        display(df[categ].value_counts())
        display(df[categ].value_counts(normalize=True) * 100)

    display(f"Broj NA vrednosti {df[categ].isna().sum()}")
    # Group data by the categorical variable
    grouped_data = df.groupby(categ)["SalePrice"].mean().reset_index()
    grouped_data = grouped_data.loc[grouped_data["SalePrice"].isna() == False,]

    # Create a bar plot
    plt.figure(figsize=(12, 6))

    plt.bar(grouped_data[categ], grouped_data["SalePrice"].sort_values())

    plt.xlabel("Category")

    plt.ylabel("Mean Output")
    plt.title(f"Influence of {categ} Variable on Output")
    plt.xticks(grouped_data[categ])
    plt.show()


def scatter_plots(df, x):
    if df[x].isna().sum() > 0:
        df = df.loc[df[x].isna() == False,]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["SalePrice"], df[x].dropna(), color="b", marker="o", label="Data Points"
    )

    plt.xlabel(f"Sale Price")
    plt.ylabel(f"{x}")
    plt.title(f"Sale price and {x}")
    plt.legend()

    plt.grid(True)
    plt.show()


def delete_column_and_update_columns_list(
    df, column_names, columns_list, update_columns_list=True
):
    df.drop(column_names, axis=1, inplace=True)

    if update_columns_list:
        if type(column_names) == list:
            [columns_list.remove(c) for c in column_names]
        else:
            columns_list.remove(column_names)


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
