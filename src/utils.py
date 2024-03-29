import os
import pickle
import sys
from typing import Dict, List, Union

import pandas as pd

from src.config import LABEL
from src.exception import CustomException
from src.logger import logging

FeaturesInfo = Dict[str, List[str]]
"""Custom type alias representing a dictionary containing information about feature categories.

Structure
---------
{
    'numerical': List[str]
        List of column names for numerical features.
    'binary': List[str]
        List of column names for binary features.
    'ordinal': List[str]
        List of column names for ordinal features.
    'nominal': List[str]
        List of column names for nominal features.
    'derived_numerical': List[str]
        List of column names for derived numerical features.
    'derived_binary': List[str]
        List of column names for derived binary features.
    'derived_ordinal': List[str]
        List of column names for derived ordinal features.
    'derived_nominal': List[str]
        List of column names for derived nominal features.
    'other': List[str]
        List of other features.
    'features_to_delete': List[str]
        List of column names for features to be deleted.
}
"""


def downcast_numerical_dtypes(df: pd.DataFrame, numerical: List[str]):
    df = df.copy()

    for c in numerical:
        df[c] = pd.to_numeric(df[c], downcast="signed", dtype_backend="pyarrow")
    return df


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


def delete_column_and_update_columns_list(
    df, column_names, columns_list, update_columns_list=True
):
    df.drop(column_names, axis=1, inplace=True)

    if update_columns_list:
        if type(column_names) == list:
            [columns_list.remove(c) for c in column_names]
        else:
            columns_list.remove(column_names)


def log_feature_info_dict(features_info: FeaturesInfo, title: str, verbose: int):
    if verbose > 1:
        features_info_str = ""
        for k, v in features_info.items():
            features_info_str += f"{k}: {v}\n"
        logging.info(f"FeaturesInfo after {title}:\n" + features_info_str)


def get_X_sets(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]]
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    if isinstance(dfs, pd.DataFrame):
        return dfs.drop(LABEL, axis=1)
    elif isinstance(dfs, list) and all(isinstance(df, pd.DataFrame) for df in dfs):
        return [df.drop(LABEL, axis=1) for df in dfs]
    else:
        raise ValueError("Input must be a single DataFrame or a list of DataFrames")


def get_y_sets(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]]
) -> Union[pd.Series, List[pd.Series]]:
    if isinstance(dfs, pd.DataFrame):
        return dfs[LABEL]
    elif isinstance(dfs, list) and all(isinstance(df, pd.DataFrame) for df in dfs):
        return [df[LABEL] for df in dfs]
    else:
        raise ValueError("Input must be a single DataFrame or a list of DataFrames")


def pickle_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def json_object(file_path, obj):
    import json

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as file_obj:
            json.dump(obj, file_obj, indent=1)

    except Exception as e:
        raise CustomException(e, sys)
