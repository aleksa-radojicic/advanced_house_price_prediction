import copy
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.logger import log_message
from src.utils import log_feature_info_dict


@dataclass
class MultivariateAnalysisConfig:
    pass


class MultivariateAnalysisTransformer(BaseEstimator, TransformerMixin):
    """Manipulates data set as it was done in multivariate analysis."""
    
    previous_transformer_obj = None

    def __init__(self, verbose: int = 0) -> None:
        super().__init__()
        self.config = MultivariateAnalysisConfig()
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        log_message(
            "Performing manipulations from multivariate analysis...", self.verbose
        )

        features_info = self.previous_transformer_obj.features_info # type: ignore
        self.features_info = copy.deepcopy(features_info)

        X = X.copy()
        X = self.ma_process_missing_values(X)
        self.features_info["features_to_delete"].append("GarageCond")

        log_feature_info_dict(self.features_info, "multivariate analysis", self.verbose)

        log_message(
            "Performed manipulations from univariate multivariate successfully.",
            self.verbose,
        )
        return X

    def set_output(*args, **kwargs):
        pass

    def ma_process_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self.ma_process_missing_values_categorical(X)
        X = self.ma_process_missing_values_numerical(X)
        X = self.ma_process_missing_values_general(X)
        return X

    def ma_process_missing_values_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Alley"] = X["Alley"].cat.add_categories("NA_str").fillna("NA_str")
        X["MasVnrType"] = X["MasVnrType"].cat.add_categories("NA_str").fillna("NA_str")
        X["GarageType"] = X["GarageType"].cat.add_categories("NA_str").fillna("NA_str")
        X["MiscFeature"] = (
            X["MiscFeature"].cat.add_categories("NA_str").fillna("NA_str")
        )
        return X

    def ma_process_missing_values_numerical(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["LotFrontage"].fillna(X["LotFrontage"].median(), inplace=True)
        X["MasVnrM2"].fillna(X["MasVnrM2"].median(), inplace=True)
        X["GarageYrBlt"].fillna(0, inplace=True)
        return X

    def ma_process_missing_values_general(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X_na_by_column = X.isna().sum()
        X_na_columns = X_na_by_column[X_na_by_column > 0].index.values

        for c in X_na_columns:
            if c in self.features_info["numerical"]:
                X[c].fillna(X[c].median(), inplace=True)
            elif (
                c
                in self.features_info["ordinal"]
                + self.features_info["nominal"]
                + self.features_info["binary"]
            ):
                X[c].fillna(X[c].mode().values[0], inplace=True)
        return X
