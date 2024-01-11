from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.logger import logging
from src.utils import FeaturesInfo, log_feature_info_dict


@dataclass
class PostFEAnalysisConfig:
    pass


class PostFEAnalysisTransformer(BaseEstimator, TransformerMixin):
    """Manipulates data set as it was done in univariate analysis."""

    def __init__(self, previous_transformer_obj, verbose: int = 0) -> None:
        super().__init__()
        self.config = PostFEAnalysisConfig()
        self.previous_transformer_obj = previous_transformer_obj
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logging.info(
            "Performing manipulations from post feature engineering analysis..."
        )

        X = X.copy()

        features_info: FeaturesInfo = self.previous_transformer_obj.features_info

        # -------TEMPORARY-------------
        # df.loc[:, ["IndoorM2", "LotM2"]] = hp.NumericColumnsTransformer(
        #     method="outlier_replacement"
        # ).fit_transform(df, ["IndoorM2", "LotM2"])[["IndoorM2", "LotM2"]]

        derived_numerical_for_deletion = [
            "OutdoorM2",
            "TotalM2",
            "%BsmtHalfBaths",
            "%HalfBaths",
            "%TotalHalfBathsAll",
            "%BsmtFullBaths",
            "%2ndFlrM2",
        ]
        features_info["features_to_delete"].extend(derived_numerical_for_deletion)
        self.features_info = features_info

        if self.verbose > 0:
            log_feature_info_dict(
                self.features_info, title="post feature engineering analysis"
            )

        logging.info(
            "Performed manipulations from post feature engineering analysis successfully."
        )

        return X

    def set_output(*args, **kwargs):
        pass
