from copy import deepcopy
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.components.data_transformation.utils import PipelineFeaturesInfo
from src.logger import log_message
from src.utils import FeaturesInfo, log_feature_info_dict


@dataclass
class PostFEAnalysisConfig:
    pass


class PostFEAnalysisTransformer(BaseEstimator, TransformerMixin):
    """Manipulates data set as it was done in univariate analysis."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__()
        self.config = PostFEAnalysisConfig()
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        log_message(
            "Performing manipulations from post feature engineering analysis...",
            self.verbose,
        )

        X = X.copy()

        features_info = deepcopy(PipelineFeaturesInfo.fe_transformer_fi)

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

        log_feature_info_dict(
            features_info, "post feature engineering analysis", self.verbose
        )

        log_message(
            "Performed manipulations from post feature engineering analysis successfully.",
            self.verbose,
        )

        if not PipelineFeaturesInfo.pfea_transformer_fi:
            PipelineFeaturesInfo.pfea_transformer_fi = features_info # type: ignore

        return X

    def set_output(*args, **kwargs):
        pass
