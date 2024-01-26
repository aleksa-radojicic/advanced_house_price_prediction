import copy
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.logger import log_message
from src.utils import FeaturesInfo, log_feature_info_dict


@dataclass
class FeatureEngineeringConfig:
    pass


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Adds derived numerical features to the input DataFrame (numerical feature engineering)."""

    previous_transformer_obj = None


    def __init__(self, verbose: int = 0) -> None:
        super().__init__()
        self.config = FeatureEngineeringConfig()
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        log_message(
            "Performing manipulations from feature engineering...", self.verbose
        )
        df = X.copy()
        features_info = self.previous_transformer_obj.features_info # type: ignore
        features_info = copy.deepcopy(features_info)

        df, self.features_info = self.fe_numerical(df, features_info)

        log_feature_info_dict(self.features_info, "feature engineering", self.verbose)

        log_message(
            "Performed manipulations from feature engineering successfully.",
            self.verbose,
        )
        return df

    def set_output(*args, **kwargs):
        pass

    def fe_numerical(self, df: pd.DataFrame, features_info: FeaturesInfo):
        # new_derived_ordinal = []
        new_derived_numerical = []

        df["TotalOverall"] = (
            df["OverallQ"] + df["OverallCond"]
        )  # Sum of overall columns from this dataset
        df["Year_Difference_Remod_Built"] = df["YearRemodAdd"] - df["YearBuilt"]
        df["Year_Difference_Sold_Built"] = df["YrSold"] - df["YearBuilt"]
        df["Year_Difference_Sold_Remod"] = df["YrSold"] - df["YearRemodAdd"]

        df["TotalHalfBathsAll"] = df["BsmtHalfBaths"] + df["HalfBaths"]
        df["TotalFullBathsAll"] = df["BsmtFullBaths"] + df["FullBaths"]
        df["TotalBathsAll"] = df["TotalHalfBathsAll"] + df["TotalFullBathsAll"]

        # All rooms (above ground and in basement) including bathrooms
        df["TotalRoomsWithBathsAll"] = df["TotalRooms"] + df["TotalBathsAll"]

        # Other rooms above ground
        df["OtherRooms"] = df["TotalRooms"] - df["Bedrooms"] - df["Kitchens"]

        # Creating ratios with respect to 'TotalRoomsWithBathsAll'
        ratios_to_TotalRoomsWithBathsAll = [
            "BsmtFullBaths",
            "BsmtHalfBaths",
            "FullBaths",
            "HalfBaths",
            "Bedrooms",
            "Kitchens",
            "OtherRooms",
            "TotalHalfBathsAll",
            "TotalFullBathsAll",
            "TotalBathsAll",
        ]
        new_derived_numerical_rel_TotalRoomsWithBathsAll = []
        for c in ratios_to_TotalRoomsWithBathsAll:
            df[f"%{c}"] = df[c] / df["TotalRoomsWithBathsAll"] * 100
            new_derived_numerical_rel_TotalRoomsWithBathsAll.append(f"%{c}")

        df["TotalPorchM2"] = (
            df["WoodDeckM2"]
            + df["OpenPorchM2"]
            + df["EnclosedPorchM2"]
            + df["3SsnPorchM2"]
            + df["ScreenPorchM2"]
        )

        df["DateSold"] = pd.to_datetime(dict(year=df["YrSold"], month=df["MoSold"], day=1))  # type: ignore

        df["OutdoorM2"] = df["LotM2"] - df["1stFlrM2"]
        df["IndoorM2"] = df["GrLivM2"] + df["TotalBsmtM2"] + df["GarageM2"]
        df["TotalM2"] = df["OutdoorM2"] + df["IndoorM2"]

        # Creating ratios with respect to 'IndoorM2'
        ratios_to_IndoorM2 = ["1stFlrM2", "2ndFlrM2", "TotalBsmtM2", "GarageM2"]
        new_derived_numerical_rel_M2 = []
        for c in ratios_to_IndoorM2:
            df[f"%{c}"] = df[c] / df["IndoorM2"] * 100
            new_derived_numerical_rel_M2.append(f"%{c}")

        # Creating ratios with respect to 'TotalM2'
        df["%OutdoorM2"] = df["OutdoorM2"] / df["TotalM2"] * 100

        new_derived_numerical.extend(
            [
                "TotalOverall",
                "Year_Difference_Remod_Built",
                "Year_Difference_Sold_Built",
                "Year_Difference_Sold_Remod",
            ]
            + [
                "TotalHalfBathsAll",
                "TotalFullBathsAll",
                "TotalBathsAll",
                "TotalRoomsWithBathsAll",
                "OtherRooms",
            ]
            + new_derived_numerical_rel_TotalRoomsWithBathsAll
            + ["TotalPorchM2", "OutdoorM2", "IndoorM2", "TotalM2"]
            + new_derived_numerical_rel_M2
            + ["%OutdoorM2"]
        )
        # new_derived_ordinal.extend()

        # derived_ordinal.extend(new_derived_ordinal)
        features_info["numerical"].extend(new_derived_numerical)
        features_info["derived_numerical"].extend(new_derived_numerical)
        features_info["other"].extend(["DateSold"])

        return df, features_info
