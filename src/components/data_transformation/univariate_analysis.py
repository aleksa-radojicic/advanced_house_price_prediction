from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.components.data_transformation.utils import create_category
from src.logger import logging
from src.utils import (FeaturesInfo, delete_column_and_update_columns_list,
                       log_feature_info_dict)


@dataclass
class UnivariateAnalysisConfig:
    NUMERICAL_INIT_RENAME_MAP = {
        # Ensuring all columns denoting area end with -SF (Square Feet)
        "MasVnrArea": "MasVnrSF",
        "LotArea": "LotSF",
        "GrLivArea": "GrLivSF",
        "GarageArea": "GarageSF",
        "BsmtFinSF1": "BsmtFin1SF",
        "BsmtFinSF2": "BsmtFin2SF",
        "EnclosedPorch": "EnclosedPorchSF",
        "3SsnPorch": "3SsnPorchSF",
        "ScreenPorch": "ScreenPorchSF",
        "PoolArea": "PoolSF",
        # Ensuring all relevant columns end in -s (plural form)
        "BsmtFullBath": "BsmtFullBaths",
        "BsmtHalfBath": "BsmtHalfBaths",
        "FullBath": "FullBaths",
        "HalfBath": "HalfBaths",
        # Consistency of Total- prefix
        "TotRmsAbvGrd": "TotalRooms",
        # Removing -AbvGrd and adding -s
        "BedroomAbvGr": "Bedrooms",
        "KitchenAbvGr": "Kitchens",
        "OverallQual": "OverallQ",
    }
    BINARY_INIT_RENAME_MAP = {}
    ORDINAL_INIT_RENAME_MAP = {
        # Ensuring all columns denoting quality end with -Q
        "ExterQual": "ExterQ",
        "BsmtQual": "BsmtQ",
        "HeatingQC": "HeatingQ",
        "KitchenQual": "KitchenQ",
        "FireplaceQu": "FireplaceQ",
        "GarageQual": "GarageQ",
        "PoolQC": "PoolQ",
        "Fence": "FenceQ",
    }
    NOMINAL_INIT_RENAME_MAP = {
        # Adding suffix -Type for columns denoting type of sth.
        "Heating": "HeatingType",
        "BsmtFinType1": "BsmtFin1Type",
        "BsmtFinType2": "BsmtFin2Type",
        # Ensuring all columns denoting condition end with -Cond
        "SaleCondition": "SaleCond",
    }
    MIN_CATEGORY_CARDINALITY = 40
    NUMERICAL_INIT = [
        "LotFrontage",
        "LotArea",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "YearRemodAdd",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageYrBlt",
        "GarageCars",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
        "YrSold",
    ]
    BINARY_INIT = ["Street", "CentralAir"]
    ORDINAL_INIT = [
        "LotShape",
        "LandContour",
        "Utilities",
        "LandSlope",
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFin1Type",
        "BsmtFin2Type",
        "HeatingQC",
        "Electrical",
        "KitchenQual",
        "Functional",
        "FireplaceQu",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "PoolQC",
        "Fence",
    ]
    NOMINAL_INIT = [
        "MSSubClass",
        "MSZoning",
        "Alley",
        "LotConfig",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "Foundation",
        "Heating",
        "GarageType",
        "MiscFeature",
        "MoSold",
        "SaleType",
        "SaleCondition",
    ]


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


class UnivariateAnalysisTransformer(BaseEstimator, TransformerMixin):
    """Manipulates data set as it was done in univariate analysis."""

    def __init__(self, features_info: FeaturesInfo, verbose: int = 0) -> None:
        super().__init__()
        self.config = UnivariateAnalysisConfig()
        self.features_info = features_info
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logging.info("Performing manipulations from univariate analysis...")

        X = X.copy()

        # Renaming initial column names to more intuitive column names
        X, self.features_info = self.rename_init_col_names(X, self.features_info)
        X = self.downcast_df(X)

        # Convert from square feet to square meters variables denoting area
        X, self.features_info["numerical"] = convert_sf_to_m2(
            X, self.features_info["numerical"]
        )

        # Perhaps these methods can be reformated as Transformers?
        X = self.ua_process_numerical(X)
        X = self.ua_process_binary(X)
        X = self.ua_process_ordinal(X)

        if self.verbose > 0:
            log_feature_info_dict(self.features_info, title="univariate analysis")

        logging.info("Performed manipulations from univariate analysis successfully.")

        return X

    def set_output(*args, **kwargs):
        pass

    def downcast_df(self, X):
        # Downcast non-numerical columns
        X = downcast_nonnumerical_dtypes(
            X,
            self.features_info["binary"],
            self.features_info["ordinal"],
            self.features_info["nominal"],
        )

        # Downcast numerical columns
        for c in self.features_info["numerical"]:
            X[c] = pd.to_numeric(X[c], downcast="signed")

        # Downcast column names
        X.columns = X.columns.astype("string[pyarrow]")
        return X

    def ua_process_numerical(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.features_info["features_to_delete"].extend(
            ["LowQualFinM2", "3SsnPorchM2", "PoolM2", "PoolQ"]
        )
        return X

    def ua_process_binary(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        delete_column_and_update_columns_list(X, "Street", self.features_info["binary"])
        return X

    def ua_process_ordinal(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols_to_add_na_category = [
            "BsmtQ",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFin1Type",
            "BsmtFin2Type",
            "FireplaceQ",
            "GarageFinish",
            "GarageQ",
            "GarageCond",
            "FenceQ",
            "PoolQ",
        ]

        for col_name in cols_to_add_na_category:
            X[col_name] = X[col_name].cat.add_categories(["NA"]).fillna("NA")

        X["Electrical"] = X["Electrical"].replace(
            {"Mix": X["Electrical"].mode()[0]}  # assuming it's the most frequent cat
        )

        category_orderings_dict = {
            "LotShape": ["IR3", "IR2", "IR1", "Reg"],
            "LandContour": ["Low", "HLS", "Bnk", "Lvl"],
            "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
            "LandSlope": ["Gtl", "Mod", "Sev"],
            "ExterQ": ["Po", "Fa", "TA", "Gd", "Ex"],
            "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtQ": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
            "BsmtFin1Type": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "BsmtFin2Type": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "HeatingQ": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Electrical": ["FuseP", "FuseF", "FuseA", "SBrkr"],
            "KitchenQ": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
            "FireplaceQ": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
            "GarageQ": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "PavedDrive": ["N", "P", "Y"],
            "FenceQ": ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
            "PoolQ": ["NA", "Fa", "TA", "Gd", "Ex"],
        }

        for col_name in self.features_info["ordinal"]:
            X[col_name] = create_category(
                X[col_name], category_orderings_dict[col_name]
            )
        return X

    def rename_init_col_names(
        self,
        df: pd.DataFrame,
        features_info: FeaturesInfo,
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
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
                **self.config.NUMERICAL_INIT_RENAME_MAP,
                **self.config.BINARY_INIT_RENAME_MAP,
                **self.config.ORDINAL_INIT_RENAME_MAP,
                **self.config.NOMINAL_INIT_RENAME_MAP,
            }
        )

        # Create lists of renamed columns for each category
        numerical = [
            self.config.NUMERICAL_INIT_RENAME_MAP.get(c, c)
            for c in self.config.NUMERICAL_INIT
        ]
        binary = [
            self.config.BINARY_INIT_RENAME_MAP.get(c, c)
            for c in self.config.BINARY_INIT
        ]
        ordinal = [
            self.config.ORDINAL_INIT_RENAME_MAP.get(c, c)
            for c in self.config.ORDINAL_INIT
        ]
        nominal = [
            self.config.NOMINAL_INIT_RENAME_MAP.get(c, c)
            for c in self.config.NOMINAL_INIT
        ]

        # Initializing other types of features
        derived_numerical = []
        derived_binary = []
        derived_ordinal = []
        derived_nominal = []
        other = []
        features_to_delete = []

        features_info = {
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

        logging.info("Renamed data frame initial column names successfully.")
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

    logging.info("Downcast non-numerical data types successfully.")

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
