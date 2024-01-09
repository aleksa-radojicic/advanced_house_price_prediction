from typing import Dict, List


RANDOM_SEED = 2023
DF_TRAIN_FILE_PATH = "data/train.csv"
DF_TEST_FILE_PATH = "data/test.csv"
INDEX = "Id"
LABEL = "SalePrice"
TEST_SIZE = 0.20
CV_SPLIT_SIZE = 5

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