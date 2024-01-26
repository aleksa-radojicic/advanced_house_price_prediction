import os
import sys
from dataclasses import dataclass
from importlib import reload
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import INDEX, LABEL
from src.exception import CustomException
from src.logger import LOG_ENDING, logging

# import src.exception

# reload(src.exception)


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    test_submission_data_path: str = os.path.join("artifacts", "test_submission.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def start(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logging.info(msg="Data ingestion method or component started")
 
        try:
            df = pd.read_csv("data/train.csv", dtype_backend="pyarrow", index_col=INDEX)

            # Downcast label column
            df[LABEL] = pd.to_numeric(df[LABEL], downcast="signed")
            self.save_main_df_artifact(df)
            
            df_train, df_test = self.save_train_test_dfs_artifacts(df)

            df_test_submission = pd.read_csv(
                "data/test.csv", dtype_backend="pyarrow", index_col=INDEX
            )
            self.save_test_submission_artifact(df_test_submission)

            logging.info(msg="Data ingestion method or component finished" + LOG_ENDING)

            return (df, df_train, df_test, df_test_submission)
        except Exception as e:
            raise CustomException(e, sys) # type: ignore

    def save_test_submission_artifact(self, df: pd.DataFrame):
        logging.info("Read dataset for submission as dataframe finished " + LOG_ENDING)

        os.makedirs(
            os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
        )

        df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        logging.info("Saved main dataset artifact" + LOG_ENDING)

    def save_train_test_dfs_artifacts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(msg="Train test split started")
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=2023)

        df_train: pd.DataFrame
        df_test: pd.DataFrame

        # The deletion of rows should be handled differently
        delete_rows(df_train)

        df_train.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
        df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
        logging.info(msg="Train test split finished" + LOG_ENDING)

        return (df_train, df_test)

    def save_main_df_artifact(self, df):
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
        logging.info("Saved main dataset artifact" + LOG_ENDING)

def delete_rows(df: pd.DataFrame) -> None:
    df.drop([598], inplace=True)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.start()
