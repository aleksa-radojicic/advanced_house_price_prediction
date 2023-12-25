import os
import sys
from dataclasses import dataclass
from importlib import reload

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

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

    def start(self):
        logging.info(msg="Data ingestion method or component started")

        try:
            df = pd.read_csv("data/train.csv")
            self.save_main_df_artifact(df)

            self.save_train_test_dfs_artifacts(df)

            df_test_submission = pd.read_csv("data/test.csv")
            self.save_test_submission_artifact(df_test_submission)

            logging.info(msg="Data ingestion method or component finished" + 5 * "*")

            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.test_submission_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def save_test_submission_artifact(self, df):
        logging.info("Read dataset for submission as dataframe finished " + 5 * "*")

        os.makedirs(
            os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
        )

        df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        logging.info("Saved main dataset artifact" + 5 * "*")

    def save_train_test_dfs_artifacts(self, df):
        logging.info(msg="Train test split started")
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=2023)

        df_train.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
        df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
        logging.info(msg="Train test split finished" + 5 * "*")

    def save_main_df_artifact(self, df):
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
        logging.info("Saved main dataset artifact" + 5 * "*")


if __name__ == "__main__":
    obj = DataIngestion()
    obj.start()
