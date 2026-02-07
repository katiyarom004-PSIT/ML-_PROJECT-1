import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


# -------------------- PROJECT ROOT --------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


# -------------------- CONFIG --------------------
@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join(BASE_DIR, "artifacts")
    train_data_path: str = os.path.join(BASE_DIR, "artifacts", "train.csv")
    test_data_path: str = os.path.join(BASE_DIR, "artifacts", "test.csv")
    raw_data_path: str = os.path.join(BASE_DIR, "artifacts", "data.csv")


# -------------------- INGESTION --------------------
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")
        try:
            # Read dataset
            df = pd.read_csv(
                r"C:\Users\katiy\OneDrive\Desktop\ML PROJ-1\notebook\data\stud.csv"
            )
            logging.info("Dataset read successfully")

            # Create artifacts directory
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # Save raw data
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            # Train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train & test data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# -------------------- RUN --------------------
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
