# Data Ingestion
#    ↓
# train.csv / test.csv / data.csv
#    ↓
# Data Transformation
#    ↓
# preprocessor.pkl + transformed arrays
#    ↓
# Model Training

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


# -------------------- CONFIG --------------------
@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join("artifacts")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


# -------------------- INGESTION --------------------
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion component")
        try:
            # Read dataset
            df = pd.read_csv(r"notebook/data/stud.csv")
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
    # Data Ingestion
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data_path,
        test_data_path
    )

    # Model Training
    model_trainer = ModelTrainer()
    r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print("Model training completed")
    print("R2 score:", r2)

# to run for model score -
# python -m src.components.data_ingestion

### OUTPUT - Model training completed
#            R2 score: 0.8804332983749564