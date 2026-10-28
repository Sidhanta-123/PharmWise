import os
import sys
import re
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('notebook', 'data', 'medicines.csv')
    clean_data_path: str = os.path.join('artifacts', 'clean_medicines.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    @staticmethod
    def extract_clean_salt(comp1, comp2):
        salts = []
        for comp in [comp1, comp2]:
            if pd.notna(comp) and str(comp).strip() != "":
                # Remove dosage inside parentheses like (100mg) or (325mg/5ml)
                clean = re.sub(r'\([^)]*\)', '', str(comp)).strip().upper()
                if clean:
                    salts.append(clean)
        return " + ".join(sorted(salts)) if salts else None

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion and ETL component")
        try:
            if os.path.exists(self.ingestion_config.clean_data_path):
                logging.info("Clean dataset already exists in artifacts. Loading path directly.")
                return self.ingestion_config.clean_data_path

            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Read raw dataset with shape: {df.shape}")

            if 'Is_discontinued' in df.columns:
                df = df[df['Is_discontinued'] == False].copy()

            if 'price(₹)' in df.columns:
                df.rename(columns={'price(₹)': 'price'}, inplace=True)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

            if 'chemical_signature' not in df.columns:
                comp2_series = df['short_composition2'] if 'short_composition2' in df.columns else None
                df['chemical_signature'] = [
                    self.extract_clean_salt(c1, c2) 
                    for c1, c2 in zip(df['short_composition1'], comp2_series if comp2_series is not None else [None]*len(df))
                ]

            df.dropna(subset=['name', 'manufacturer_name', 'price', 'chemical_signature'], inplace=True)

            os.makedirs(os.path.dirname(self.ingestion_config.clean_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.clean_data_path, index=False)
            logging.info(f"Saved cleaned dataset to {self.ingestion_config.clean_data_path}")

            return self.ingestion_config.clean_data_path

        except Exception as e:
            raise CustomException(e, sys)