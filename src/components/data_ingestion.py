import glob
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
                clean = re.sub(r'\([^)]*\)', '', str(comp)).strip().upper()
                if clean:
                    salts.append(clean)
        return " + ".join(sorted(salts)) if salts else None

    def _resolve_raw_path(self):
        if os.path.exists(self.ingestion_config.raw_data_path):
            return self.ingestion_config.raw_data_path
        csv_files = glob.glob(os.path.join('notebook', 'data', '*.csv'))
        return csv_files[0] if csv_files else None

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion and ETL component")
        try:
            required_metadata = {'pack_size_label', 'short_composition1'}

            if os.path.exists(self.ingestion_config.clean_data_path):
                existing_cols = set(pd.read_csv(self.ingestion_config.clean_data_path, nrows=1).columns)
                if required_metadata.issubset(existing_cols):
                    logging.info("Clean dataset already contains pack_size_label and compositions.")
                    return self.ingestion_config.clean_data_path
                logging.info("Rebuilding clean_medicines.csv to include pack_size_label and exact compositions.")

            raw_path = self._resolve_raw_path()
            if raw_path is None:
                if os.path.exists(self.ingestion_config.clean_data_path):
                    logging.warning("Raw CSV not found in notebook/data; falling back to existing clean_medicines.csv.")
                    return self.ingestion_config.clean_data_path
                raise FileNotFoundError("No raw dataset found in notebook/data/")

            df = pd.read_csv(raw_path, low_memory=False)
            logging.info(f"Read raw dataset from {raw_path} with shape: {df.shape}")

            if 'Is_discontinued' in df.columns:
                df = df[df['Is_discontinued'] == False].copy()

            if 'price(₹)' in df.columns:
                df.rename(columns={'price(₹)': 'price'}, inplace=True)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

            comp2_series = df['short_composition2'] if 'short_composition2' in df.columns else [None] * len(df)
            df['chemical_signature'] = [
                self.extract_clean_salt(c1, c2)
                for c1, c2 in zip(df['short_composition1'], comp2_series)
            ]

            df.dropna(subset=['name', 'manufacturer_name', 'price', 'chemical_signature'], inplace=True)

            keep_cols = [
                c for c in [
                    'name', 'manufacturer_name', 'price', 'pack_size_label',
                    'short_composition1', 'short_composition2', 'chemical_signature'
                ] if c in df.columns
            ]
            df = df[keep_cols]

            os.makedirs(os.path.dirname(self.ingestion_config.clean_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.clean_data_path, index=False)
            logging.info(f"Saved enriched clean dataset to {self.ingestion_config.clean_data_path}")

            return self.ingestion_config.clean_data_path

        except Exception as e:
            raise CustomException(e, sys)