from importlib.metadata import files
import os
from dataclasses import dataclass, field
import sys

from plotly import files
from typing import Dict, Tuple
from src.logger import logging
from src.exception import CustomException
import pandas as pd


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the Data Ingestion component.
    """

    raw_data_dir: str = os.path.join("data", "raw")

    # Canonical table name -> accepted filenames
    mandatory_tables: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "postings": (
                "postings.csv",
                "job_postings.csv",
            ),
            "companies": (
                "companies.csv",
            ),
            "industries": (
                "industries.csv",
            ),
            "job_industries": (
                "job_industries.csv",
            ),
        }
    )

    optional_tables: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "benefits": (
                "benefits.csv",
            ),
            "company_specialities": (
                "company_specialities.csv",
            ),
            "employee_counts": (
                "employee_counts.csv",
            ),
            "job_skills": (
                "job_skills.csv",
            ),
            "skills": (
                "skills.csv",
            ),
            "salaries": (
                "salaries.csv",
            ),
        }
    )
    
    @property
    def all_files(self) -> Tuple[str, ...]:
        files: list[str] = []
            
        for accepted_files in self.mandatory_tables.values():
            files.extend(accepted_files)
                
        for accepted_files in self.optional_tables.values():
            files.extend(accepted_files)
                
            return tuple(files)
        
class DataIngestion:

    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()

    def discover_versions(self) -> list[str]:
        """
        Discover all dataset version folders.
        """
        
        versions = sorted(
            (
                folder
                for folder in os.listdir(self.config.raw_data_dir)
                if folder.startswith("Version_")
                and os.path.isdir(os.path.join(self.config.raw_data_dir, folder))
                ),
            key=lambda x: int(x.split("_")[1])
            )

        if not versions:
            raise FileNotFoundError(
                f"No dataset versions found in "
                f"{self.config.raw_data_dir}"
            )

        return versions

    def _load_csv(self,version_path: str,file_name: str) -> pd.DataFrame:
        """
        Load a single CSV file.
        """
        file_path = os.path.join(version_path, file_name)
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            raise CustomException(e, sys)

        if df.empty:
            raise ValueError(
                f"{file_name} is empty."
            )

        return df

    def load_version(
        self,
        version_name: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load every CSV belonging to one dataset version.
        """

        version_path = os.path.join(
            self.config.raw_data_dir,
            version_name,
        )

        if not os.path.isdir(version_path):
            raise FileNotFoundError(
                f"{version_name} not found."
            )

        version_data: Dict[str, pd.DataFrame] = {}

        # ---------- Mandatory ----------
        for table_name, accepted_files in self.config.mandatory_tables.items():

            matched_file = None

            for file_name in accepted_files:

                file_path = os.path.join(
                    version_path,
                    file_name,
                )

                if os.path.exists(file_path):
                    matched_file = file_name
                    break

            if matched_file is None:
                raise FileNotFoundError(
                    f"[{version_name}] Missing mandatory table '{table_name}'. "
                    f"Expected one of: {accepted_files}"
                )

            df = self._load_csv(
                version_path,
                matched_file,
            )

            version_data[table_name] = df

            logging.info(
                "[%s] Loaded %s as '%s' (%d rows)",
                version_name,
                matched_file,
                table_name,
                len(df),
            )

        # ---------- Optional ----------
        for table_name, accepted_files in self.config.optional_tables.items():

            matched_file = None

            for file_name in accepted_files:

                file_path = os.path.join(
                    version_path,
                    file_name,
                )

                if os.path.exists(file_path):
                    matched_file = file_name
                    break

            if matched_file is None:

                logging.warning(
                    "[%s] Optional table '%s' missing. Expected one of: %s",
                    version_name,
                    table_name,
                    accepted_files,
                )

                continue

            df = self._load_csv(
                version_path,
                matched_file,
            )

            version_data[table_name] = df

            logging.info(
                "[%s] Loaded %s as '%s' (%d rows)",
                version_name,
                matched_file,
                table_name,
                len(df),
            )

        return version_data

    def initiate_data_ingestion(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load every dataset version.
        """

        logging.info("=" * 70)
        logging.info("DATA INGESTION STARTED")
        logging.info("=" * 70)

        try:

            all_versions: Dict[str, Dict[str, pd.DataFrame]] = {}

            versions = self.discover_versions()

            logging.info(
                "Versions Found: %s",
                ", ".join(versions),
            )

            for version in versions:

                logging.info(
                    "Loading %s",
                    version,
                )

                all_versions[version] = self.load_version(version)

            logging.info("=" * 70)
            logging.info(
                "Loaded %d dataset version(s).",
                len(all_versions),
            )
            logging.info("DATA INGESTION COMPLETED")
            logging.info("=" * 70)

            return all_versions

        except Exception as e:

            logging.exception(
                "Data Ingestion Failed"
            )

            raise CustomException(e, sys)
        
if __name__ == "__main__":

    ingestion = DataIngestion()

    datasets = ingestion.initiate_data_ingestion()

    logging.info(
        "Loaded Versions: %s",
        list(datasets.keys())
    )