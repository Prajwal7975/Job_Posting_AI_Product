from dataclasses import dataclass
import os
from typing import Dict, Tuple
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataIngestionConfig:


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
