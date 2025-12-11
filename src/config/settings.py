# settings.py

# Configuration settings for the Drug Repurposing Assistant

API_KEYS = {
    "pubmed": "your_pubmed_api_key_here",
    "clinical_trials": "your_clinical_trials_api_key_here",
    "molecular_analysis": "your_molecular_analysis_api_key_here"
}

DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "your_db_user",
    "password": "your_db_password",
    "database": "drug_repurposing_db"
}

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": "drug_repurposing_assistant.log"
}

# Add any additional configuration settings as needed.