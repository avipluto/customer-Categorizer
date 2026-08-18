import os
import sys

import certifi
import pymongo
from dotenv import load_dotenv

from src.constant.database import DATABASE_NAME
from src.constant.env_variable import MONGODB_URL_KEY
from src.exception import CustomerException

# Ensure .env is loaded even if this module is imported before app.py's load_dotenv() runs
load_dotenv()

ca = certifi.where()


def _get_mongo_url() -> str:
    """
    Looks up the Mongo connection string using MONGODB_URL_KEY first,
    then falls back to a few common alternate names in case of a
    naming mismatch between env_variable.py and the actual .env file.
    """
    candidate_keys = [
        MONGODB_URL_KEY,
        "MONGO_DB_URL",
        "MONGODB_URL",
        "MONGO_URI",
        "MONGODB_URI",
    ]

    for key in candidate_keys:
        value = os.getenv(key)
        if value:
            return value

    raise Exception(
        f"Mongo connection string not found. Tried env keys: {candidate_keys}. "
        f"Please confirm the exact key name in your .env file matches one of these, "
        f"or update MONGODB_URL_KEY in src/constant/env_variable.py to match your .env file."
    )


class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = _get_mongo_url()
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise CustomerException(e, sys)