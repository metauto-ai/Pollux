import io
import os
import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Final
from tqdm import tqdm

load_dotenv()

MONGODB_URI: Final[str] = os.environ["MONGODB_URI"]
MONGODB_USER: Final[str] = os.environ["MONGODB_USER"]
MONGODB_PASSWORD: Final[str] = os.environ["MONGODB_PASSWORD"]
encoded_user = quote_plus(MONGODB_USER)
encoded_password = quote_plus(MONGODB_PASSWORD)
MONGODB_URI = f"mongodb+srv://{encoded_user}:{encoded_password}@{MONGODB_URI}"
