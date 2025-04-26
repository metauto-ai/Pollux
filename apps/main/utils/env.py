import os
import logging
from urllib.parse import quote_plus
from dotenv import load_dotenv
import wandb


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        load_dotenv()
        self.MONGODB_USER = os.environ["MONGODB_USER"]
        self.MONGODB_PASSWORD = os.environ["MONGODB_PASSWORD"]
        self.MONGODB_HOST = os.environ["MONGODB_URI"]
        self.S3KEY = os.environ["S3KEY"]
        self.S3SECRET = os.environ["S3SECRET"]
        self.LOCAL_TEMP_DIR = "/dev/shm"

        # Encode credentials for MongoDB URI
        encoded_user = quote_plus(self.MONGODB_USER)
        encoded_password = self.MONGODB_PASSWORD
        self.MONGODB_URI = (
            f"mongodb+srv://{encoded_user}:{encoded_password}@{self.MONGODB_HOST}"
        )

        # Initialize Weights & Biases (WANDB) once
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            logging.info("WANDB_API_KEY found in environment variables")
        else:
            logging.warning("WANDB_API_KEY not found in environment variables")


# Singleton instance
env = Config()
