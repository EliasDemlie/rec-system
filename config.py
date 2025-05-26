import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
SECRET_KEY = os.getenv("SECRET_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")

PRODUCT_DB_NAME = os.getenv("PRODUCT_DB_NAME")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")