import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
# SECRET_KEY = os.getenv("SECRET_KEY")

PRODUCT_DB_NAME = os.getenv("PRODUCT_DB_NAME")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")