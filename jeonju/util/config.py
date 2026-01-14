import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE = os.getenv("DATABASE")
COLLECTION = os.getenv("COLLECTION")

HCX_HOST=os.getenv("HCX_HOST")
HCX_HOST_GW=os.getenv("HCX_HOST_GW")

HCX_HOST_DASH=os.getenv("HCX_HOST_DASH")
HCX_HOST_DASH_GW=os.getenv("HCX_HOST_DASH_GW")

HCX_API_KEY=os.getenv("HCX_API_KEY")
HCX_APIGW_KEY=os.getenv("HCX_APIGW_KEY")
HCX_REQUEST_ID=os.getenv("HCX_REQUEST_ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")