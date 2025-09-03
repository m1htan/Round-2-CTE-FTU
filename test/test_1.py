import inspect
import os
from FiinQuantX import FiinSession
from dotenv import load_dotenv

load_dotenv(dotenv_path='../config/.env')

USERNAME = os.getenv("FIINQUANT_USERNAME")
PASSWORD = os.getenv("FIINQUANT_PASSWORD")

client = FiinSession(username=USERNAME, password=PASSWORD).login()

print("Signature:", inspect.signature(client.FundamentalAnalysis().get_ratios))
