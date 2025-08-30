from FiinQuantX import FiinSession
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='../config/.env')

# Truy cập các biến môi trường
FIINQUANT_USERNAME = os.getenv("FIINQUANT_USERNAME")
FIINQUANT_PASSWORD = os.getenv("FIINQUANT_PASSWORD")

client = FiinSession(
    username=FIINQUANT_USERNAME,
    password=FIINQUANT_PASSWORD,
).login()

fi = client.FiinIndicator()
# Lấy dữ liệu có sẵn
tickers = ['HPG','SSI','VN30']
df = client.Fetch_Trading_Data(
    realtime = False,
    tickers = tickers,
    fields = ['open','high','low','close','volume','bu','sd','fn','fs','fb'],
    adjusted=True,
    from_date='2024-12-04',
    to_date = '1m'
    ).get_data()

print(df)
