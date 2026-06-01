import os
import pandas as pd
from pandas_datareader.data import DataReader

os.environ["FRED_API_KEY"] = "da9cfb903e0572be8654d1d4173cf174"  # или задайте в окружении заранее

def save_fred(series_id: str, out_path: str, start="2018-01-01"):
    s = DataReader(series_id, "fred", start=start)[series_id]
    df = s.dropna().reset_index()
    df.columns = ["ts", "close"]
    # важно: привести к тому же виду, что ваши Binance CSV (UTC timestamp)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.to_csv(out_path, index=False)
    print(series_id, "->", out_path, "rows:", len(df))

save_fred("DTWEXBGS", "technical_analysis_trainer/data/macro_dtwexbgs_1d.csv")
save_fred("VIXCLS", "technical_analysis_trainer/data/macro_vixcls_1d.csv")