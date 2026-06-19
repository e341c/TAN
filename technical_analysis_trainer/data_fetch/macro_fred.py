"""Fetch macro series from FRED into CSV."""
import os

import pandas as pd
from pandas_datareader.data import DataReader


def save_fred(series_id: str, out_path: str, start: str = "2018-01-01") -> None:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        os.environ["FRED_API_KEY"] = key
    s = DataReader(series_id, "fred", start=start)[series_id]
    df = s.dropna().reset_index()
    df.columns = ["ts", "close"]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.to_csv(out_path, index=False)
    print(series_id, "->", out_path, "rows:", len(df))


if __name__ == "__main__":
    save_fred("DTWEXBGS", "technical_analysis_trainer/data/macro_dtwexbgs_1d.csv")
    save_fred("VIXCLS", "technical_analysis_trainer/data/macro_vixcls_1d.csv")
