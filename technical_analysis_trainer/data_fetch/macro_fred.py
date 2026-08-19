"""
Fetch macro series from FRED into the CSVs the trend model reads.

These feed `xmkt_log_ret_dxy` and `xmkt_log_ret_risk_proxy`. When the files go
stale the features do not error — `attach_cross_market_features` forward-fills
the last known value, so they silently flatten to a constant and the model is
served a frozen input it never saw during training. Refresh them on the same
schedule as the price data.

Usage:
    python -m technical_analysis_trainer.data_fetch.macro_fred            # train + predict
    python -m technical_analysis_trainer.data_fetch.macro_fred --check    # report staleness only
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd
from pandas_datareader.data import DataReader

# series id -> (basename used by the trend config, tolerated staleness in days)
#
# The two series publish on different cadences: VIXCLS is daily, DTWEXBGS is
# weekly with a release lag. A single threshold either cries wolf on the dollar
# index or lets a genuinely dead VIX feed slide for a week.
SERIES = {
    "DTWEXBGS": ("macro_dtwexbgs_1d", 14),  # broad dollar index -> xmkt_log_ret_dxy
    "VIXCLS": ("macro_vixcls_1d", 5),       # volatility index   -> xmkt_log_ret_risk_proxy
}

TRAIN_DIR = "technical_analysis_trainer/data/train"
PREDICT_DIR = "technical_analysis_trainer/data/predict"
# Inference only needs recent history; training wants the full series.
PREDICT_TAIL_DAYS = 400


def fetch_series(series_id: str, start: str = "2018-01-01") -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        os.environ["FRED_API_KEY"] = key
    s = DataReader(series_id, "fred", start=start)[series_id]
    df = s.dropna().reset_index()
    df.columns = ["ts", "close"]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def save_series(series_id: str, basename: str, start: str = "2018-01-01") -> dict:
    df = fetch_series(series_id, start=start)

    train_path = os.path.join(TRAIN_DIR, f"{basename}.csv")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    df.to_csv(train_path, index=False)

    predict_path = os.path.join(PREDICT_DIR, f"{basename}_predict.csv")
    os.makedirs(PREDICT_DIR, exist_ok=True)
    cutoff = df["ts"].max() - pd.Timedelta(days=PREDICT_TAIL_DAYS)
    df[df["ts"] >= cutoff].to_csv(predict_path, index=False)

    last = df["ts"].max()
    age_days = (pd.Timestamp.now(tz="UTC") - last).days
    print(
        f"{series_id:10} rows={len(df):5}  last={last.date()}  age={age_days}d"
        f"  -> {train_path}, {predict_path}"
    )
    return {"series": series_id, "rows": len(df), "last": last, "age_days": age_days}


def check_staleness(max_age_days: int | None = None) -> int:
    """Report how old the on-disk files are. Returns a non-zero count if stale."""
    stale = 0
    now = datetime.now(timezone.utc)
    for basename, default_max in SERIES.values():
        limit = max_age_days if max_age_days is not None else default_max
        for path in (
            os.path.join(TRAIN_DIR, f"{basename}.csv"),
            os.path.join(PREDICT_DIR, f"{basename}_predict.csv"),
        ):
            if not os.path.isfile(path):
                print(f"MISSING  {path}")
                stale += 1
                continue
            df = pd.read_csv(path)
            last = pd.to_datetime(df["ts"], utc=True).max()
            age = (pd.Timestamp(now) - last).days
            flag = "STALE  " if age > limit else "ok     "
            print(f"{flag} {path}  last={last.date()} age={age}d (limit {limit}d)")
            if age > limit:
                stale += 1
    return stale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="report staleness, do not fetch")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument(
        "--max-age-days",
        type=int,
        default=None,
        help="override the per-series staleness limit for every series",
    )
    a = p.parse_args()

    if a.check:
        stale = check_staleness(a.max_age_days)
        print(f"\n{stale} stale/missing file(s)")
        raise SystemExit(1 if stale else 0)

    for series_id, (basename, _) in SERIES.items():
        save_series(series_id, basename, start=a.start)


if __name__ == "__main__":
    main()
