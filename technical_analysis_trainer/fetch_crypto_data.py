from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import argparse
import requests
import time
import os


def fetch_klines(symbol: str, interval: str, months: int) -> pd.DataFrame:
    client = Client()

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30 * months)

    print(f"📅 Fetching {months} months of {interval} klines for {symbol}")
    print(f"   From {start_time:%Y-%m-%d} to {end_time:%Y-%m-%d}\n")

    klines_all = []
    chunk_start = start_time
    chunk_size_days = 5

    while chunk_start < end_time:
        chunk_end = min(chunk_start + timedelta(days=chunk_size_days), end_time)

        try:
            klines = client.get_historical_klines(
                symbol,
                interval,
                chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception as e:
            print(f"⚠️ Error fetching {chunk_start:%Y-%m-%d} → {chunk_end:%Y-%m-%d}: {e}")
            time.sleep(1)
            continue

        if not klines:
            print("⚠️ No data returned, stopping.")
            break

        klines_all.extend(klines)
        chunk_start = chunk_end
        print(f"✅ Loaded up to {chunk_end:%Y-%m-%d}, total rows: {len(klines_all)}")
        time.sleep(0.25)

    if not klines_all:
        raise RuntimeError("❌ No kline data fetched.")

    df = pd.DataFrame(klines_all, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    return df


def fetch_funding_rate(symbol: str, months: int) -> pd.DataFrame:
    """
    Загружает funding rate с Binance Futures (выходит каждые 8 часов).
    Публичный endpoint, ключи не нужны.
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30 * months)

    print(f"\n📅 Fetching funding rate for {symbol}")

    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_rows = []
    chunk_start = start_time

    while chunk_start < end_time:
        params = {
            "symbol": symbol,
            "startTime": int(chunk_start.timestamp() * 1000),
            "endTime": int(min(chunk_start + timedelta(days=30), end_time).timestamp() * 1000),
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"⚠️ Error fetching funding rate: {e}")
            time.sleep(1)
            continue

        if not data:
            break

        all_rows.extend(data)
        last_ts = pd.to_datetime(data[-1]["fundingTime"], unit="ms")
        chunk_start = last_ts + timedelta(hours=8)
        print(f"✅ Funding rate loaded up to {last_ts:%Y-%m-%d}, total: {len(all_rows)}")
        time.sleep(0.25)

    if not all_rows:
        raise RuntimeError("❌ No funding rate data fetched.")

    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["ts", "funding_rate"]]
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    print(f"   Funding rate range: {df['funding_rate'].min():.6f} .. {df['funding_rate'].max():.6f}")
    return df


def merge_funding_rate(klines_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Мёрджит funding rate к свечам через forward fill.
    Funding rate выходит раз в 8H — все промежуточные свечи
    получают последнее известное значение (как в реальном трейдинге).
    """
    df = klines_df.copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # merge_asof — берёт ближайшее предыдущее значение funding rate для каждой свечи
    funding_df = funding_df.sort_values("ts")
    df = pd.merge_asof(df, funding_df, on="ts", direction="backward")

    # Первые свечи до первого funding rate останутся NaN — заполним нулём
    df["funding_rate"] = df["funding_rate"].fillna(0.0)

    return df


def fetch_data(symbol: str, interval: str, months: int, output_dir: str, with_funding: bool = True):
    klines_df = fetch_klines(symbol, interval, months)

    if with_funding:
        try:
            funding_df = fetch_funding_rate(symbol, months)
            df = merge_funding_rate(klines_df, funding_df)
            print(f"\n✅ Merged funding rate: {df['funding_rate'].notna().sum()} rows with data")
        except Exception as e:
            print(f"⚠️ Could not fetch funding rate: {e}. Saving klines only.")
            df = klines_df
    else:
        df = klines_df

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{symbol.lower()}_{interval}.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {output_path} with {len(df)} rows, columns: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical data from Binance.")
    parser.add_argument("--symbol",   type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--months",   type=int, default=6)
    parser.add_argument("--output",   type=str, default="./data")
    parser.add_argument("--no-funding", action="store_true", help="Skip funding rate fetch")
    parser.add_argument(
        "--also",
        nargs="*",
        default=[],
        metavar="SYMBOL",
        help="Additional symbols (klines only, no funding) for cross-market CSVs, e.g. ETHUSDT SOLUSDT",
    )
    args = parser.parse_args()

    fetch_data(args.symbol, args.interval, args.months, args.output, with_funding=not args.no_funding)
    for extra in args.also:
        fetch_data(extra, args.interval, args.months, args.output, with_funding=False)