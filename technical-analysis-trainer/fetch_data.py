#!/usr/bin/env python3
from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import argparse
import time
import os

def fetch_data(symbol: str, interval: str, months: int, output_dir: str):
    client = Client()  # public API (ключи не требуются)

    # диапазон дат
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30 * months)

    print(f"📅 Fetching {months} months of {interval} data for {symbol}")
    print(f"   From {start_time:%Y-%m-%d} to {end_time:%Y-%m-%d}\n")

    klines_all = []
    chunk_start = start_time
    chunk_size_days = 5  # Binance = ~7200 свечей за 5 дней (1000 req limit)

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
        print("❌ No data fetched.")
        return

    # преобразование в DataFrame
    df = pd.DataFrame(klines_all, columns=[
        "ts","open","high","low","close","volume",
        "close_time","quote_asset_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df = df[["ts","open","high","low","close","volume"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float
    })

    # сохраняем
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{symbol.lower()}_{interval}.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {output_path} with {len(df)} rows ({len(df)/60/24:.1f} days)")

# === CLI интерфейс ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical candlestick data from Binance.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("--interval", type=str, default="1m", help="Candlestick interval, e.g. 1m, 5m, 1h, 1d")
    parser.add_argument("--months", type=int, default=6, help="Number of months of data to fetch")
    parser.add_argument("--output", type=str, default="./data", help="Output directory for CSV file")
    args = parser.parse_args()

    fetch_data(args.symbol, args.interval, args.months, args.output)