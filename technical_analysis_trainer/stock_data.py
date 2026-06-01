import argparse
import os
from typing import List

import matplotlib.pyplot as plt  # оставим как запасной вариант
import pandas as pd
import plotly.express as px
import yfinance as yf


# По умолчанию берём нефть Brent через тикер Yahoo Finance
DEFAULT_TICKER = "BZ=F"
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1h"


def fetch_stock_data(
    ticker: str = DEFAULT_TICKER,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Загружает данные по акции/товару/индексу с Yahoo Finance.

    Возвращает DataFrame с колонками:
    ts, open, high, low, close, adj_close, volume.
    """
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(f"Не удалось получить данные для тикера {ticker!r}")

    # Приводим возможный MultiIndex колонок к обычным строкам
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

    data = data.reset_index()

    # В разных режимах индекс может называться "Date" или "Datetime"
    rename_map: dict[str, str] = {}
    if "Date" in data.columns:
        rename_map["Date"] = "ts"
    if "Datetime" in data.columns:
        rename_map["Datetime"] = "ts"

    rename_map.update(
        {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    data = data.rename(columns=rename_map)

    return data[["ts", "open", "high", "low", "close", "adj_close", "volume"]]


def plot_single_ticker(
    ticker: str = DEFAULT_TICKER,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> None:
    """
    Строит график цены закрытия для одного инструмента.
    """
    df = fetch_stock_data(ticker=ticker, period=period, interval=interval)

    fig = px.line(
        df,
        x="ts",
        y="close",
        title=f"Цена {ticker}",
        labels={"ts": "Дата", "close": "Цена закрытия"},
    )
    fig.update_layout(legend_title_text="Инструмент")
    fig.show()


def plot_multi_tickers(
    tickers: List[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    normalize: bool = False,
) -> None:
    """
    Строит несколько графиков на одном полотне.

    Если normalize=True, все ряды нормируются к 100 в первой точке.
    """
    if not tickers:
        raise ValueError("Список тикеров пуст.")

    series_dict: dict[str, pd.Series] = {}

    for ticker in tickers:
        df = fetch_stock_data(ticker=ticker, period=period, interval=interval)
        s = df.set_index("ts")["close"].sort_index()
        if s.empty:
            continue
        if normalize:
            s = s / s.iloc[0] * 100.0
        series_dict[ticker] = s

    if not series_dict:
        raise RuntimeError("Не удалось получить данные ни по одному тикеру.")

    # Объединяем ряды в один DataFrame для Plotly
    frames: list[pd.DataFrame] = []
    for ticker, s in series_dict.items():
        frames.append(
            pd.DataFrame(
                {
                    "ts": s.index,
                    "value": s.values,
                    "ticker": ticker,
                }
            )
        )

    df_plot = pd.concat(frames, ignore_index=True)

    y_label = (
        "Нормированная цена (первая точка = 100)" if normalize else "Цена закрытия"
    )
    title = (
        "Сравнение динамики (нормировано к 100 в начальный момент)"
        if normalize
        else "Сравнение динамики цен"
    )

    fig = px.line(
        df_plot,
        x="ts",
        y="value",
        color="ticker",
        title=title,
        labels={"ts": "Дата", "value": y_label, "ticker": "Инструмент"},
    )
    fig.update_layout(legend_title_text="Инструмент")
    fig.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Загрузка и отображение рыночных данных с Yahoo Finance. "
            "По умолчанию — график цен на нефть Brent (BZ=F)."
        )
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=DEFAULT_TICKER,
        help="Тикер инструмента (по умолчанию BZ=F — нефть Brent).",
    )
    parser.add_argument(
        "--period",
        type=str,
        default=DEFAULT_PERIOD,
        help="Период истории, напр. 1y, 6mo, 5y, max (по умолчанию 1y).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=DEFAULT_INTERVAL,
        help="Таймфрейм, напр. 1d, 1h, 1wk, 1mo (по умолчанию 1d).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Путь к CSV для сохранения данных (опционально).",
    )
    parser.add_argument(
        "--multi",
        nargs="+",
        help=(
            "Список тикеров для совместного графика. "
            "Если указан, игнорируется --ticker и строится общий график."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Нормировать несколько рядов к 100 в начальной точке.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Не показывать окно с графиком, только сохранить CSV (если указан --output).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Режим нескольких тикеров (наложенные графики)
    if args.multi:
        if args.output:
            # В этом режиме сохраняем только первый тикер,
            # чтобы не плодить много файлов без необходимости.
            df_first = fetch_stock_data(
                ticker=args.multi[0],
                period=args.period,
                interval=args.interval,
            )
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            df_first.to_csv(args.output, index=False)
            print(f"Сохранено {len(df_first)} строк в {args.output}")

        if not args.no_show:
            plot_multi_tickers(
                tickers=args.multi,
                period=args.period,
                interval=args.interval,
                normalize=args.normalize,
            )
        return

    # Одиночный тикер (по умолчанию — нефть Brent)
    df = fetch_stock_data(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
    )

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Сохранено {len(df)} строк в {args.output}")

    if not args.no_show:
        plot_single_ticker(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
        )


if __name__ == "__main__":
    main()
