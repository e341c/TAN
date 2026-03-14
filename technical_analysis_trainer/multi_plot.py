import argparse
import os
from typing import List

import pandas as pd
import plotly.express as px

from stock_data import DEFAULT_INTERVAL, DEFAULT_PERIOD, DEFAULT_TICKER, plot_multi_tickers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Отображение нескольких инструментов на одном графике. "
            "По умолчанию включает нефть Brent (BZ=F)."
        )
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=(
            "Список тикеров через пробел (данные тянутся с Yahoo Finance). "
            "По умолчанию, если не указано ни --tickers, ни --files, берётся только нефть Brent (BZ=F)."
        ),
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
        "--files",
        nargs="+",
        help=(
            "Пути к локальным CSV-файлам с колонками вида "
            "ts,open,high,low,close,... (как btcusdt_1h.csv и stock.csv). "
            "Будет построен общий график по выбранной колонке."
        ),
    )
    parser.add_argument(
        "--column",
        type=str,
        default="close",
        help="Имя числовой колонки для сравнения из CSV (по умолчанию 'close').",
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
        help=(
            "Нормировать ряды к 100 в начальной точке (по умолчанию True). "
            "Удобно для сравнения динамики."
        ),
    )
    return parser.parse_args()


def _plot_from_csv_files(
    files: List[str],
    value_column: str = "close",
    normalize: bool = True,
) -> None:
    """
    Строит несколько рядов из локальных CSV, накладывая их друг на друга.

    Ожидаемый формат CSV: колонка 'ts' с датой/временем и числовая колонка value_column
    (например, 'close' или 'funding_rate').
    """
    if not files:
        raise ValueError("Не переданы файлы для сравнения.")

    series_dict: dict[str, pd.Series] = {}

    for path in files:
        df = pd.read_csv(path)

        if "ts" not in df.columns:
            raise ValueError(f"В файле {path!r} нет колонки 'ts'.")
        if value_column not in df.columns:
            raise ValueError(f"В файле {path!r} нет колонки {value_column!r}.")

        # Приводим время к единому формату: UTC без таймзоны
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.assign(ts=ts).dropna(subset=["ts"])
        df["ts"] = df["ts"].dt.tz_convert(None)
        s = df.set_index("ts")[value_column].astype(float).sort_index()

        if s.empty:
            continue

        if normalize:
            s = s / s.iloc[0] * 100.0

        label = os.path.splitext(os.path.basename(path))[0]
        series_dict[label] = s

    if not series_dict:
        raise RuntimeError("Не удалось получить ни одной временной серии из файлов.")

    # Объединяем серии в один DataFrame для Plotly
    frames: list[pd.DataFrame] = []
    for label, s in series_dict.items():
        frames.append(
            pd.DataFrame(
                {
                    "ts": s.index,
                    "value": s.values,
                    "series": label,
                }
            )
        )

    df_plot = pd.concat(frames, ignore_index=True)

    y_label = (
        f"{value_column} (нормировано, старт = 100)" if normalize else value_column
    )
    title = (
        f"Сравнение {value_column} из файлов (нормировано)"
        if normalize
        else f"Сравнение {value_column} из файлов"
    )

    fig = px.line(
        df_plot,
        x="ts",
        y="value",
        color="series",
        title=title,
        labels={"ts": "Дата", "value": y_label, "series": "Серия"},
    )
    fig.update_layout(legend_title_text="Серия")
    fig.show()


def main() -> None:
    args = _parse_args()
    normalize = bool(args.normalize)

    # 1) Если переданы локальные CSV-файлы — сравниваем их
    if args.files:
        _plot_from_csv_files(
            files=args.files,
            value_column=args.column,
            normalize=normalize,
        )
        return

    # 2) Иначе работаем через тикеры Yahoo Finance
    tickers: List[str] = args.tickers or [DEFAULT_TICKER]

    plot_multi_tickers(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        normalize=normalize,
    )


if __name__ == "__main__":
    main()

