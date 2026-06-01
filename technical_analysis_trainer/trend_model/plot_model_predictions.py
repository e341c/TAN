"""
Отдельный entrypoint: тест выбранной модели + интерактивный график прогнозов.

Переиспользует пайплайн из test.py, но добавляет удобные опции:
  - сохранить HTML (--save-html)
  - не открывать браузер (--no-show)

Примеры:
  python -m technical_analysis_trainer.trend_model.plot_model_predictions --run 2026-04-18T20-55-59
  python -m technical_analysis_trainer.trend_model.plot_model_predictions --run 2026-04-18T20-55-59 --save-html artifacts/predictions.html --no-show
"""

from __future__ import annotations

import argparse
import os

from technical_analysis_trainer.trend_model.test import run_evaluation


def main() -> None:
    base_dir = os.path.dirname(__file__)
    default_cfg = os.path.join(base_dir, "test_config.yaml")

    ap = argparse.ArgumentParser(description="Plot model predictions for a selected ONNX run.")
    ap.add_argument(
        "--config",
        default=os.getenv("CONFIG_PATH", default_cfg),
        help="YAML как при обучении (data_path, window, horizon, threshold, …)",
    )
    ap.add_argument(
        "--run",
        default=os.getenv("MODEL_RUN", ""),
        metavar="TIMESTAMP",
        help="Подпапка в output_dir (например 2026-04-18T20-55-59). Пусто = последняя по mtime.",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Маркеры и бинарные метрики только при max(softmax) >= этого значения",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Температура softmax (argmax не меняется при T=1)",
    )
    ap.add_argument(
        "--save-html",
        default="",
        metavar="PATH",
        help="Если задано, сохранить график в HTML по этому пути (относительный путь — от корня репозитория).",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Не открывать график (полезно вместе с --save-html).",
    )
    args = ap.parse_args()

    fig = run_evaluation(
        cfg_path=args.config,
        run_id=args.run,
        min_confidence=float(args.min_confidence),
        temperature=float(args.temperature),
    )

    if args.save_html:
        # Plotly сам создаёт директории не всегда предсказуемо; пусть путь будет простой.
        # Относительный путь — от корня репозитория (как в config).
        out_path = args.save_html
        if not os.path.isabs(out_path):
            out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", out_path))
        fig.write_html(out_path)
        print(f"Saved: {out_path}")

    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()

