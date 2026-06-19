"""Telegram one-shot prediction message."""
from __future__ import annotations

import asyncio
import os

import yaml
from telegram import Bot

from technical_analysis_trainer.assembly.orchestrator import build_snapshot


def _telegram_config() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        return token, chat_id
    cfg_path = os.environ.get(
        "TELEGRAM_CONFIG",
        os.path.join(os.path.dirname(__file__), "..", "telegram.yaml.example"),
    )
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("BOT_TOKEN", "")), str(cfg.get("CHAT_ID", ""))
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID or TELEGRAM_CONFIG yaml")


async def send_message() -> None:
    bot_token, chat_id = _telegram_config()
    bot = Bot(token=bot_token)

    snapshot = await build_snapshot()
    data = snapshot.quantile_dict() or {}
    trend_data = snapshot.trend_dict() or {}

    down = trend_data.get("DOWN") or trend_data.get("probabilities", {}).get("DOWN", 0)
    up = trend_data.get("UP") or trend_data.get("probabilities", {}).get("UP", 0)

    message = f"""
📊 BTC Prediction

🕒 Time: {data['ts']:%Y-%m-%d %H:%M}

📉 Q10: {data['q10']:.2f}
📊 Q50 (median): {data['q50']:.2f}
📈 Q90: {data['q90']:.2f}

🔁 Log returns:
• {data['log_returns'][0]:+.5f}
• {data['log_returns'][1]:+.5f}
• {data['log_returns'][2]:+.5f}

Trend:
Down: {down}
Up: {up}

Predicted trend - {trend_data.get("predicted_trend")}
"""
    await bot.send_message(chat_id=chat_id, text=message.strip())


if __name__ == "__main__":
    asyncio.run(send_message())
