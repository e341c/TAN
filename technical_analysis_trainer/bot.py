import asyncio
import json
import yaml, os
from telegram import Bot

# from bot_config import BOT_TOKEN, CHAT_ID
from technical_analysis_trainer.real_time_predictions import getPreds

BASE_DIR = os.path.dirname(__file__)
cfg_path = os.getenv("CONFIG_PATH", os.path.join(BASE_DIR, "config.yaml"))

# cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)


BOT_TOKEN = cfg["BOT_TOKEN"]
CHAT_ID = cfg["CHAT_ID"]


def build_message(data: dict) -> str:
    return "⏰ Hourly report\n" f"Time: {data['time']}\n" f"Status: {data['status']}"


async def send_message():
    bot = Bot(token=BOT_TOKEN)

    data, trendData = await getPreds()
    print("Data", data)
    print("Trend Data", trendData)

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

{trendData}
"""

    print(message)

    await bot.send_message(chat_id=CHAT_ID, text=message)  # type: ignore


if __name__ == "__main__":
    asyncio.run(send_message())
