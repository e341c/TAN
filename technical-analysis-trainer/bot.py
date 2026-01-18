import asyncio
import json
from telegram import Bot
from datetime import datetime

from config import BOT_TOKEN, CHAT_ID
from real_time_predictions import getPreds


def get_data():
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Bot is alive",
    }


def build_message(data: dict) -> str:
    return "⏰ Hourly report\n" f"Time: {data['time']}\n" f"Status: {data['status']}"


async def send_message():
    bot = Bot(token=BOT_TOKEN)

    data = await getPreds()
    print("Data", data)

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
"""

    print(message)

    await bot.send_message(chat_id=CHAT_ID, text=message) # type: ignore


if __name__ == "__main__":
    asyncio.run(send_message())
