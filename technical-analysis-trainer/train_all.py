# py-trainer/train_all.py
import yaml
import subprocess
import os

# --- 1️⃣ Загрузка базового конфигурационного файла ---
with open("config.yaml") as f:
    base_cfg = yaml.safe_load(f)

symbol = base_cfg["symbol"]
base_output_dir = base_cfg["output_dir"]
os.makedirs(base_output_dir, exist_ok=True)

# --- 2️⃣ Горизонты, которые нужно обучить ---
horizons = {
    "short": 1,
    "mid": 12,
    "long": 24,
}

windows = {
    "short": 6,
    "mid": 24,
    "long": 48,
}

# --- 3️⃣ Цикл по каждому горизонту ---
for name, horizon in horizons.items():
    print(f"\n🚀 Training {name.upper()} model (horizon={horizon})...\n")

    # создаём временный конфиг для конкретной модели
    cfg_path = f"config_{name}.yaml"
    cfg = base_cfg.copy()
    cfg["horizon"] = horizon
    cfg["window"] = windows[name]
    cfg["output_dir"] = os.path.join(base_output_dir, name)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # запускаем train.py с этим конфигом
    subprocess.run(["python", "train.py"], env={**os.environ, "CONFIG_PATH": cfg_path})

    # после завершения можно удалить временный конфиг
    os.remove(cfg_path)

print("\n✅ All models trained successfully!")
print(f"📂 Saved in: {base_output_dir}")