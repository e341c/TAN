# py-trainer/trainer_api.py
from flask import Flask, jsonify
import threading, os

app = Flask(__name__)
status = {"running": False, "last_version": None} # type: ignore

@app.route("/train", methods=["POST"])
def train():
    if status["running"]:
        return jsonify({"error": "Training already running"}), 409

    status["running"] = True
    thread = threading.Thread(target=_run_train)
    thread.start()
    return jsonify({"status": "started"}), 200

def _run_train():
    os.system("python train.py")
    status["running"] = False

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(status)

@app.route("/latest", methods=["GET"])
def latest_model():
    model_dir = "models"
    versions = sorted(os.listdir(model_dir))
    if not versions:
        return jsonify({"error": "no models"}), 404
    latest = versions[-1]
    status["last_version"] = latest
    return jsonify({"version": latest})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)