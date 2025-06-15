# Remote training client for testing (FastAPI version)

import json
import time
import argparse
import requests

from modules.cloud.RestCloud import RestCloud
from modules.util.config.TrainConfig import TrainConfig

"""
For testing purposes only.
Connects to a FastAPI server and requests model training via REST.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote training client for ComfyUI.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration file.")
    args = parser.parse_args()

    config = TrainConfig.default_values()
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    config.from_dict(config_dict)

    cloud = RestCloud(config)
    base_url = "http://localhost:8000"
    task_id = cloud.request_training(f"{base_url}/train", config)

    if not task_id:
        print("❌ Failed to start training.")
        exit(1)

    print(f"✅ Training started with task_id: {task_id}")
    count = 0

    while True:
        if count > 10:
            print("🛑 Stopping training after 50 seconds")
            requests.post(f"{base_url}/stop/{task_id}")
            time.sleep(10)

            model_resp = requests.get(f"{base_url}/model/{task_id}")
            if model_resp.status_code == 200:
                model_path = f"{task_id}_model.safetensors"
                with open(model_path, "wb") as f:
                    f.write(model_resp.content)
                print(f"📦 Model saved to: {model_path}")
            else:
                print("❌ Failed to retrieve model:", model_resp.status_code, model_resp.text)
            break

        response = requests.get(f"{base_url}/status/{task_id}")
        if response.status_code == 200:
            status = response.json()
            print(f"[{count * 5}s] Status: {status}")
            if status.get("status") in ["finished", "error"]:
                break
        else:
            print(f"⚠️ Error getting status: {response.status_code} {response.text}")

        time.sleep(5)
        count += 1
