# Connect via REST to a flask server and request training
import json
import time

import requests

from modules.cloud.RestCloud import request_training
from modules.util.config.TrainConfig import TrainConfig

if __name__ == "__main__":
    config = TrainConfig.default_values()

    # read config via argparse
    import argparse
    parser = argparse.ArgumentParser(description="Remote training client for ComfyUI.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration file.")
    args = parser.parse_args()
    config_file = args.config

    with open(config_file, "r") as f:
        config_dict = json.load(f)
    config.from_dict(config_dict)

    task_id = request_training("http://localhost:5000/train", config)
    if not task_id:
        print("Failed to start training.")
        exit(1)

    # Repeat status request every 5 seconds unless the task is finished
    count = 0
    while True:
        if count > 10:
            print("Stopping training after 50 seconds")
            requests.post(f"http://localhost:5000/stop/{task_id}")
            break
        response = requests.get(f"http://localhost:5000/status/{task_id}")
        if response.status_code == 200:
            status = response.json()
            if status in ["finished", "failed"]:
                break
        else:
            print(f"Error getting status: {response}")
        time.sleep(5)
        count += 1
