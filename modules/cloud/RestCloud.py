import json
import os
import tempfile
import zipfile
from abc import ABC
from pathlib import Path

import requests

from modules.cloud.BaseCloud import BaseCloud
from modules.util.TrainProgress import TrainProgress
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.time_util import get_string_timestamp


def pack_concept(local: Path, recursive: bool):
    """
    create a compressed backup of the dataset
    ask the system for the temp directory and use that as the target directory
    """
    file = tempfile.TemporaryFile(mode="w+b", buffering=-1)

    # convert posix path to string
    with zipfile.ZipFile(file, 'w') as zipF:
        # iterate recursively over the directory
        for root, dirs, files in os.walk(local):
            # skip subdirectories if parameter subdirectories is False
            if not recursive and root != str(local):
                continue
            for file_name in files:
                # create the full path to the file
                file_path = os.path.join(root, file_name)
                # add the file to the zip file
                zipF.write(file_path, os.path.relpath(file_path, local))

    # Read the file at once and store it in memory, then remove the temporary file
    file.seek(0)
    data = file.read()
    file.close()
    return data


def request_training(url, train_config: TrainConfig) -> str:
    """
    Request training on a remote server via REST API
    Returns the task ID if successful, otherwise None
    """
    data = [
        ("config", ("config", json.dumps(train_config.to_dict()), 'application/json')),
        ("secrets", ("secrets", json.dumps(train_config.secrets.to_dict()), 'application/json')),
    ]

    # Compress concept files and add them to the request
    concept_data = []
    for c in train_config.concepts:
        if not c.enabled:
            continue
        print(f"Bundling concept {c.name}...")
        binary_data = pack_concept(Path(c.path), recursive=c.include_subdirectories)
        concept_data.append(
            ('concepts', (os.path.basename(c.name), binary_data, 'application/zip')))
    data.extend(concept_data)

    response = requests.post(url, files=data)
    if response.status_code == 202:
        print("Training started successfully.")
        return response.json()["task_id"]
    else:
        print(f"Error starting training: {response}")


class RestCloud(BaseCloud, ABC):
    def __init__(self, config: TrainConfig, reattach: bool = False):
        super().__init__(config)
        self.connection = None
        self.callback_connection = None
        self.command_connection = None
        self.tensorboard_tunnel_stop = None
        self.config = config
        self.reattach = reattach

        name = config.cloud.run_id if config.cloud.detach_trainer else get_string_timestamp()
        self.task_id = None
        # self.callback_file = f'{config.cloud.remote_dir}/{name}.callback'
        # self.command_pipe = f'{config.cloud.remote_dir}/{name}.command'
        # self.config_file = f'{config.cloud.remote_dir}/{name}.json'
        # self.exit_status_file = f'{config.cloud.remote_dir}/{name}.exit'
        # self.log_file = f'{config.cloud.remote_dir}/{name}.log'
        # self.pid_file = f'{config.cloud.remote_dir}/{name}.pid'

    def _url(self, endpoint: str, argument=None) -> str:
        # build url by combining the connection info from secrets file
        # secrets.host, port=secrets.port, user=secrets.user
        url = f"http://{self.config.secrets.cloud.host}:{self.config.secrets.cloud.port}/{endpoint}"
        if argument:
            url += f"/{argument}"
        return url

    def run_trainer(self):
        self.task_id = request_training(self._url("train"), self.config)
        if not self.task_id:
            raise Exception("Failed to start training.")

    def stop(self):
        response = requests.post(self._url("stop", self.task_id))
        if response.status_code != 200:
            raise Exception(f"Error stopping training: {response}")

    def delete_workspace(self):
        response = requests.post(self._url("delete_workspace", self.task_id))
        if response.status_code != 200:
            raise Exception(f"Error deleting workspace: {response}")

    def send_commands(self, commands: TrainCommands):
        print("Sending commands to remote server...")
        print(commands)
        # TODO

    def download_output_model(self):
        print("Downloading output model...")
        response = requests.get(self._url("model", self.task_id))
        if response.status_code == 200:
            with open(self.config.output_model_destination, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Error downloading model: {response}")

    def _connect(self):
        pass

    def _install_onetrainer(self, update: bool = False):
        raise NotImplementedError("Onetrainer installation not supported on this cloud type")

    def _make_tensorboard_tunnel(self):
        raise NotImplementedError("Tensorboard tunnel not supported on this cloud type")

    def upload_config(self, commands: TrainCommands = None):
        # We do not need to upload the config file, as it is already included in the request_training function
        pass

    def _upload_config_file(self, local: Path):
        # We do not need to upload the config file, as it is already included in the request_training function
        pass

    def can_reattach(self) -> bool:
        return False

    def sync_workspace(self):
        pass

    def exec_callback(self, callbacks: TrainCallbacks):
        response = requests.get(self._url("status", self.task_id))
        if response.status_code == 200:
            data = response.json()
            # Assuming the response contains a field "status" with the training status
            callbacks.on_update_status(data["status"])

            # Read training progress
            if "progress" in data:
                progress = data["progress"]
                max_sample = 1  # TODO
                callbacks.on_update_train_progress(TrainProgress(
                    progress["epoch"], progress["epoch_step"], progress["epoch_sample"], progress["global_step"]),
                    max_sample,
                    self.config.epochs,
                )
        else:
            print(f"Error getting status: {response}")


    def close(self):
        pass
