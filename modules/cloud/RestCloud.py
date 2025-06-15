import json
import logging
import os
import tempfile
import time
import zipfile
from abc import ABC
from pathlib import Path

import requests

from modules.cloud.BaseCloud import BaseCloud
from modules.util.TrainProgress import TrainProgress
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from remote_srv import TaskState

UPDATE_INTERVAL_SECS = 10


class RestCloud(BaseCloud, ABC):
    def __init__(self, config: TrainConfig, callback: TrainCallbacks = None):
        super().__init__(config)
        self.last_alive_status = None
        self.config = config
        self.callback = callback
        self.last_status = TaskState.QUEUED
        self.task_id = None

    def run_trainer(self):
        if not self.can_reattach():
            self.task_id = RestCloud.request_training(self._url("train"), self.config)
            if not self.task_id:
                raise Exception("Failed to start training.")
            logging.info(f"Training started with task ID: {self.task_id}")
        else:
            logging.info(f"Attaching to existing task ID: {self.task_id}")

        # Wait until status is "finished" or "failed"
        while True:
            if self.last_status in [TaskState.FINISHED, TaskState.ERROR]:
                logging.info(f"Training finished with status: {self.last_status}")
                break
            time.sleep(UPDATE_INTERVAL_SECS)

    def _url(self, endpoint: str, argument=None) -> str:
        # build url by combining the connection info from secrets file
        # secrets.host, port=secrets.port, user=secrets.user
        url = f"{self.config.secrets.cloud.host}:{self.config.secrets.cloud.port}/{endpoint}"
        if argument:
            url += f"/{argument}"
        return url

    @staticmethod
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

        logging.info(f"Packed concept {local} into a zip file of size {len(data) / (1024 * 1024):.2f} MB")
        return data

    @staticmethod
    def request_training(url, train_config: TrainConfig):
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
            logging.info(f"Bundling concept {c.name}...")
            binary_data = RestCloud.pack_concept(Path(c.local_path), recursive=c.include_subdirectories)
            concept_data.append(
                ('concepts', (os.path.basename(c.name), binary_data, 'application/zip')))
        data.extend(concept_data)

        logging.info(f"Uploading concept data and requesting training with config.")
        response = requests.post(url, files=data)
        if response.status_code == 202:
            logging.info("Training started successfully.")
            response_json = response.json()
            return response_json["task_id"]
        else:
            logging.error(f"Could not start training: {response}")
            return None

    def get_tensorboard_data(self):
        response = requests.get(self._url("tensorboard", self.task_id))
        if response.status_code == 200:
            # Server should send a zip file with tensorboard data
            # Write the response content to a temporary file
            target_dir = os.path.join(os.getcwd(), self.config.local_workspace_dir, "tensorboard")
            temp_path = Path(target_dir) / "tensorboard_data.zip"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            # Unzip the data
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        else:
            logging.warning(f"Error getting tensorboard data: {response}")
            return None

    def stop(self):
        response = requests.post(self._url("stop", self.task_id))
        if response.status_code != 200:
            raise Exception(f"Error stopping training: {response}")

    def delete_workspace(self):
        response = requests.post(self._url("delete_workspace", self.task_id))
        if response.status_code != 200:
            raise Exception(f"Error deleting workspace: {response}")

    def send_commands(self, commands: TrainCommands):
        if commands.get_stop_command():
            logging.info("Sending stop command to remote server.")
            self.stop()
            return

        for entry in commands.get_and_reset_sample_custom_commands():
            logging.info(f"Sending custom sample command: {entry}")
            response = requests.post(self._url("sample_custom", self.task_id), json=entry.to_dict())
            if response.status_code != 200:
                logging.error(f"Error sending custom sample command: {response}")

        if commands.get_and_reset_sample_default_command():
            logging.info("Sending default sample command.")
            response = requests.post(self._url("sample_default", self.task_id))
            if response.status_code != 200:
                logging.error(f"Error sending default sample command: {response}")

        if commands.get_and_reset_backup_command():
            logging.info("Sending backup command.")
            response = requests.post(self._url("backup", self.task_id))
            if response.status_code != 200:
                logging.error(f"Error sending backup command: {response}")

        if commands.get_and_reset_save_command():
            logging.info("Sending save command.")
            response = requests.post(self._url("save", self.task_id))
            if response.status_code != 200:
                logging.error(f"Error sending save command: {response}")


    def download_output_model(self):
        logging.info("Downloading output model...")
        response = requests.get(self._url("model", self.task_id))
        if response.status_code == 200:
            if not self.config.output_model_destination:
                logging.error("No output model destination specified in config.")

            filename = self.config.output_model_destination
            # remove the path and keep only the filename (keep the extension)
            filename = os.path.basename(filename)

            target_path = os.path.join(os.getcwd(), "models", filename)
            with open(target_path, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Error downloading model: {response}")

    def _connect(self):
        pass

    def setup(self):
        self.last_status = TaskState.INITIALIZING
        self.task_id = self.config.cloud.run_id if self.config.cloud.run_id else None

    def _install_onetrainer(self, update: bool = False):
        # We do not need to install onetrainer
        pass

    def _make_tensorboard_tunnel(self):
        # We do not need to make a tunnel for tensorboard, as tensorboard data is downloaded directly
        pass

    def upload_config(self, commands: TrainCommands = None):
        # We do not need to upload the config file, as it is already included in the request_training function
        pass

    def _upload_config_file(self, local: Path):
        # We do not need to upload the config file, as it is already included in the request_training function
        pass

    def can_reattach(self) -> bool:
        if self.task_id is None:
            return False
        return self.get_update() != TaskState.UNKNOWN

    def sync_workspace(self):
        if self.last_status is None or self.last_status in (TaskState.QUEUED, TaskState.INITIALIZING):
            return
        try:
            # Get the latest tensorboard data and overwrite the old one
            self.get_tensorboard_data()
            # Get the latest file updates (samples, saves, backups)
            self.get_file_updates()
        except Exception as e:
            logging.error(f"Error while syncing workspace: {e}")

    def get_file_updates(self):
        if self.config.cloud.download_samples:
            self.get_file_type_update("samples")
        if self.config.cloud.download_saves:
            self.get_file_type_update("save")
        if self.config.cloud.download_backups:
            self.get_file_type_update("backup")

    def get_file_type_update(self, file_type: str):
        # collect local files for the given file type
        local_files = []
        target_dir = os.path.join(os.getcwd(), self.config.local_workspace_dir, file_type)
        if not os.path.exists(target_dir):
            # create the directory if it does not exist
            os.makedirs(target_dir, exist_ok=True)

        for root, dirs, files in os.walk(target_dir):
            for file in files:
                # file is the relative path + file name but without the root path
                file_path = os.path.join(root, file)
                # convert to relative path
                relative_path = os.path.relpath(file_path, target_dir)
                local_files.append(relative_path)

        response = requests.get(self._url(f"files/list/{file_type}", self.task_id))
        if response.status_code == 200:
            # Response should contain a list of files
            # Check for updates and download new files
            remote_files = response.json()
            logging.debug(f"Remote {file_type} files: {remote_files}")
            for remote_file in remote_files:
                if remote_file not in local_files:
                    logging.info(f"Downloading new {file_type} file: {remote_file}")
                    self.download_file(file_type, remote_file)
        else:
            logging.error(f"Error getting {file_type} file list: {response.status_code} - {response.text}")

    def download_file(self, file_type: str, remote_file: str):
        """
        Download a file from the remote server.
        :param file_type: Type of the file (e.g., 'samples', 'save', 'backup').
        :param remote_file: Name of the remote file to download.
        """
        url = self._url(f"files/download/{file_type}/{self.task_id}")
        params = {'filename': remote_file}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            target_dir = os.path.join(os.getcwd(), self.config.local_workspace_dir, file_type)
            local_path = os.path.join(target_dir, remote_file)

            with open(local_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded {file_type} file: {remote_file}")
        else:
            logging.error(f"Error downloading {file_type} file {remote_file}: {response.status_code} - {response.text}")

    def exec_callback(self, callbacks: TrainCallbacks):
        """
        This method is called periodically to check the training status and progress.
        """
        if self.task_id is None:
            return
        # We do not want to spam the server with requests, so we only check the status every UPDATE_INTERVAL_SECS seconds
        # return if last update is less than UPDATE_INTERVAL_SECS seconds ago
        if (self.last_alive_status is not None
                and time.time() - self.last_alive_status < UPDATE_INTERVAL_SECS):
            return

        self.last_status = self.get_update()
        if self.last_status not in [TaskState.ERROR, TaskState.UNKNOWN]:
            self.last_alive_status = time.time()
        elif self.last_status == TaskState.ERROR:
            raise Exception("An error occurred during training. Please check the server logs for more information.")
        elif self.last_status == TaskState.UNKNOWN:
            logging.warning("Received unknown status from server, retrying...")
            # if last alive status is more than 60 seconds ago, we assume the server is down
            if self.last_alive_status is not None and time.time() - self.last_alive_status > 60:
                raise Exception("Server is down or task ID is invalid. Please check the server status.")


    def get_update(self):
        response = requests.get(self._url("status", self.task_id))
        if response.status_code == 200:
            data = response.json()

            logging.debug(f"RestCloud: Received status update: {data}")
            # Assuming the response contains a field "status" with the training status
            self.callback.on_update_status(data["status"])
            # Read training progress
            if "progress" in data:
                progress = data["progress"]
                self.callback.on_update_train_progress(TrainProgress(
                    progress["epoch"], progress["epoch_step"], progress["epoch_sample"],
                    progress["global_step"]),
                    progress['max_sample'],
                    progress['max_epoch'],
                )
            return data["rest_status"]
        else:
            logging.error(f"Error getting status: {response}")
        return "unknown"

    def close(self):
        pass
