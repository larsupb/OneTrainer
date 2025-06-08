# use flask to create a server that accepts requests for training
import io
import json
import logging
import os
import threading
import traceback
import zipfile

from flask import request, jsonify, send_file

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config import TrainConfig

task_states = dict()

work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace-remote")
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models-remote")
cache_dir = os.path.join(work_dir, "cache")
os.makedirs(work_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(models_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(models_dir, "vae"), exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler(os.path.join(work_dir, "server.log")),
        logging.StreamHandler()
    ]
)
logging.log(logging.INFO, f"Using work dir: {work_dir}")


def remove_workspace_files():
    # Delete the workspace
    for root, dirs, files in os.walk(work_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

def start_training_wrapper(task_id, config):
    try:
        start_training(task_id, config)
    except Exception as e:
        # Log the error and store it
        task_states[task_id]["easy_status"] = "failed"
        task_states[task_id]["error"] = str(e)
        task_states[task_id]["traceback"] = traceback.format_exc()

        # Optionally log it somewhere else as well
        logging.log(logging.ERROR, f"Error in task {task_id}: {e}")
        logging.log(logging.ERROR, traceback.format_exc())


def start_training(task_id, train_config_dict: dict):
    train_config = TrainConfig.TrainConfig.default_values()
    train_config.from_dict(train_config_dict)

    train_config.workspace_dir = work_dir

    # Create a task state entry
    task_states[task_id] = {"easy_status": "running"}

    callbacks = TrainCallbacks()
    callbacks.set_on_update_train_progress(lambda progress, a, b:
                                           task_states[task_id].update({"progress": progress.to_dict()}))
    callbacks.set_on_update_status(lambda status: task_states[task_id].update({"status": status}))
    callbacks.set_on_sample_default(lambda sample: task_states[task_id].update({"sample": sample}))
    callbacks.set_on_sample_custom(lambda sample: task_states[task_id].update({"sample": sample}))

    commands = TrainCommands()
    task_states[task_id]["commands"] = commands

    trainer = GenericTrainer(train_config, callbacks, commands)
    trainer.start()
    trainer.train()
    trainer.end()

    task_states[task_id] = {"easy_status": "completed"}


def stop_training(task_id):
    if task_id in task_states:
        task_states[task_id]["commands"].stop()
        return 200
    else:
        return jsonify({"error": "Task not found or not running"}), 404


def training_request_handler():
    # Get the request data
    data = request.get_data()
    if not data:
        return jsonify({"error": "Invalid data"}), 400

    config = json.loads(request.files.get('config').read()) if request.files.get('config') else {}
    concepts = request.files.getlist('concepts')

    if not config:
        return jsonify({"error": "Invalid config"}), 400
    if not concepts or len(concepts) == 0:
        return jsonify({"error": "No concepts provided"}), 400

    # create a random task_id
    task_id = os.urandom(16).hex()
    task_work_dir = work_dir

    # Replace the config path in the config with the task work dir
    config["output_model_destination"] = os.path.join(task_work_dir, "output")
    # Replace the model path with the remote models directory
    config["base_model_name"] = os.path.join(models_dir, "checkpoints", os.path.basename(config["base_model_name"]))
    if config["vae"]["model_name"]:
        config["vae"]["model_name"] = os.path.join(models_dir, "vae", config["vae"]["model_name"])
    # Replace the cache path with the work dir and create the cache dir
    config["cache_dir"] = cache_dir

    # Unpack training concepts
    for concept in concepts:
        name, content = concept.filename, concept.read()
        # write the content to a temporary file
        zip_file_path = os.path.join(task_work_dir, name + '.zip')

        target_path = os.path.join(task_work_dir, 'concepts', name)
        # make sure the target path exists
        os.makedirs(target_path, exist_ok=True)

        with open(zip_file_path, 'wb') as f:
            f.write(content)
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(target_path)
        # Remove the zip file after extraction
        os.remove(zip_file_path)

        # Replace the concept path in the config with the target path
        # Find the concept name in the config
        for c in config["concepts"]:
            if c['name'] == name:
                # Update the path
                c["path"] = target_path
                break

    # Start the training process in a separate thread and watch for errors
    threading.Thread(target=start_training_wrapper, args=(task_id, config)).start()

    # return the task_id
    return jsonify({"task_id": task_id}), 202


def create_app():
    from flask import Flask
    app = Flask(__name__)

    # Clean up the workspace directory on startup
    remove_workspace_files()

    @app.route('/train', methods=['POST'])
    def train():
        return training_request_handler()

    @app.route('/status/<task_id>', methods=['GET'])
    def status(task_id):
        if task_id in task_states:
            out = {
                "status": task_states[task_id]["status"],
                "easy_status": task_states[task_id]["easy_status"],
            }
            if "progress" in task_states[task_id]:
                out["progress"] = task_states[task_id]["progress"]
            return jsonify(out), 200
        else:
            return jsonify({"error": "Task not found"}), 404

    @app.route('/stop/<task_id>', methods=['POST'])
    def stop(task_id):
        if task_id not in task_states:
            # Stop the first running task found
            for k, v in task_states.items():
                task_id = k
                break
        return stop_training(task_id)

    @app.route('/delete_workspace', methods=['POST'])
    def delete_workspace():
        # Check if any task is running. If so, do not delete the workspace
        if any(task["easy_status"] == "running" for task in task_states.values()):
            return jsonify({"error": "Cannot delete workspace while tasks are running"}), 400

        remove_workspace_files()

    @app.route('/model/<task_id>', methods=['GET'])
    def get_latest_model(task_id):
        if task_id in task_states:
            # Assuming the model is saved in a specific directory
            model_dir = os.path.join(work_dir, task_id, "model")
            if os.path.exists(model_dir):
                # Create a zip file of the model directory
                zip_file_path = os.path.join(work_dir, f"{task_id}_model.zip")
                with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                    for root, _, files in os.walk(model_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, os.path.relpath(file_path, model_dir))
                return send_file(zip_file_path, as_attachment=True)
            else:
                return jsonify({"error": "Model not found"}), 404
        else:
            return jsonify({"error": "Task not found"}), 404

    return app


if __name__ == '__main__':
    # Create the Flask app
    app = create_app()
    # Run the Flask app
    app.run(debug=False)
