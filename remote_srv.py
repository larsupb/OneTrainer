import json
import logging
import os
import traceback
import zipfile
from typing import List

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util import TrainProgress
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config import TrainConfig

task_states = {}

app = FastAPI()

# Add CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_methods=["*"],
    allow_headers=["*"]
)

work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace-remote")
output_dir = os.path.join(work_dir, "models")
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models-remote")
cache_dir = os.path.join(work_dir, "cache")

for path in [work_dir, output_dir, models_dir, cache_dir,
             os.path.join(models_dir, "checkpoints"), os.path.join(models_dir, "vae")]:
    os.makedirs(path, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def remove_workspace_files():
    for root, dirs, files in os.walk(work_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


@app.on_event("startup")
def on_startup():
    remove_workspace_files()


def start_training_wrapper(task_id, config):
    try:
        start_training(task_id, config)
    except Exception as e:
        logging.error(f"Error in task {task_id}: {e}")
        logging.error(traceback.format_exc())

        commands = task_states[task_id].get("commands")
        if commands:
            commands.stop()

        task_states[task_id]["easy_status"] = "failed"
        task_states[task_id]["error"] = str(e)
        task_states[task_id]["traceback"] = traceback.format_exc()


def start_training(task_id, train_config_dict):
    train_config = TrainConfig.TrainConfig.default_values()
    train_config.from_dict(train_config_dict)
    train_config.workspace_dir = work_dir

    def on_train_progress(train_progress: TrainProgress, max_sample, max_epoch):
        task_states[task_id]["progress"] = {
            "epoch": train_progress.epoch,
            "epoch_sample": train_progress.epoch_sample,
            "epoch_step": train_progress.epoch_step,
            "global_step": train_progress.global_step,
            "max_sample": max_sample,
            "max_epoch": max_epoch
        }

    callbacks = TrainCallbacks()
    callbacks.set_on_update_train_progress(on_train_progress)
    callbacks.set_on_update_status(lambda s: task_states[task_id].update({"status": s}))
    callbacks.set_on_sample_default(lambda s: task_states[task_id].update({"sample": s}))
    callbacks.set_on_sample_custom(lambda s: task_states[task_id].update({"sample": s}))

    commands = TrainCommands()
    task_states[task_id]["commands"] = commands
    if train_config.tensorboard:
        task_states[task_id]["tensorboard_logs"] = os.path.join(train_config.workspace_dir, "tensorboard")

    trainer = GenericTrainer(train_config, callbacks, commands)
    trainer.start()
    trainer.train()
    trainer.end()

    task_states[task_id]["easy_status"] = "completed"


@app.post("/train")
async def train(background_tasks: BackgroundTasks,
                config: UploadFile = File(...),
                concepts: List[UploadFile] = File(...)):
    if not config:
        raise HTTPException(400, detail="Invalid config")
    if not concepts or len(concepts) == 0:
        raise HTTPException(400, detail="No concepts provided")

    task_id = os.urandom(16).hex()
    task_states[task_id] = {"easy_status": "running", "status": "initializing"}

    config_data = json.loads(await config.read())
    filename = config_data["output_model_destination"].split("/")[-1]
    config_data["output_model_destination"] = os.path.join(output_dir, filename)

    checkpoint_name = config_data["base_model_name"].split("checkpoints/")[-1]
    config_data["base_model_name"] = os.path.join(models_dir, "checkpoints", checkpoint_name)

    if config_data["vae"]["model_name"]:
        config_data["vae"]["model_name"] = os.path.join(models_dir, "vae", config_data["vae"]["model_name"])

    config_data["cache_dir"] = cache_dir

    for concept in concepts:
        name = concept.filename
        content = await concept.read()
        zip_path = os.path.join(work_dir, name + '.zip')
        concept_path = os.path.join(work_dir, 'concepts', name)
        os.makedirs(concept_path, exist_ok=True)

        with open(zip_path, 'wb') as f:
            f.write(content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(concept_path)
        os.remove(zip_path)

        for c in config_data["concepts"]:
            if c['name'] == name:
                c["path"] = concept_path
                break

    background_tasks.add_task(start_training_wrapper, task_id, config_data)

    return JSONResponse({"task_id": task_id}, status_code=202)


@app.get("/status/{task_id}")
async def status(task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")
    state = task_states[task_id]
    result = {
        "status": state.get("status", ""),
        "easy_status": state.get("easy_status", "")
    }
    if "progress" in state:
        result["progress"] = state["progress"]
    return result


@app.get("/tensorboard/{task_id}")
async def tensorboard(task_id: str):
    logs_dir = task_states.get(task_id, {}).get("tensorboard_logs")
    if not logs_dir or not os.path.exists(logs_dir):
        raise HTTPException(404, detail="TensorBoard data not found")

    zip_path = os.path.join(work_dir, f"{task_id}_tensorboard_logs.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(logs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, logs_dir))
    return FileResponse(zip_path, media_type="application/zip", filename=f"{task_id}_tensorboard_logs.zip")


@app.get("/files/list/{file_type}/{task_id}")
async def list_files(file_type: str, task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")
    if file_type not in ["samples", "save", "backup"]:
        raise HTTPException(400, detail="Invalid file type")

    dir_path = os.path.join(work_dir, file_type)
    if not os.path.exists(dir_path):
        return []

    files = []
    for root, _, file_list in os.walk(dir_path):
        for file in file_list:
            files.append(os.path.relpath(os.path.join(root, file), dir_path))
    return files


@app.get("/files/download/{file_type}/{task_id}")
async def download_file(file_type: str, task_id: str, filename: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")
    if file_type not in ["samples", "save", "backup"]:
        raise HTTPException(400, detail="Invalid file type")

    file_path = os.path.join(work_dir, file_type, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="File not found")

    return FileResponse(file_path, filename=filename, media_type='application/octet-stream')


@app.post("/stop/{task_id}")
async def stop(task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")

    commands = task_states[task_id].get("commands")
    if commands:
        commands.stop()
    return {"status": "stopped"}


@app.post("/delete_workspace/{task_id}")
async def delete_workspace(task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")

    if any(task["easy_status"] == "running" for task in task_states.values()):
        raise HTTPException(400, detail="Cannot delete workspace while tasks are running")
    remove_workspace_files()
    return {"status": "workspace deleted"}


@app.get("/model/{task_id}")
async def get_latest_model(task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")
    if not os.path.exists(output_dir):
        raise HTTPException(404, detail="Model not found")
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    if not files:
        raise HTTPException(404, detail="No model file found")
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
    return FileResponse(os.path.join(output_dir, latest), filename=latest)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0", port=8000, log_level="info")
