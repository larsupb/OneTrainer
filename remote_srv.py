import json
import logging
import os
import time
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


# create enum for task states
class TaskState:
    INITIALIZING = "initializing"
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    UNKNOWN = "unknown"


app = FastAPI()

# Add CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_methods=["*"],
    allow_headers=["*"]
)

base_work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace-remote")
output_dir = os.path.join(base_work_dir, "models")
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models-remote")
cache_dir = os.path.join(base_work_dir, "cache")

for path in [base_work_dir, output_dir, models_dir, cache_dir]:
    os.makedirs(path, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def get_task_workdir(task_id):
    task_dir = os.path.join(base_work_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir


def remove_workspace_files(task_id):
    for root, dirs, files in os.walk(get_task_workdir(task_id), topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


@app.on_event("startup")
def on_startup():
    remove_workspace_files(base_work_dir)


def start_training_wrapper(task_id, config):
    try:
        start_training(task_id, config)

    except Exception as e:
        logging.error(f"Error in task {task_id}: {e}")
        logging.error(traceback.format_exc())

        commands = task_states[task_id].get("commands")
        if commands:
            commands.stop()

        task_states[task_id]["rest_status"] = TaskState.ERROR
        task_states[task_id]["log"].append(traceback.format_exc())


def start_training(task_id, train_config_dict):
    train_config = TrainConfig.TrainConfig.default_values()
    train_config.from_dict(train_config_dict)
    train_config.workspace_dir = get_task_workdir(task_id)

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

    task_states[task_id]["start_date"] = time.time()
    task_states[task_id]["rest_status"] = TaskState.RUNNING
    trainer = GenericTrainer(train_config, callbacks, commands)
    trainer.start()
    trainer.train()
    trainer.end()
    task_states[task_id]["rest_status"] = TaskState.FINISHED
    task_states[task_id]["end_date"] = time.time()


@app.post("/train")
async def train(background_tasks: BackgroundTasks,
                config: UploadFile = File(...),
                concepts: List[UploadFile] = File(...)):
    if not config:
        raise HTTPException(400, detail="Invalid config")
    if not concepts or len(concepts) == 0:
        raise HTTPException(400, detail="No concepts provided")

    task_id = os.urandom(8).hex()
    task_states[task_id] = {
        "status": "initializing",
        "rest_status": TaskState.INITIALIZING,
        "queue_date": time.time(),
        "start_date": None,
        "end_date": None,
        "log": [],
    }

    config_data = json.loads(await config.read())
    filename = config_data["output_model_destination"].split("/")[-1]
    config_data["output_model_destination"] = os.path.join(output_dir, filename)

    local_checkpoint = os.path.join(models_dir, config_data["base_model_name"])
    if os.path.exists(local_checkpoint):
        config_data["base_model_name"] = local_checkpoint

    if config_data["vae"]["model_name"]:
        config_data["vae"]["model_name"] = os.path.join(models_dir, config_data["vae"]["model_name"])

    config_data["cache_dir"] = cache_dir

    for concept in concepts:
        name = concept.filename
        content = await concept.read()
        zip_path = os.path.join(get_task_workdir(task_id), name + '.zip')
        concept_path = os.path.join(get_task_workdir(task_id), 'concepts', name)
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
        "rest_status": state.get("rest_status", TaskState.UNKNOWN),
    }
    if "progress" in state:
        result["progress"] = state["progress"]
    return result


@app.get("/tensorboard/{task_id}")
async def tensorboard(task_id: str):
    logs_dir = task_states.get(task_id, {}).get("tensorboard_logs")
    if not logs_dir or not os.path.exists(logs_dir):
        raise HTTPException(404, detail="TensorBoard data not found")

    zip_path = os.path.join(base_work_dir, f"{task_id}_tensorboard_logs.zip")
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

    dir_path = os.path.join(get_task_workdir(task_id), file_type)
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

    file_path = os.path.join(get_task_workdir(task_id), file_type, filename)
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
    return JSONResponse({"exec": "stopping task"}, status_code=200)


@app.post("/delete_workspace/{task_id}")
async def delete_workspace(task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")

    if any(task["rest_status"] not in (TaskState.FINISHED, TaskState.ERROR) for task in task_states.values()):
        raise HTTPException(400, detail="Cannot delete workspace while tasks are running")
    remove_workspace_files(task_id)
    return JSONResponse({"exec": "workspace deleted"}, status_code=200)


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


@app.get("/task")
async def get_all_tasks():
    # List all tasks with their IDs and statuses
    tasks = []
    for task_id, state in task_states.items():
        tasks.append({
            "task_id": task_id,
            "status": state.get("status", "unknown"),
            "rest_status": state.get("rest_status", TaskState.UNKNOWN),
            "queue_date": state.get("queue_date"),
            "start_date": state.get("start_date"),
            "end_date": state.get("end_date"),
            "error": state.get("error", ""),
        })
    return JSONResponse(tasks, status_code=200)


@app.get("sample_custom/<task_id>")
async def sample_custom(task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")

    commands = task_states[task_id].get("commands")
    if commands:
        commands.sample_custom()
    return JSONResponse({"exec": "sample_custom executed."}, status_code=200)


@app.get("sample_default/<task_id>")
async def sample_default(task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")

    commands = task_states[task_id].get("commands")
    if commands:
        commands.sample_default()
    return JSONResponse({"exec": "sample_default executed."}, status_code=200)


@app.get("/backup/{task_id}")
async def backup(task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")

    commands = task_states[task_id].get("commands")
    if commands:
        commands.backup()
    return JSONResponse({"exec": "backup command executed"}, status_code=200)


@app.get("/save/{task_id}")
async def save(task_id: str):
    if task_id not in task_states:
        raise HTTPException(404, detail="Task not found")

    commands = task_states[task_id].get("commands")
    if commands:
        commands.save()
    return JSONResponse({"exec": "save command executed"}, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
