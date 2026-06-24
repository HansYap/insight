from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from .florence_model import FlorenceInferencer
from .chroma_database import SceneMemory
from utils.normalise_vmotion import normalize_vmotion
import uvicorn
import json
import uuid
import shutil
from pathlib import Path
import base64
import numpy as np
import cv2
import yaml


def load_config(path="config/vmotion_norm.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["normalization"]

app = FastAPI()
inferencer = FlorenceInferencer() 
memory = SceneMemory()

PENDING_DIR = Path(__file__).parent.parent / "data" / "pending"
PENDING_DIR.mkdir(exist_ok=True)
FRAMES_DIR = PENDING_DIR / "frames"
FRAMES_DIR.mkdir(exist_ok=True)
QUEUE_FILE = PENDING_DIR / "queue.json"
NORM_CONFIG = load_config()

app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")

def load_queue() -> list:
    if not QUEUE_FILE.exists():
        return []
    return json.loads(QUEUE_FILE.read_text())

def save_queue(queue: list):
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))


@app.post("/describe")
async def describe_frame(
    frame: UploadFile = File(...),
    prompt: str = Form(default="<MORE_DETAILED_CAPTION>"),
    mean_magnitude: float = Form(default=None),
    std_magnitude:  float = Form(default=None),
    directionality: float = Form(default=None),
    coverage_ratio: float = Form(default=None),
    dominant_sin:   float = Form(default=None),
    dominant_cos:   float = Form(default=None),
):
    image_bytes = await frame.read()
    result = inferencer.describe(image_bytes, prompt)
    description = result["description"]

    v_motion_norm = None
    if mean_magnitude is not None:
        raw = {
            "mean_magnitude": mean_magnitude,
            "std_magnitude":  std_magnitude,
            "directionality": directionality,
            "coverage_ratio": coverage_ratio,
            "dominant_sin":   dominant_sin,
            "dominant_cos":   dominant_cos,
        }
        v_motion_norm = normalize_vmotion(raw, NORM_CONFIG)
    
    match = memory.query(description, v_motion=v_motion_norm)


    return {
        "description": description,
        "confident": match["confident"],
        "label": match["label"],
        "score": match["score"],
        "v_motion": v_motion_norm.tolist() if v_motion_norm is not None else None,
    }

@app.post("/queue-pending")
async def queue_pending(
    frame: UploadFile = File(...),
    description: str = Form(...),
    event_type: str = Form(...),
    score: float = Form(...),
    v_motion: str = Form(default=None),
):
    item_id = str(uuid.uuid4())
    frame_filename = f"{item_id}.jpg"
    frame_bytes = await frame.read()
    (FRAMES_DIR / frame_filename).write_bytes(frame_bytes)

    queue = load_queue()

    queue.append({
        "id": item_id,
        "event_type": event_type,
        "description": description,
        "score": score,
        "frame_url": f"/frames/{frame_filename}",
        "timestamp": __import__("time").time(),
        "v_motion": json.loads(v_motion),
    })
    save_queue(queue)
    return {"queued": True, "id": item_id, "total_pending": len(queue)}


@app.get("/pending")
def get_pending():
    return load_queue()


@app.post("/label/{item_id}")
async def label_item(item_id: str, activity: str = Form(...), subject: str = Form(...)):
    queue = load_queue()
    item = next((i for i in queue if i["id"] == item_id), None)

    if not item:
        return {"error": "Item not found"}
    
    v_motion_norm = None
    if item.get("v_motion"):
        v_motion_norm = np.array(item["v_motion"])

    ## TODO ==== Store subsequent activities even if no longer need to label
   
    used_label = memory.store(item["description"], activity, subject, v_motion=v_motion_norm)

    queue = [i for i in queue if i["id"] != item_id]
    save_queue(queue)
    frame_path = FRAMES_DIR / f"{item_id}.jpg"
    if frame_path.exists():
        frame_path.unlink()

    return {"labeled": True, "label": used_label, "remaining": len(queue)}

# api_server.py
@app.get("/labels")
def get_labels():
    """Return all unique labels currently in ChromaDB."""
    if memory.collection.count() == 0:
        return {"labels": []}
    all_meta = memory.collection.get(include=["metadatas"])["metadatas"]
    
    # count usage of each label
    counts = {}
    for m in all_meta:
        label = m["label"]
        counts[label] = counts.get(label, 0) + 1
    
    # sort by most used first
    sorted_labels = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return {"labels": [{"label": l, "count": c} for l, c in sorted_labels]}


@app.delete("/dismiss/{item_id}")
def dismiss_item(item_id: str):
    """Discard an item without labeling (e.g. false trigger)."""
    queue = load_queue()
    queue = [i for i in queue if i["id"] != item_id]
    save_queue(queue)
    frame_path = FRAMES_DIR / f"{item_id}.jpg"
    if frame_path.exists():
        frame_path.unlink()
    return {"dismissed": True}


@app.post("/save_frame")
async def save_frame(request: Request):
    body = await request.json()
    rel_path = body["path"]          # e.g. "motion/TRIGGER_0001.jpg"
    img_b64  = body["image"]

    import base64, numpy as np
    img_bytes = base64.b64decode(img_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    save_path = Path("debug_frames") / rel_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), frame)
    return {"saved": str(save_path)}


@app.get("/inbox", response_class=HTMLResponse)
def inbox_ui():
    return HTMLResponse(content=open(
        Path(__file__).parent / "inbox.html"
    ).read())


@app.get("/health")
def health():
    return {"status": "ok", "memories": memory.collection.count()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)