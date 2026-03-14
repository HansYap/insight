from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from florence_model import FlorenceInferencer
from chroma_database import SceneMemory
import uvicorn
import json
import uuid
import shutil
from pathlib import Path


app = FastAPI()
inferencer = FlorenceInferencer() 
memory = SceneMemory()

PENDING_DIR = Path(__file__).parent.parent / "data" / "chroma_db" / "pending"
PENDING_DIR.mkdir(exist_ok=True)
FRAMES_DIR = PENDING_DIR / "frames"
FRAMES_DIR.mkdir(exist_ok=True)
QUEUE_FILE = PENDING_DIR / "queue.json"

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
    prompt: str = Form(default="<MORE_DETAILED_CAPTION>")
):
    image_bytes = await frame.read()
    result = inferencer.describe(image_bytes, prompt)
    description = result["description"]

    match = memory.query(description)

    return {
        "description": description,
        "confident": match["confident"],
        "label": match["label"],
        "score": match["score"],
    }

@app.post("/queue-pending")
async def queue_pending(
    frame: UploadFile = File(...),
    description: str = Form(...),
    event_type: str = Form(...),
    score: float = Form(...)
):
    """Pi calls this when Florence is uncertain. Saves frame + metadata to inbox."""
    item_id = str(uuid.uuid4())
    
    # Save frame to disk so the UI can display it
    frame_filename = f"{item_id}.jpg"
    frame_bytes = await frame.read()
    (FRAMES_DIR / frame_filename).write_bytes(frame_bytes)

    # Add to queue
    queue = load_queue()
    queue.append({
        "id": item_id,
        "event_type": event_type,
        "description": description,
        "score": score,
        "frame_url": f"/frames/{frame_filename}",
        "timestamp": __import__("time").time()
    })
    save_queue(queue)

    return {"queued": True, "id": item_id, "total_pending": len(queue)}


@app.get("/pending")
def get_pending():
    return load_queue()


@app.post("/label/{item_id}")
async def label_item(item_id: str, label: str = Form(...)):
    """User submits a label from the inbox UI."""
    queue = load_queue()
    item = next((i for i in queue if i["id"] == item_id), None)
    
    if not item:
        return {"error": "Item not found"}

    # Store to ChromaDB
    memory.store(item["description"], label)

    # Remove from queue and clean up frame
    queue = [i for i in queue if i["id"] != item_id]
    save_queue(queue)
    frame_path = FRAMES_DIR / f"{item_id}.jpg"
    if frame_path.exists():
        frame_path.unlink()

    return {"labeled": True, "label": label, "remaining": len(queue)}


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


@app.get("/inbox", response_class=HTMLResponse)
def inbox_ui():
    return HTMLResponse(content=open(
        Path(__file__).parent / "inbox.html"
    ).read())


@app.get("/health")
def health():
    return {"status": "ok", "memories": memory.collection.count()}

if __name__ == "__main__":
    # 0.0.0.0 so Pi can reach it over LAN
    uvicorn.run(app, host="0.0.0.0", port=8000)