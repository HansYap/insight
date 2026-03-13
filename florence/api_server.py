# thinkpad/server.py
from fastapi import FastAPI, UploadFile, File, Form
from florence_model import FlorenceInferencer
import uvicorn
from chroma_database import SceneMemory

app = FastAPI()
inferencer = FlorenceInferencer() 
memory = SceneMemory()

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

app.post("/store")
async def store_memory(description: str = Form(...), label: str = Form(...)):   
    memory.store(description, label)
    return {"stored": True, "label": label,}

@app.get("/health")
def health():
    return {"status": "ok", "memories": memory.collection.count()}

if __name__ == "__main__":
    # 0.0.0.0 so Pi can reach it over LAN
    uvicorn.run(app, host="0.0.0.0", port=8000)