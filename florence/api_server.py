# thinkpad/server.py
from fastapi import FastAPI, UploadFile, File, Form
from florence_model import FlorenceInferencer
import uvicorn

app = FastAPI()
inferencer = FlorenceInferencer()  # loads once at startup

@app.post("/describe")
async def describe_frame(
    frame: UploadFile = File(...),
    prompt: str = Form(default="<MORE_DETAILED_CAPTION>")
):
    image_bytes = await frame.read()
    result = inferencer.describe(image_bytes, prompt)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # 0.0.0.0 so Pi can reach it over LAN
    uvicorn.run(app, host="0.0.0.0", port=8000)