# thinkpad/florence_model.py
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import io

MODEL_ID = "microsoft/Florence-2-base"  # ~900MB — use base not large for now

class FlorenceInferencer:
    def __init__(self):
        print("Loading Florence-2... (first load takes ~30s)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        print(f"Florence-2 loaded on {self.device}")

    def describe(self, image_bytes: bytes, prompt: str = "<MORE_DETAILED_CAPTION>") -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )
        
        description = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        description = self.processor.post_process_generation(
            description, task=prompt, image_size=(image.width, image.height)
        )
        
        return {"description": description, "prompt_used": prompt}