import os
import threading
from pathlib import Path
from typing import Optional, Tuple

from unsloth import FastLanguageModel

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

app = FastAPI(title="Instrumentation Loop Assistant")
_model: Optional[PeftModel] = None
_tokenizer: Optional[PreTrainedTokenizerBase] = None
_inference_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)


class GenerateResponse(BaseModel):
    text: str


def _load_model() -> Tuple[PeftModel, PreTrainedTokenizerBase]:
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if not torch.cuda.is_available():
        raise HTTPException(
            status_code=503,
            detail="CUDA GPU not detected. Inference requires a supported NVIDIA GPU.",
        )

    adapter_setting = os.environ.get("ADAPTER_PATH")
    if not adapter_setting:
        raise HTTPException(
            status_code=503,
            detail="Set ADAPTER_PATH to the saved LoRA adapter directory before starting the server.",
        )

    adapter_path = Path(adapter_setting).expanduser()
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path
    if not adapter_path.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"LoRA adapter directory does not exist: {adapter_path}",
        )

    base_model = os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=int(os.environ.get("MAX_SEQ_LENGTH", "2048")),
        dtype=dtype,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    FastLanguageModel.for_inference(model)

    _model = model
    _tokenizer = tokenizer
    return model, tokenizer


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "base_model": os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
        "adapter_configured": bool(os.environ.get("ADAPTER_PATH")),
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be blank.")

    with _inference_lock:
        model, tokenizer = _load_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an instrumentation engineering assistant. Be precise, identify "
                    "assumptions, ask for missing project data, and never present unverified "
                    "output as approved design."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=700,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return GenerateResponse(
            text=tokenizer.decode(new_tokens, skip_special_tokens=True)
        )


frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
