"""HTTP inference server for speculative T5 decoding."""

import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration

from speculative_decoding.decoding import DecodingStats, speculative_decode

MODEL_NAME = os.environ.get("MODEL_NAME", "google-t5/t5-small")
DRAFT_MODEL_NAME = os.environ.get("DRAFT_MODEL_NAME", "google-t5/t5-small")
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", "t5-small")
DTYPE = os.environ.get("DTYPE", "float32")
PORT = int(os.environ.get("PORT", "8000"))

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
TASK_PREFIXES = {
    "translate_en_de": "translate English to German: ",
    "summarize": "summarize: ",
    "raw": "",
}


@dataclass
class ModelState:
    tokenizer: PreTrainedTokenizerBase
    target: T5ForConditionalGeneration
    draft: T5ForConditionalGeneration
    device: torch.device


state: ModelState | None = None
gpu_lock = threading.Lock()


def validate_compatible_models(
    target: T5ForConditionalGeneration,
    draft: T5ForConditionalGeneration,
    target_tokenizer: PreTrainedTokenizerBase,
    draft_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Fail at startup unless both models use the same token IDs and T5 task conventions."""
    if not target.config.is_encoder_decoder or not draft.config.is_encoder_decoder:
        raise ValueError("Both target and draft must be encoder-decoder models.")
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("Target and draft models must have the same vocabulary size.")
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("Target and draft tokenizers must map tokens to identical token IDs.")
    special_token_ids = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    if any(getattr(target_tokenizer, key) != getattr(draft_tokenizer, key) for key in special_token_ids):
        raise ValueError("Target and draft tokenizers must use identical special-token IDs.")
    if target_tokenizer.all_special_ids != draft_tokenizer.all_special_ids:
        raise ValueError("Target and draft tokenizers must use identical special-token sets.")
    if target.config.decoder_start_token_id != draft.config.decoder_start_token_id:
        raise ValueError("Target and draft models must use the same decoder start token.")
    if target.config.eos_token_id != draft.config.eos_token_id:
        raise ValueError("Target and draft models must use the same EOS token.")
    target_token_ids = target_tokenizer.get_vocab().values()
    draft_token_ids = draft_tokenizer.get_vocab().values()
    if max(target_token_ids, default=-1) >= target.config.vocab_size:
        raise ValueError("Target tokenizer IDs must fit within the target model vocabulary.")
    if max(draft_token_ids, default=-1) >= draft.config.vocab_size:
        raise ValueError("Draft tokenizer IDs must fit within the draft model vocabulary.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DTYPE not in DTYPES:
        raise ValueError(f"DTYPE must be one of {list(DTYPES)}, got {DTYPE!r}")

    print(f"Loading target {MODEL_NAME} and draft {DRAFT_MODEL_NAME} as {DTYPE} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME)
    target = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=DTYPES[DTYPE])
    draft = T5ForConditionalGeneration.from_pretrained(DRAFT_MODEL_NAME, dtype=DTYPES[DTYPE])
    validate_compatible_models(target, draft, tokenizer, draft_tokenizer)
    target.to(device).eval()
    draft.to(device).eval()
    state = ModelState(tokenizer=tokenizer, target=target, draft=draft, device=device)
    print("Target and draft models ready.", flush=True)
    try:
        yield
    finally:
        state = None


app = FastAPI(title="Speculative T5 inference server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Input text (without task prefix unless task='raw').")
    task: Literal["translate_en_de", "summarize", "raw"] = "translate_en_de"
    max_new_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.0, ge=0.0, description="0 = greedy (argmax), >0 = sampling.")
    top_k: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    gamma: int = Field(4, ge=1, le=16, description="Maximum number of draft tokens per target-model call.")
    mode: Literal["speculative", "autoregressive"] = "speculative"
    seed: int | None = None


class GenerateResponse(BaseModel):
    model: str
    text: str
    token_ids: list[int]
    num_new_tokens: int
    latency_ms: float
    mode: Literal["speculative", "autoregressive"]
    iterations: int | None = None
    drafted_tokens: int | None = None
    accepted_tokens: int | None = None
    acceptance_rate: float | None = None


@app.get("/health")
def health():
    if state is None:
        raise HTTPException(status_code=503, detail="target and draft models are not loaded yet")
    return {"status": "ok"}


@app.get("/info")
def info():
    if state is None:
        raise HTTPException(status_code=503, detail="target and draft models are not loaded yet")
    target = state.target
    draft = state.draft
    target_num_parameters = sum(parameter.numel() for parameter in target.parameters())
    return {
        "model": SERVED_NAME,
        "checkpoint": MODEL_NAME,
        "draft_checkpoint": DRAFT_MODEL_NAME,
        "device": str(state.device),
        "dtype": DTYPE,
        "num_parameters": target_num_parameters,
        "target_num_parameters": target_num_parameters,
        "draft_num_parameters": sum(parameter.numel() for parameter in draft.parameters()),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if state is None:
        raise HTTPException(status_code=503, detail="target and draft models are not loaded yet")
    tokenizer = state.tokenizer
    target = state.target
    draft = state.draft
    device = state.device

    inputs = tokenizer(
        TASK_PREFIXES[req.task] + req.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    gen_kwargs = {"max_new_tokens": req.max_new_tokens, "num_beams": 1}
    if req.temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=req.temperature,
            top_k=req.top_k or 0,
            top_p=req.top_p or 1.0,
        )
    else:
        gen_kwargs.update(do_sample=False)

    with gpu_lock:
        if req.seed is not None:
            torch.manual_seed(req.seed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            stats: DecodingStats | None = None
            if req.mode == "speculative":
                token_ids, stats = speculative_decode(
                    target=target,
                    draft=draft,
                    input_ids=inputs["input_ids"],
                    temperature=req.temperature,
                    gamma=req.gamma,
                    max_new_tokens=req.max_new_tokens,
                    use_cache=True,
                    top_k=req.top_k,
                    top_p=req.top_p,
                )
            else:
                output = target.generate(**inputs, **gen_kwargs)
                token_ids = output[0].tolist()[1:]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000

    return GenerateResponse(
        model=SERVED_NAME,
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids=token_ids,
        num_new_tokens=len(token_ids),
        latency_ms=latency_ms,
        mode=req.mode,
        iterations=stats.iterations if stats else None,
        drafted_tokens=stats.drafted if stats else None,
        accepted_tokens=stats.accepted if stats else None,
        acceptance_rate=stats.acceptance_rate if stats else None,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
