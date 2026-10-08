import os
import secrets
from retrieval import retrieve
from threading import Thread
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

app = FastAPI()

API_KEY = os.environ.get("API_KEY")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="Server has no API_KEY configured")
    if key is None or not secrets.compare_digest(key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: str = "You are a helpful assistant."
    max_new_tokens: int = 200
    temperature: float = 0.7


class ChatResponse(BaseModel):
    reply: str

class AskRequest(BaseModel):
    question: str
    max_new_tokens: int = 150

def build_inputs(req: ChatRequest):
    messages = [{"role": "system", "content": req.system_prompt}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


def build_gen_kwargs(req: ChatRequest):
    gen_kwargs = {"max_new_tokens": req.max_new_tokens}
    if req.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = req.temperature
    else:
        gen_kwargs["do_sample"] = False
    return gen_kwargs

   
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(req: ChatRequest):
    inputs = build_inputs(req)
    output = model.generate(**inputs, **build_gen_kwargs(req))

    prompt_length = inputs["input_ids"].shape[1]
    reply = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return {"reply": reply}

@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_stream(req: ChatRequest):
    inputs = build_inputs(req)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(**inputs, streamer=streamer, **build_gen_kwargs(req))
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def token_generator():
        for piece in streamer:
            yield piece

    return StreamingResponse(token_generator(), media_type="text/plain")

MIN_SCORE = 0.25


@app.post("/ask", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def ask(req: AskRequest):
    results = retrieve(req.question, k=2)

    if results[0][0] < MIN_SCORE:
        return {"reply": "I don't know based on the available documents."}

    context = "\n\n".join(chunk for _, chunk in results)
    system_prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=False)

    prompt_length = inputs["input_ids"].shape[1]
    reply = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return {"reply": reply}
