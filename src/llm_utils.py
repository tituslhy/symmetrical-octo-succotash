import os

import openai
from dotenv import find_dotenv, load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai

_ = load_dotenv(find_dotenv())

ollama_llm = "qwen3.5:35b"

client= wrap_openai(
    openai.OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/"),
        api_key=os.environ["OPENAI_API_KEY"]
    )
)

def _resolve_model(model: str | None) -> str:
    """Use Ollama model when pointing at local endpoint, otherwise use provided model."""
    if os.environ.get("OPENAI_BASE_URL", "").startswith("http://localhost"):
        return ollama_llm
    return model or "gpt-5.4-nano"  

def call_llm(messages: list[dict], model: str = None) -> str:
    """Calls the OpenAI LLM client"""
    response = client.chat.completions.create(
        model    = _resolve_model(model),
        messages = messages,
        temperature = 0,  # deterministic — important for eval reproducibility
    )
    return response.choices[0].message.content

@traceable(name="stream_llm")
def stream_llm(messages: list[dict], model: str = None):
    """Streams response chunks, yields strings, returns full response at end."""
    response = client.chat.completions.create(
        model       = _resolve_model(model),
        messages    = messages,
        temperature = 0,
        stream      = True,
    )
    full = []
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            full.append(delta)
            yield delta
    return "".join(full)
