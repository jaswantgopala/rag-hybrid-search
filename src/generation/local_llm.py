import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:latest"


def local_chat(prompt, model=DEFAULT_MODEL, temperature=0):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def local_chat_json(prompt, model=DEFAULT_MODEL, fallback=None):
    raw = local_chat(prompt, model=model)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback if fallback is not None else {}