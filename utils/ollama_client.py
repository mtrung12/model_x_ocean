from dotenv import load_dotenv
import os
import requests
from typing import Dict, List

load_dotenv()

_BASE_URL = "https://ollama.com/api"


def _get_headers() -> Dict[str, str]:
    api_key = os.getenv("OLLAMA_KEY")
    if not api_key:
        raise ValueError("OLLAMA_KEY is not set.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def create_message(sys_prompt: str, usr_prompt: str) -> List[Dict]:
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]


def ollama_call(
    user_prompt: str,
    system_prompt: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = create_message(system_prompt, user_prompt)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens,
        },
    }
    resp = requests.post(
        f"{_BASE_URL}/chat",
        headers=_get_headers(),
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
