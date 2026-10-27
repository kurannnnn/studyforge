"""
Talks to the local llama-server (llama.cpp) over its OpenAI-compatible
HTTP API. The model runs in that separate process; if it is not running,
it is started automatically (see src/server_manager.py).
"""

import json

import requests

from src.config import SERVER_URL

_ready = False


def _check_ready() -> None:

    global _ready

    if _ready:
        return

    from src.server_manager import ensure_running, is_up

    if not is_up():
        try:
            ensure_running()
        except Exception as error:
            raise RuntimeError(f"The model server is not running and could not be started: {error}") from error

    _ready = True


def _payload(prompt, system, max_new_tokens, temperature, top_p, repetition_penalty, stream):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": stream,
        "cache_prompt": True,
    }


def _post(payload: dict, stream: bool):

    global _ready

    for attempt in (1, 2):

        _check_ready()

        try:
            response = requests.post(
                f"{SERVER_URL}/v1/chat/completions",
                json=payload,
                stream=stream,
                timeout=900,
            )
            break
        except requests.ConnectionError:
            # The server stopped since the last call: start it again once.
            _ready = False
            if attempt == 2:
                raise RuntimeError(f"Can't reach the model server at {SERVER_URL}.")

    if response.status_code != 200:
        raise RuntimeError(
            f"llama-server returned {response.status_code}: {response.text[:300]}"
        )

    return response


def chat(prompt, system, max_new_tokens, temperature, top_p, repetition_penalty) -> str:

    response = _post(
        _payload(prompt, system, max_new_tokens, temperature, top_p, repetition_penalty, False),
        stream=False,
    )

    return response.json()["choices"][0]["message"]["content"].strip()


def chat_stream(prompt, system, max_new_tokens, temperature, top_p, repetition_penalty):
    """Yield the reply piece by piece."""

    response = _post(
        _payload(prompt, system, max_new_tokens, temperature, top_p, repetition_penalty, True),
        stream=True,
    )

    response.encoding = "utf-8"

    with response:
        for line in response.iter_lines(decode_unicode=True):

            if not line or not line.startswith("data:"):
                continue

            data = line[5:].strip()

            if data == "[DONE]":
                break

            choices = json.loads(data).get("choices") or []

            if not choices:
                continue

            piece = choices[0].get("delta", {}).get("content")

            if piece:
                yield piece
