import json
import re
import urllib.error
import urllib.request


OLLAMA_HOST = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"


def ollama_chat(
    messages: list[dict],
    model: str = CHAT_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout: int = 120,
    json_mode: bool = False,
    num_predict: int | None = None
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "top_p": top_p
        }
    }
    if json_mode:
        payload["format"] = "json"
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    result = post_json(f"{OLLAMA_HOST}/api/chat", payload, timeout=timeout)
    return result.get("message", {}).get("content", "").strip()


def generate_reply(system_prompt: str, user_message: str) -> str:
    return ollama_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.8,
        top_p=0.9,
        timeout=180
    )


def get_embedding(text: str) -> list[float]:
    cleaned = " ".join(text.strip().split())[:4000]
    if not cleaned:
        return []

    try:
        result = post_json(
            f"{OLLAMA_HOST}/api/embeddings",
            {"model": EMBED_MODEL, "prompt": cleaned},
            timeout=45
        )
        return [float(value) for value in result.get("embedding", [])]
    except RuntimeError:
        result = post_json(
            f"{OLLAMA_HOST}/api/embed",
            {"model": EMBED_MODEL, "input": cleaned},
            timeout=45
        )
        embeddings = result.get("embeddings", [])
        if embeddings and isinstance(embeddings[0], list):
            return [float(value) for value in embeddings[0]]
        return []


def classify_emotion_json(text: str) -> dict:
    prompt = """
Classify the user's message for an emotional companion app.
Return only valid JSON with these keys:
emotion, intensity, risk, confidence, reason.

Allowed emotions:
happy, sad, lonely, heartbroken, angry, anxious, romantic, confused, guilty, neutral, crisis

Allowed intensity:
low, medium, high

Allowed risk:
normal, crisis

Use crisis only when the user expresses self-harm, suicide, immediate danger, or intent to harm someone.
""".strip()

    raw = ollama_chat(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
        top_p=0.8,
        timeout=90,
        json_mode=True,
        num_predict=160
    )
    return parse_json_object(raw)


def parse_json_object(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise RuntimeError("Ollama did not return JSON")
        return json.loads(match.group(0))


def post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to Ollama at {url}") from exc
