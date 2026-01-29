import requests
from app.config import CLOUDFLARE_ACCOUNT_ID, EMBEDDING_MODEL, CLOUDFLARE_AI_BASE_URL, HEADERS

def embed_text(texts: list[str]) -> list[list[float]]:
    url = f"{CLOUDFLARE_AI_BASE_URL}/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{EMBEDDING_MODEL}"

    response = requests.post(
        url,
        headers=HEADERS,
        json={"text": texts},
        timeout=30,
    )
    response.raise_for_status()

    embeddings = response.json()["result"]["data"]
    return embeddings