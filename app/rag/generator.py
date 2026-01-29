import requests
from app.config import (
    CLOUDFLARE_ACCOUNT_ID,
    LLM_MODEL,
    CLOUDFLARE_AI_BASE_URL,
    HEADERS,
)

def generate_answer(context: str, question: str) -> str:
    """
    Generate an answer using Cloudflare LLM
    strictly grounded in the provided context.
    """

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    url = f"{CLOUDFLARE_AI_BASE_URL}/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{LLM_MODEL}"

    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400,
    }

    response = requests.post(
        url,
        headers=HEADERS,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()["result"]

    # Cloudflare returns plain text in "response"
    return result["response"]
