import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


# =========================
# REQUIRED ENV VARIABLES
# =========================

CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")


# =========================
# VALIDATION (FAIL FAST)
# =========================

if not CLOUDFLARE_API_TOKEN:
    raise RuntimeError(
        "CLOUDFLARE_API_TOKEN is missing. "
        "Set it in your .env file."
    )

if not CLOUDFLARE_ACCOUNT_ID:
    raise RuntimeError(
        "CLOUDFLARE_ACCOUNT_ID is missing. "
        "Set it in your .env file."
    )


# =========================
# CLOUDFLARE AI MODELS
# =========================

# Embedding model
EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5"

# LLM model
LLM_MODEL = "@cf/meta/llama-3-8b-instruct"


# =========================
# CLOUDFLARE API BASE URL
# =========================

CLOUDFLARE_AI_BASE_URL = (
    "https://api.cloudflare.com/client/v4/accounts"
)


# =========================
# COMMON HEADERS
# =========================

HEADERS = {
    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
    "Content-Type": "application/json",
}

