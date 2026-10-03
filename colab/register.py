"""
Registers this Colab runtime's current public URL with our backend, so the
backend knows where to forward /api/correct requests.

Reads secrets from environment variables only — never hardcode tokens here.
"""

import os
import requests

BACKEND_URL = os.environ.get("BACKEND_URL")
BACKEND_REGISTRATION_TOKEN = os.environ.get("BACKEND_REGISTRATION_TOKEN")


def register_with_backend(colab_url: str) -> None:
    if not BACKEND_URL:
        raise RuntimeError("BACKEND_URL environment variable is not set.")
    if not BACKEND_REGISTRATION_TOKEN:
        raise RuntimeError("BACKEND_REGISTRATION_TOKEN environment variable is not set.")

    endpoint = f"{BACKEND_URL.rstrip('/')}/api/colab/register"

    response = requests.post(
        endpoint,
        json={"url": colab_url},
        headers={"Authorization": f"Bearer {BACKEND_REGISTRATION_TOKEN}"},
        timeout=10,
    )

    if not response.ok:
        raise RuntimeError(
            f"Failed to register with backend ({response.status_code}): {response.text}"
        )

    print(f"Registered {colab_url} with backend at {endpoint}")
