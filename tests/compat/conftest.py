"""Shared fixtures for the llm-init Python compatibility suite.

The suite sits on top of the docker-compose stack in
``deploy/compose/llamacpp.yml`` — same stack the Go integration tests
use. Two fixtures matter:

``llm_init_url``
    Base URL of llm-init's data plane. Defaults to
    ``http://localhost:8080`` and can be overridden via
    ``LLM_INIT_URL`` so CI can point at a job-local stack.

``compose_stack``
    Session-scoped autouse fixture that brings the llamacpp compose
    stack up (``docker compose ... up -d --wait``) and tears it down
    on session end. Skipped entirely when ``LLM_INIT_URL`` is set —
    that means a stack is already running and we should reuse it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "deploy" / "compose" / "llamacpp.yml"

# v1.1.0 unified config. MODEL_MODE seeds the model-spec (chat);
# MODEL_SOURCE carries the HF repo + inline --include / --revision flags
# that the removed HF_REPO / HF_FILE / HF_REVISION envs used to hold, and
# the context size folds into ENGINE_ARGS (-c). The commit SHA is pinned
# because llm-init rejects "main" / branch refs at config load time;
# refresh it when HF retags the repo.
DEFAULT_MODEL_ENV = {
    "MODEL_NAME": "tinyllama-q4",
    "MODEL_MODE": "chat",
    "MODEL_SOURCE": (
        "hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF "
        "--include tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf "
        "--revision 52e7645ba7c309695bec7ac98f4f005b139cf465"
    ),
    "ENGINE_ARGS": "-c 2048",
    "LLM_INIT_PORT": "8080",
    "LLAMACPP_PORT": "8081",
    "LOG_LEVEL": "info",
    "LOG_FORMAT": "json",
}


@pytest.fixture(scope="session")
def llm_init_url() -> str:
    """Base URL llm-init publishes its data plane on."""
    return os.environ.get("LLM_INIT_URL", "http://localhost:8080").rstrip("/")


@pytest.fixture(scope="session")
def model_name() -> str:
    return os.environ.get("MODEL_NAME", DEFAULT_MODEL_ENV["MODEL_NAME"])


@pytest.fixture(scope="session", autouse=True)
def compose_stack(llm_init_url: str) -> Iterator[None]:
    """Start the llamacpp compose stack for the test session.

    Skipped when ``LLM_INIT_URL`` is already set (means: an external
    stack is already running, e.g. wired by CI) or ``docker`` is not
    on PATH (means: developer is running compat tests against a manual
    instance).
    """
    if os.environ.get("LLM_INIT_URL"):
        yield
        return

    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH; cannot bring compose stack up")

    if not COMPOSE_FILE.exists():
        pytest.skip(f"compose file missing: {COMPOSE_FILE}")

    env = os.environ.copy()
    for key, value in DEFAULT_MODEL_ENV.items():
        env.setdefault(key, value)

    up = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--wait"],
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )
    if up.returncode != 0:
        pytest.skip(f"docker compose up failed (rc={up.returncode}); see logs above")

    try:
        _wait_ready(llm_init_url, timeout_s=600)
        yield
    finally:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v",
             "--remove-orphans"],
            env=env,
            cwd=REPO_ROOT,
            check=False,
        )


def _wait_ready(base_url: str, timeout_s: float) -> None:
    """Poll /readyz until 200 or until the timeout fires."""
    deadline = time.time() + timeout_s
    last_status: int | None = None
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/readyz", timeout=5.0)
            last_status = resp.status_code
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            last_status = None
        time.sleep(2.0)
    raise TimeoutError(
        f"readiness timeout after {timeout_s}s (last status={last_status})"
    )
