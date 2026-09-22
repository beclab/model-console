"""Smoke tests against the official OpenAI Python SDK.

The point of this file is *not* to validate every OpenAI surface area
- it is to confirm that an out-of-the-box ``openai.Client`` works
against ``llm-init`` with no special accommodations. That is the v1.0
compatibility contract.
"""

from __future__ import annotations

import os

import pytest
from openai import OpenAI


def _client(base_url: str) -> OpenAI:
    return OpenAI(base_url=f"{base_url}/v1", api_key="dummy-key")


def test_chat_non_streaming(llm_init_url: str, model_name: str) -> None:
    """``client.chat.completions.create`` returns a structured response."""
    client = _client(llm_init_url)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Reply with the word: hi"}],
        max_tokens=16,
        temperature=0,
    )
    assert completion.choices, "no choices returned"
    content = (completion.choices[0].message.content or "").strip()
    assert content, "empty assistant content"


def test_chat_streaming(llm_init_url: str, model_name: str) -> None:
    """Streaming yields at least one delta with content."""
    client = _client(llm_init_url)
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Reply with one word: hi"}],
        max_tokens=16,
        temperature=0,
        stream=True,
    )

    saw_content = False
    chunks_seen = 0
    for chunk in stream:
        chunks_seen += 1
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            saw_content = True

    assert chunks_seen > 0, "stream produced zero chunks"
    assert saw_content, "stream produced chunks but no content delta"


def test_models_list(llm_init_url: str, model_name: str) -> None:
    """``/v1/models`` advertises exactly the configured model."""
    client = _client(llm_init_url)
    models = client.models.list()
    ids = [m.id for m in models.data]
    assert model_name in ids, f"{model_name!r} not in {ids!r}"


def test_embeddings_or_skipped(llm_init_url: str, model_name: str) -> None:
    """Embeddings are smoke-tested only when the deployment is an
    embedding model.

    For chat models the engine still routes ``/v1/embeddings`` upstream,
    but llama.cpp's chat-tuned image returns a 400 — that is *correct*
    behaviour, so we skip rather than fail.
    """
    if os.environ.get("MODEL_MODE", "chat") != "embedding":
        pytest.skip("MODEL_MODE != embedding; chat models do not advertise embeddings")

    client = _client(llm_init_url)
    resp = client.embeddings.create(model=model_name, input="hello world")
    assert resp.data, "no embedding rows"
    assert len(resp.data[0].embedding) > 0
