"""OpenWebUI alias smoke test.

OpenWebUI hard-codes ``/api/chat/completions`` (rather than the
canonical ``/v1/chat/completions``). llm-init exposes the exact same
handler under that path; this test verifies the alias works with a
streamed response, since OpenWebUI relies on SSE.
"""

from __future__ import annotations

import json

import httpx


def test_openwebui_alias_streams_sse(llm_init_url: str, model_name: str) -> None:
    body = {
        "model": model_name,
        "stream": True,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with one word: hi"}],
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    saw_data_frame = False
    saw_done = False
    with httpx.stream(
        "POST",
        f"{llm_init_url}/api/chat/completions",
        json=body,
        headers=headers,
        timeout=120,
    ) as resp:
        assert resp.status_code == 200, (
            f"alias returned {resp.status_code}: {resp.read()!r}"
        )
        for line in resp.iter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload == "[DONE]":
                saw_done = True
                break
            saw_data_frame = True
            json.loads(payload)

    assert saw_data_frame, "alias produced no data: frames"
    assert saw_done, "alias produced data: frames but no terminal [DONE]"
