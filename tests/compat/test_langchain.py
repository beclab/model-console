"""LangChain compatibility smoke test.

Confirms that ``langchain_openai.ChatOpenAI`` works against llm-init
with the same ``base_url`` override pattern users follow with the
upstream OpenAI service.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


def test_langchain_chatopenai_invoke(llm_init_url: str, model_name: str) -> None:
    chat = ChatOpenAI(
        base_url=f"{llm_init_url}/v1",
        api_key="dummy-key",
        model=model_name,
        temperature=0,
        max_tokens=16,
        timeout=120,
    )
    response = chat.invoke([HumanMessage(content="Reply with one word: hi")])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str) and response.content.strip(), (
        "ChatOpenAI returned empty content"
    )
