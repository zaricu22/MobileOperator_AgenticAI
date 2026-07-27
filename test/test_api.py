"""
Integration tests for POST /chat (api/main.py).

These mirror the four use cases exercised in llm/langgraph_agent.py's
own __main__ demo, but driven through the FastAPI layer instead of calling
app.invoke() directly, since that's the surface real callers actually hit:

  1. Turn 1 - a brand-new conversation for a thread_id (no prior state).
  2. Turn 2 - a follow-up on the SAME thread_id (separate HTTP request),
     resolved using the previous turn's answer (prior state) via the MemorySaver
     checkpointer - the "memory across separate invoke() calls" behavior.
  3. Turn 3 - a question the model can't confidently answer, driving the
     clarify -> answer cycle until MAX_RETRIES forces it to stop.
  4. Turn 4 - a sensitive ("refund") request that hits request_refund_node's
     interrupt(). This API layer never calls Command(resume=...), so it must
     fall back to a safe response instead of exposing "__interrupt__" or
     raising a KeyError on the missing "answer" field.

Integration tests execution notes:
    - These tests run against the FastAPI app in api/main.py, 
    which imports the compiled LangGraph agent from llm/langgraph_agent.py.
    Only the ChatOpenAI `llm` call inside answer_node is faked.
    Every other code part of langgraph_agent (routing, the clarify cycle, the checkpointer)
    runs for real, not faked.

    - Not covered here: request_refund_node's post-interrupt() approve/deny
    branch. interrupt() suspends execution right there, and the function never
    resumes because api/main.py never calls Command(resume=...) - see
    test_langgraph_agent.py for that.

pip install pytest httpx
pytest test/test_api.py
"""
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import api.main as api_module
import llm.langgraph_agent as graph_module
from api.main import api

client = TestClient(api)


class FakeExecutor:
    """Stand-in for AgentExecutor, used by the backend="langchain" tests
    below. llm/langchain_agent.py builds its `agent` runnable at IMPORT TIME
    with the real LLM baked in, so - unlike graph_module.llm above - there's
    no module attribute to monkeypatch afterward that would actually be seen
    by a real AgentExecutor. Faking the executor itself instead tests what
    api/main.py actually owns: backend dispatch and per-phone-number session
    reuse/isolation, not the ReAct loop internals (see test_web_app.py for
    the same approach against web/app.py)."""

    def __init__(self):
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs["input"])
        return {"output": f"fake langchain answer #{len(self.calls)}"}


# Whenever a test reaches `llm` call during real execution of the compiled graph,
# it returns provided response instead of hitting real OpenAI -
# everything else part of the graph still runs as-is.
#
# CAUTION: The fake LLM totally ignores its input (provided args). Because of
# that, it cannot test whether the right context is sent to the LLM - it just
# verifies graph routing/state. Tests can easily fail to recognize a bad/
# incomplete LLM input context.
def set_llm_responses(monkeypatch, responses):
    """Replace the graph's shared `llm` with a fake that returns `responses`
    in order (cycling if exhausted). answer_node/clarify_node look up `llm`
    as a module global at call time, so patching the attribute is enough -
    no need to touch the compiled graph itself."""
    monkeypatch.setattr(graph_module, "llm", FakeListChatModel(responses=responses))


def test_turn1_new_conversation_returns_answer(monkeypatch):
    set_llm_responses(monkeypatch, ["You're on Plan M with 3.2GB of data left."])

    response = client.post(
        "/chat",
        json={"phone_number": "0641234567", "question": "What plan am I on?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "You're on Plan M with 3.2GB of data left."
    assert body["retry_count"] == 0


def test_turn2_followup_uses_checkpointed_last_answer(monkeypatch):
    phone_number = "0641234568"
    set_llm_responses(
        monkeypatch,
        [
            "You're on Plan M with 3.2GB of data left.",
            "Yes, your plan includes EU roaming with a 10GB cap.",
        ],
    )

    turn1 = client.post(
        "/chat",
        json={"phone_number": phone_number, "question": "What plan am I on?"},
    )
    assert turn1.status_code == 200

    # Separate HTTP request, same thread_id. We deliberately don't send any
    # prior answer - the checkpointer must supply "last_answer" so "it" in
    # this question resolves against turn 1's answer.
    turn2 = client.post(
        "/chat",
        json={"phone_number": phone_number, "question": "Does it cover roaming?"},
    )

    assert turn2.status_code == 200
    body = turn2.json()
    assert body["answer"] == "Yes, your plan includes EU roaming with a 10GB cap."
    assert body["retry_count"] == 0


def test_turn3_uncertain_answer_triggers_clarify_loop(monkeypatch):
    phone_number = "0641234569"
    set_llm_responses(
        monkeypatch,
        [
            "I'm not sure about that.",
            "Still unclear from the info I have.",
            "Yes, this plan supports 5G roaming in Japan.",
        ],
    )

    response = client.post(
        "/chat",
        json={
            "phone_number": phone_number,
            "question": "Does it support 5G roaming in Japan?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Yes, this plan supports 5G roaming in Japan."
    # Two clarify loops fired (retry_count 0->1->2) before route_after_answer's
    # `retry_count < MAX_RETRIES` check forced "end" regardless of confidence.
    assert body["retry_count"] == 2


def test_turn4_sensitive_refund_request_is_not_auto_approved(monkeypatch):
    # request_refund_node never calls the LLM (it only calls interrupt()),
    # but patching it out keeps this test from ever touching a real model
    # if that ever changes.
    set_llm_responses(monkeypatch, ["unused"])

    response = client.post(
        "/chat",
        json={
            "phone_number": "0641234570",
            "question": "I want a refund for last month",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "This request needs review by a support agent."
    assert body["retry_count"] == 0


def test_langchain_backend_embeds_phone_number_and_returns_retry_count_zero(monkeypatch):
    monkeypatch.setattr(api_module, "langchain_sessions", {})
    monkeypatch.setattr(api_module, "build_agent_executor", FakeExecutor)
    phone_number = "0641234571"

    response = client.post(
        "/chat",
        json={
            "phone_number": phone_number,
            "question": "How much data do I have left?",
            "backend": "langchain",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake langchain answer #1"
    # retry_count is a LangGraph-only concept - 0 just means "not applicable".
    assert body["retry_count"] == 0
    # get_account_usage (a tool the agent can call) needs phone_number, but
    # AgentExecutor only ever sees free-text "input" - this is the only way
    # to prove the tool CAN actually be called with the right number.
    executor = api_module.langchain_sessions[phone_number]
    assert executor.calls == [f"(My phone number is {phone_number}.) How much data do I have left?"]


def test_langchain_backend_reuses_session_across_separate_requests(monkeypatch):
    monkeypatch.setattr(api_module, "langchain_sessions", {})
    monkeypatch.setattr(api_module, "build_agent_executor", FakeExecutor)
    phone_number = "0641234572"

    client.post(
        "/chat",
        json={"phone_number": phone_number, "question": "What's in Plan M?", "backend": "langchain"},
    )
    client.post(
        "/chat",
        json={"phone_number": phone_number, "question": "Does it cover roaming?", "backend": "langchain"},
    )

    # Exactly ONE executor was ever created for this phone number - proves
    # the second request reused the first's AgentExecutor (and therefore its
    # ConversationBufferMemory) instead of losing history every request.
    assert len(api_module.langchain_sessions) == 1
    assert len(api_module.langchain_sessions[phone_number].calls) == 2
