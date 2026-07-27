"""
Direct tests against the compiled LangGraph agent (llm/langgraph_agent.py),
bypassing the FastAPI layer entirely.

Integration tests in test/test_api.py already exercises this graph through POST /chat,
this unit-test file is for additional uncovered aspects of the graph's behavior 
missing from the API layer tests.
As before, unit-tests also fake the real ChatOpenAI LLM with FakeListChatModel, 
so these run deterministically, offline, and without an OPENAI_API_KEY.

Only the paths api/main.py actually missing - it never calls
Command(resume=...), and it only checks the final answer/retry_count JSON,
which can't tell "the checkpointer correctly reloaded last_answer" apart
from "the fake LLM ignored its input and returned the next determined string anyway." 
These tests call agent_graph.invoke()/get_state() directly, so they can cover:

  - The FULL human-in-the-loop round trip: interrupt() pausing, then
    Command(resume=...) actually resuming with an approval or a denial -
    api/main.py doesn't implement this half at all.
  - Checkpointing: state persisted between two separate invoke() calls with
    the same thread_id, verified via agent_graph.get_state() (the actual
    persisted channel values), not inferred from output text - and
    isolation between different thread_ids.

pip install pytest
pytest test/test_langgraph_agent.py
"""
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.types import Command

import llm.langgraph_agent as graph_module

agent_graph = graph_module.app

# Whenever a test reaches `llm` call during real execution of the compiled graph,
# it returns provided response instead of hitting real OpenAI -
# everything else part of the graph still runs as-is.
#
# CAUTION: The fake LLM totally ignores its input (provided args). Because of
# that, it cannot test whether the right context is sent to the LLM - it just
# verifies graph routing/state. Tests can easily fail to recognize a bad/
# incomplete LLM input context.
def set_llm_responses(monkeypatch, responses):
    monkeypatch.setattr(graph_module, "llm", FakeListChatModel(responses=responses))


def test_refund_approved_after_human_resumes(monkeypatch):
    set_llm_responses(monkeypatch, ["unused"])
    config = {"configurable": {"thread_id": "lg-refund-approve"}}
    agent_graph.invoke(
        {"phone_number": "0641234567", "question": "I want a refund", "retry_count": 0},
        config=config,
    )

    # Same thread_id, no new question - this is the support-agent-approves
    # action from an approval queue/admin UI, not a new user message.
    result = agent_graph.invoke(Command(resume=True), config=config)

    assert result["answer"] == "Your refund has been approved and will be processed within 5 business days."


def test_refund_denied_after_human_resumes(monkeypatch):
    set_llm_responses(monkeypatch, ["unused"])
    config = {"configurable": {"thread_id": "lg-refund-deny"}}
    agent_graph.invoke(
        {"phone_number": "0641234567", "question": "I want a refund", "retry_count": 0},
        config=config,
    )

    result = agent_graph.invoke(Command(resume=False), config=config)

    assert result["answer"] == "Your refund request needs further review and has not been approved yet."


def test_checkpointer_persists_last_answer_across_separate_invoke_calls(monkeypatch):
    config = {"configurable": {"thread_id": "lg-memory-thread"}}
    set_llm_responses(
        monkeypatch,
        ["You're on Plan M with 3.2GB of data left.", "Yes, EU roaming is included, 10GB cap."],
    )

    agent_graph.invoke(
        {"phone_number": "0641234567", "question": "What plan am I on?", "retry_count": 0},
        config=config,
    )
    # Asserting on the checkpointer's persisted state directly - not just on
    # the next call's output - is what actually proves the checkpointer (and
    # not the caller) is the one supplying last_answer to turn 2.
    assert agent_graph.get_state(config).values["last_answer"] == "You're on Plan M with 3.2GB of data left."

    agent_graph.invoke(
        {"phone_number": "0641234567", "question": "Does it cover roaming?", "retry_count": 0},
        config=config,
    )
    assert agent_graph.get_state(config).values["last_answer"] == "Yes, EU roaming is included, 10GB cap."


def test_checkpointer_isolates_state_between_different_thread_ids(monkeypatch):
    set_llm_responses(monkeypatch, ["answer for thread A", "answer for thread B"])
    config_a = {"configurable": {"thread_id": "lg-thread-a"}}
    config_b = {"configurable": {"thread_id": "lg-thread-b"}}

    agent_graph.invoke({"phone_number": "0641234567", "question": "q1", "retry_count": 0}, config=config_a)
    agent_graph.invoke({"phone_number": "0641234567", "question": "q2", "retry_count": 0}, config=config_b)

    assert agent_graph.get_state(config_a).values["last_answer"] == "answer for thread A"
    assert agent_graph.get_state(config_b).values["last_answer"] == "answer for thread B"
