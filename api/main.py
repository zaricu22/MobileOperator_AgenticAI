"""
Mobile Operator support API — wraps both agents behind a FastAPI HTTP
endpoint: llm/langgraph_agent.py (stateful graph, human-in-the-loop, the
default) and llm/langchain_agent.py (ReAct agent with tools/RAG), selected
per-request via ChatRequest.backend - same choice web/app.py's dropdown
offers, just over HTTP instead of a page.

This file does NOT redefine either agent — it imports the already-compiled
LangGraph `app` and langchain_agent's `build_agent_executor` from their own
modules and just adds a request/response layer on top. If either agent's
logic changes, it changes in ONE place; this file never needs to know how
answer_node/clarify_node or the ReAct loop work internally.

Note: the sensitive-request / human-approval path (request_refund_node,
interrupt(), Command(resume=...)) is intentionally NOT handled here — this
is the plain chat endpoint only. A refund-style question would currently
come back from the graph with "__interrupt__" instead of "answer"; see the
fallback below. langchain_agent.py has no equivalent concept at all - it has
no refund tool or approval step, so it'll just answer directly.

pip install fastapi uvicorn
uvicorn api.main:api --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

# These two are singletons: the graph is compiled once at import time, and
# build_agent_executor is the factory web/app.py also uses to hand each
# phone number its own AgentExecutor.
from llm.langgraph_agent import app as agent_graph
from llm.langchain_agent import build_agent_executor

# `api` (not `app`) to avoid shadowing the imported LangGraph `app` above —
# uvicorn is pointed at this name: `uvicorn api.main:api`
api = FastAPI(title="Mobile Operator Support API")

# One AgentExecutor per phone number, created lazily on first use - unlike
# agent_graph's MemorySaver checkpointer, AgentExecutor has no built-in
# per-thread storage, so this dict IS the session store (same pattern
# web/app.py uses for its own LangChain sessions). The two files keep
# SEPARATE dicts - a LangChain conversation started here won't carry over if
# continued through web/app.py, and vice versa.
langchain_sessions: dict[str, object] = {}


# ---- Request/response shapes ----
class ChatRequest(BaseModel):
    phone_number: str
    question: str
    backend: str = "langgraph"


class ChatResponse(BaseModel):
    answer: str
    retry_count: int


# ---- The actual endpoint ----
@api.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if request.backend == "langchain":
        answer = answer_via_langchain(request.phone_number, request.question)
        # retry_count is a LangGraph-specific concept (clarify-loop attempts) -
        # 0 here just means "not applicable", same as the refund fallback below.
        return ChatResponse(answer=answer, retry_count=0)
    return answer_via_langgraph(request.phone_number, request.question)


def answer_via_langgraph(phone_number: str, question: str) -> ChatResponse:
    # thread_id = phone_number: each customer's conversation gets its own
    # checkpointed thread, so the graph's MemorySaver keeps their last_answer
    # across separate HTTP requests exactly like it does across the separate
    # app.invoke() calls in llm/langgraph_agent.py's own __main__ demo.
    config = {"configurable": {"thread_id": phone_number}}

    result = agent_graph.invoke(
        {"phone_number": phone_number, "question": question, "retry_count": 0},
        config=config,
    )

    # fallback for the one case not handled in this file: a sensitive request
    # (e.g. "refund") pauses the graph and adds "__interrupt__" to the state -
    # checking "answer" instead would be wrong, since a paused turn still
    # carries over the PREVIOUS turn's "answer" from checkpointed state (it's
    # not removed, just not updated) - since we're not wiring up approval
    # here, surface something reasonable rather than that stale answer.
    if "__interrupt__" in result:
        return ChatResponse(answer="This request needs review by a support agent.", retry_count=0)

    return ChatResponse(answer=result["answer"], retry_count=result["retry_count"])


def answer_via_langchain(phone_number: str, question: str) -> str:
    if phone_number not in langchain_sessions:
        langchain_sessions[phone_number] = build_agent_executor()
    executor = langchain_sessions[phone_number]

    # get_account_usage (a tool the agent can call) takes phone_number as an
    # argument, but AgentExecutor only ever sees free-text "input" - so the
    # phone number has to be embedded in the question itself for the model to
    # pass it along when it decides to call that tool.
    result = executor.invoke({"input": f"(My phone number is {phone_number}.) {question}"})
    return result["output"]


# ---- Simple health check, handy for load balancers / uptime checks ----
@api.get("/health")
def health() -> dict:
    return {"status": "ok"}