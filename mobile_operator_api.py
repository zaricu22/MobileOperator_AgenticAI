"""
Mobile Operator support API — wraps the compiled LangGraph agent from
mobile_operator_langgraph.py behind a FastAPI HTTP endpoint.

This file does NOT redefine the graph — it imports the already-compiled
`app` (the graph) from the other script and just adds a request/response
layer on top. If the graph logic changes, it changes in ONE place
(mobile_operator_langgraph.py); this file never needs to know how
answer_node/clarify_node work internally.

Note: the sensitive-request / human-approval path (request_refund_node,
interrupt(), Command(resume=...)) is intentionally NOT handled here — this
is the plain chat endpoint only. A refund-style question would currently
come back from the graph without an "answer" key; see the fallback below.

pip install fastapi uvicorn
uvicorn mobile_operator_api:api --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from mobile_operator_langgraph import app as agent_graph

# `api` (not `app`) to avoid shadowing the imported LangGraph `app` above —
# uvicorn is pointed at this name: `uvicorn mobile_operator_api:api`
api = FastAPI(title="Mobile Operator Support API")


# ---- Request/response shapes ----
class ChatRequest(BaseModel):
    phone_number: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    retry_count: int


# ---- The actual endpoint ----
@api.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # thread_id = phone_number: each customer's conversation gets its own
    # checkpointed thread, so the graph's MemorySaver keeps their last_answer
    # across separate HTTP requests exactly like it does across the separate
    # app.invoke() calls in mobile_operator_langgraph.py's own __main__ demo.
    config = {"configurable": {"thread_id": request.phone_number}}

    result = agent_graph.invoke(
        {
            "phone_number": request.phone_number,
            "question": request.question,
            "retry_count": 0,
        },
        config=config,
    )

    # fallback for the one case not handled in this file: a sensitive request
    # (e.g. "refund") would pause the graph and return "__interrupt__" instead
    # of "answer" — since we're not wiring up approval here, surface something
    # reasonable rather than a raw KeyError.
    if "answer" not in result:
        return ChatResponse(answer="This request needs review by a support agent.", retry_count=0)

    return ChatResponse(answer=result["answer"], retry_count=result["retry_count"])


# ---- Simple health check, handy for load balancers / uptime checks ----
@api.get("/health")
def health() -> dict:
    return {"status": "ok"}