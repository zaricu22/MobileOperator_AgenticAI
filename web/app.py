"""
Mobile Operator support chat — simple Flask web page over both agents:
llm/langgraph_agent.py (stateful graph, human-in-the-loop) and
llm/langchain_agent.py (ReAct agent with tools/RAG). A dropdown above the
chat log picks which one answers each message.

Standalone, single-process demo: this file imports both already-built agents
(the compiled LangGraph `app` api/main.py also wraps, and langchain_agent's
`build_agent_executor`) and serves both a chat page and its own /chat route,
so nothing else needs to be running.

Why standalone instead of a Flask page in front of the existing FastAPI service:
fronting api/main.py would mean running two servers (uvicorn + flask)
and adding CORS just to let the browser call a different origin -
extra moving parts with no real benefit for a single demo page. Importing
the compiled `agent_graph` directly costs nothing (it's the exact same
object api/main.py uses) and keeps "run one command, open one page" true.

FastAPI vs Flask:
- FastAPI is API-first: it excels at JSON endpoints, auto-generates OpenAPI/Swagger docs, and gives you async + Pydantic auto-mapping/validation (via Request/Response model). 
It can render HTML (via Jinja2Templates/StaticFiles), but you have to wire/link that up yourself (no /template folder auto-scan like Flask does),
also 'request' parameter must be defined and passed to the template response (boilerplate code).
- Flask is the opposite default: Jinja2 templating and render_template() are built in from the start, no boilerplate syntax for simple rendering (returning HTML pages). 
It's equally capable of being a pure JSON API (jsonify() everywhere) with no HTML at all,
but it does not provide auto-mapping/validation (lack Request/Response model) or OpenAPI docs as FastAPI does.

pip install flask
$env:GROQ_API_KEY = "gsk_..."
python -m web.app
"""

from flask import Flask, jsonify, render_template, request

# This WEB script is wrapper around the compiled graph object.
# The graph is compiled once, at import time, with whole this script,
# and become a module-level singleton for this Python process.
from llm.langgraph_agent import app as agent_graph
from llm.langchain_agent import build_agent_executor

flask_app = Flask(__name__)

# One AgentExecutor per phone number, created lazily on first use - unlike
# agent_graph's MemorySaver checkpointer, AgentExecutor has no built-in
# per-thread storage, so this dict IS the session store, keyed the same way
# (by phone_number) to keep the two backends' behavior comparable.
langchain_sessions: dict[str, object] = {}


@flask_app.get("/")
def index():
    return render_template("chat.html")


@flask_app.post("/chat")
def chat():
    data = request.get_json(force=True)
    phone_number = data["phone_number"]
    question = data["question"]
    backend = data.get("backend", "langgraph")

    if backend == "langchain":
        answer = answer_via_langchain(phone_number, question)
    else:
        answer = answer_via_langgraph(phone_number, question)

    return jsonify(answer=answer)


def answer_via_langgraph(phone_number: str, question: str) -> str:
    # thread_id = phone_number: same per-user session pattern as
    # api/main.py, so the checkpointer keeps last_answer across
    # separate requests/page turns for this phone number.
    config = {"configurable": {"thread_id": phone_number}}

    result = agent_graph.invoke(
        {"phone_number": phone_number, "question": question, "retry_count": 0},
        config=config,
    )

    # Sensitive requests (e.g. "refund") pause the graph via interrupt() and
    # add "__interrupt__" to the state - checking "answer" instead would be
    # wrong, since a paused turn still carries over the PREVIOUS turn's
    # "answer" from checkpointed state (it's not removed, just not updated).
    # This page doesn't implement the human-approval resume flow, so surface
    # a safe message instead.
    if "__interrupt__" in result:
        return "This request needs review by a support agent."
    return result["answer"]


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


if __name__ == "__main__":
    flask_app.run(debug=True)
