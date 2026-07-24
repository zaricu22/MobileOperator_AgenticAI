# MobileOperator_AgenticAI

Sample Python scripts demonstrating **Agentic AI patterns** with **LangChain** and **LangGraph**, using a mobile operator customer support scenario.

These examples progress from basic LangChain agent concepts to advanced LangGraph stateful systems, showing how LLMs can independently reason, call tools, and maintain conversation context.

---

## 📋 Overview

Three complementary implementations solve the same domain (mobile operator support queries):

| Feature | `mobile_operator_langchain.py` | `mobile_operator_langgraph.py` | `mobile_operator_api.py` |
|---------|--------|---------|---------|
| **Framework** | LangChain (`AgentExecutor`) | LangGraph (`StateGraph`) | FastAPI + LangGraph |
| **Agent Type** | ReAct loop (autonomous multi-step) | Stateful graph with conditional routing | HTTP API wrapping LangGraph |
| **Memory** | ConversationBufferMemory | MemorySaver checkpointer (persistent) | MemorySaver (via LangGraph) |
| **Tool Calling** | Full agent loop: picks tools, observes results, re-reasons | Limited: no dynamic tool calls shown | N/A (delegates to LangGraph) |
| **Conversation** | Single multi-turn session in one `.invoke()` call | Turn-by-turn (separate `.invoke()` per message) | Turn-by-turn HTTP requests |
| **Human-in-Loop** | Not implemented | ✅ Implemented (interrupt/resume for sensitive actions) | ✅ Supported (via LangGraph) |
| **Use Case** | Chatbot with complex reasoning and multi-tool orchestration | Real-world app: stateful user sessions across invocations | Production web service: stateful users across HTTP requests |
| **Complexity** | Intermediate (covers many concepts) | Advanced (graph-based state management + interrupts) | Production-ready (web API layer) |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install langchain langchain-openai langchain-community langgraph faiss-cpu pypdf pydantic fastapi uvicorn
export OPENAI_API_KEY=sk-...
```

For PDF example: ensure `plans_handbook.pdf` exists in the working directory.

### Run LangChain Example

```bash
python mobile_operator_langchain.py
```

Demonstrates:
- JSON output parsing (structured classification)
- Tool calling (single-step, manual)
- ReAct agent loop (autonomous multi-step)
- RAG (knowledge base + PDF retrieval)

### Run LangGraph Example (Standalone)

```bash
python mobile_operator_langgraph.py
```

Demonstrates:
- Graph-based state machine
- Conditional routing (branching logic)
- Loops/cycles within a turn
- Cross-turn memory via checkpointing
- **Human-in-the-loop:** interrupt/resume for sensitive operations (e.g., refunds)

### Run LangGraph via HTTP API

```bash
# Terminal 1: Start the FastAPI server
uvicorn mobile_operator_api:api --reload

# Terminal 2: Test the endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "0641234567", "question": "What plan am I on?"}'
```

The API exposes:
- **POST `/chat`** — stateful chat endpoint (phone_number = thread_id)
- **GET `/health`** — health check for load balancers

---

## 🔗 API Keys & Environment

```bash
export OPENAI_API_KEY=sk-...
```

All scripts use `gpt-4o-mini` (free tier eligible). Swap the model name to use a different LLM.

---

## 📝 Fake Data

All scripts use mocked account data instead of real databases:

```python
FAKE_ACCOUNTS = {
    "0641234567": {
        "plan": "Plan M",
        "data_left_gb": 3.2,
        "roaming_included": "EU only, 10GB cap"
    }
}
```

Replace with real calls (SQL DB, REST API, gRPC) to ground queries in actual user data.

---

## 🛠️ Concepts Covered

### All Scripts

- ✅ **LLM** (ChatOpenAI)
- ✅ **Prompt Engineering** (PromptTemplate, structured formats)
- ✅ **LangChain** (LCEL pipes: `prompt | llm | parser`)
- ✅ **Context Engineering** (injecting facts into prompts)
- ✅ **Context-Augmented Generation** (account info, format instructions)

### LangChain-Only

- ✅ **Vector Database** (FAISS)
- ✅ **RAG** (retriever, embeddings, PDF ingestion)
- ✅ **Tool System** (@tool, bind_tools, tool calling)
- ✅ **Agent Loop** (ReAct: Thought/Action/Observation)
- ✅ **Memory** (ConversationBufferMemory)
- ✅ **JSON Parsing** (JsonOutputParser, validation)

### LangGraph & API

- ✅ **Graph Nodes** (functions, state mutations)
- ✅ **Conditional Routing** (branching logic based on state)
- ✅ **Cycles** (loops within one turn)
- ✅ **Checkpointing** (state persistence across invocations / HTTP requests)
- ✅ **State Management** (TypedDict, immutable updates)
- ✅ **Human-in-the-Loop** (`interrupt()` pauses sensitive operations, `Command(resume=...)` approves/denies)
- ✅ **Thread Management** (phone_number as thread_id for per-user sessions)

### Future Implementations

- 🔨 **Streaming** (.stream, .astream)
- 🔨 **Async** (.ainvoke)
- 🔨 **Parallel Execution** (fan-out, Send)
- 🔨 **Subgraphs** (nested graphs)
- 🔨 **ToolNode** (integrated tool execution)
- 🔨 **Time Travel** (state history replay)

---

## 🔑 Key Architectural Differences

### LangChain (One Big Session)

```python
# All turns in one call
memory = ConversationBufferMemory()
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# Turn 1 + Turn 2 + Turn 3 in sequence, same memory object
r1 = executor.invoke({"input": "..."})
r2 = executor.invoke({"input": "..."})  # memory carries turn 1 automatically
```

**Pros:** Simple, memory implicit  
**Cons:** Doesn't match request/response HTTP model

### LangGraph (Turn-by-Turn, Checkpointed)

```python
# State persisted across calls
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "user123"}}

# Turn 1
r1 = app.invoke({"input": "..."}, config=config)
# ... HTTP response sent, user types next message ...
# Turn 2
r2 = app.invoke({"input": "..."}, config=config)  # checkpointer reloads state
```

**Pros:** Matches stateful web app architecture  
**Cons:** Requires explicit thread_id management

### LangGraph with Human-in-the-Loop

```python
# Sensitive operations can pause and wait for human approval
config = {"configurable": {"thread_id": "user123"}}

# Request triggers an interrupt()
result = app.invoke({"input": "refund my plan"}, config=config)
if "__interrupt__" in result:
    # A human reviews the request...
    result = app.invoke(Command(resume=True), config=config)  # Same thread_id
```

**Pros:** Real-world requirement for sensitive operations; checkpointer keeps state while paused  
**Cons:** Requires external mechanism (agent UI, approval queue) to resume

### FastAPI + LangGraph (Production Web Service)

```python
@api.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": request.phone_number}}
    result = agent_graph.invoke({"phone_number": ..., "question": ...}, config=config)
    return ChatResponse(answer=result["answer"], retry_count=result["retry_count"])
```

**Pros:** Stateless HTTP layer (thread_id manages state), scalable, standard JSON API  
**Cons:** API layer must handle sensitive operation results (e.g., `__interrupt__` pauses)

---

## 🎯 Use Cases

### LangChain Example (`mobile_operator_langchain.py`)

- **Customer support chatbot** with multi-turn problem solving
- **Document Q&A** with real PDF retrieval
- **Knowledge base search** with diverse results (MMR)
- **Autonomous tool orchestration** (model decides what to call)

### LangGraph Example (`mobile_operator_langgraph.py`)

- **Session-based application** (stateful across invocations)
- **User thread management** (one thread per user ID)
- **Retry/clarification loops** (internal to one turn)
- **Human approval workflows** (interrupt before sensitive actions)
- **Real-world customer support system** (matches typical architecture)

### FastAPI Example (`mobile_operator_api.py`)

- **Production HTTP API** serving the LangGraph agent
- **Per-user stateful sessions** (thread_id = phone_number)
- **Scalable multi-user support** (checkpointer handles state, not the API)
- **Integration point** for external approval systems (human-in-the-loop)

---

## 🚀 Next Steps

1. **Add real tools:** Replace FAKE_ACCOUNTS with actual database calls
2. **Add vector search:** Ingest real customer policies, not hardcoded strings
3. **Stream responses:** Use `.stream()` for real-time feedback
4. **Monitor & trace:** Use LangSmith to debug agent behavior
5. **Deploy the API:** Use Docker + a production ASGI server (Gunicorn + Uvicorn)
6. **Wire up human approval:** Build an admin UI or approval queue for sensitive operations (paused via `interrupt()`)
