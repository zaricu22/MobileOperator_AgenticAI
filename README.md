# MobileOperator_AgenticAI

Sample Python scripts demonstrating **Agentic AI patterns** with **LangChain** and **LangGraph**, using a mobile operator customer support scenario.

These examples progress from basic LangChain agent concepts to advanced LangGraph stateful systems, showing how LLMs can independently reason, call tools, and maintain conversation context.

---

## 📋 Overview

Two complementary implementations solve the same domain (mobile operator support queries):

| Feature | `mobile_operator_langchain.py` | `mobile_operator_langgraph.py` |
|---------|--------|---------|
| **Framework** | LangChain (`AgentExecutor`) | LangGraph (`StateGraph`) |
| **Agent Type** | ReAct loop (autonomous multi-step) | Stateful graph with conditional routing |
| **Memory** | ConversationBufferMemory | MemorySaver checkpointer (persistent) |
| **Tool Calling** | Full agent loop: picks tools, observes results, re-reasons | Limited: no dynamic tool calls shown |
| **Conversation** | Single multi-turn session in one `.invoke()` call | Turn-by-turn (separate `.invoke()` per message) |
| **Use Case** | Chatbot with complex reasoning and multi-tool orchestration | Real-world app: stateful user sessions across HTTP requests |
| **Complexity** | Intermediate (covers many concepts) | Advanced (graph-based state management) |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install langchain langchain-openai langchain-community faiss-cpu pypdf pydantic
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

### Run LangGraph Example

```bash
python mobile_operator_langgraph.py
```

Demonstrates:
- Graph-based state machine
- Conditional routing (branching logic)
- Loops/cycles within a turn
- Cross-turn memory via checkpointing

---

## 🔗 API Keys & Environment

```bash
export OPENAI_API_KEY=sk-...
```

Both scripts use `gpt-4o-mini` (free tier eligible). Swap the model name to use a different LLM.

---

## 📝 Fake Data

Both scripts use mocked account data instead of real databases:

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

### Both Scripts

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

### LangGraph-Only

- ✅ **Graph Nodes** (functions, state mutations)
- ✅ **Conditional Routing** (branching logic based on state)
- ✅ **Cycles** (loops within one turn)
- ✅ **Checkpointing** (state persistence across HTTP requests)
- ✅ **State Management** (TypedDict, immutable updates)

### Future Implementations

- 🔨 **Streaming** (.stream, .astream)
- 🔨 **Async** (.ainvoke)
- 🔨 **Human-in-the-Loop** (interrupts)
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

---

## 🎯 Use Cases

### LangChain Example (`mobile_operator_langchain.py`)

- **Customer support chatbot** with multi-turn problem solving
- **Document Q&A** with real PDF retrieval
- **Knowledge base search** with diverse results (MMR)
- **Autonomous tool orchestration** (model decides what to call)

### LangGraph Example (`mobile_operator_langgraph.py`)

- **Session-based web app** (stateful across HTTP requests)
- **User thread management** (one thread per user ID)
- **Retry/clarification loops** (internal to one turn)
- **Real-world customer support system** (matches typical architecture)

---

## 🚀 Next Steps

1. **Add real tools:** Replace FAKE_ACCOUNTS with actual database calls
2. **Integrate into web app:** Wrap LangGraph in FastAPI/Flask
3. **Add vector search:** Ingest real customer policies, not hardcoded strings
4. **Stream responses:** Use `.stream()` for real-time feedback
5. **Add human approval:** Use LangGraph interrupts for sensitive operations
6. **Monitor & trace:** Use LangSmith to debug agent behavior

---

## MIT License
