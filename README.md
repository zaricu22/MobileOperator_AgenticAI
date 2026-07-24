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

## 📚 Script Details

### 1️⃣ `mobile_operator_langchain.py` — ReAct Agent + RAG

**Best for:** Understanding autonomous agent reasoning and tool orchestration.

#### Key Concepts

| Concept | Implementation |
|---------|---------|
| **LLM** | `ChatOpenAI("gpt-4o-mini")` — powers classification, tool selection, reasoning |
| **Prompt Engineering** | `extract_prompt` (forces JSON) + `react_prompt` (forces Thought/Action/Observation format) |
| **Vector Database** | FAISS: two indexes (`vector_db` from hardcoded KB, `pdf_vector_db` from real PDF) |
| **RAG** | `search_plans_kb` tool wraps MMR retriever; uses diversity + score thresholds |
| **Tool System** | `@tool` decorator + `llm.bind_tools()` → model chooses tools, AgentExecutor loops |
| **Memory** | `ConversationBufferMemory` persists chat history across agent calls |
| **Routing** | ReAct loop: Thought → Action → Observation → re-reason until answer |

#### Code Structure

```python
# Section 1: Knowledge Base (hardcoded strings + MMR retriever)
kb_chunks = [...]
retriever_diverse = vector_db.as_retriever(search_type="mmr", ...)

# Section 2: PDF Ingestion (real document → pages → chunks → vectors)
loader = PyPDFLoader("plans_handbook.pdf")
chunks = splitter.split_documents(raw_docs)
pdf_retriever = FAISS.from_documents(chunks, embeddings).as_retriever()

# Section 3: Tools (@tool functions the agent can call)
@tool
def get_account_usage(phone_number: str) -> str: ...
@tool
def search_plans_kb(query: str) -> str: ...

# Section 4: JSON Output Parser (force structured output)
json_parser = JsonOutputParser(pydantic_object=QueryInfo)
extract_chain = extract_prompt | llm | json_parser

# Section 5: Memory (carry chat history)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Section 6: ReAct Agent Loop (bind tools, create agent, execute)
llm_with_tools = llm.bind_tools(tools)
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# Section 7: Run multi-turn conversation
r1 = agent_executor.invoke({"input": "What's in Plan M?"})
r2 = agent_executor.invoke({"input": "Does it cover roaming in the EU?"})
```

#### What Happens When You Call `agent_executor.invoke()`

1. **LLM sees:** tools list, chat history, current question
2. **LLM produces:** `Thought: ..., Action: search_plans_kb, Action Input: ...`
3. **Agent executor parses** the Action + Input, calls the tool
4. **LLM sees:** previous Thought/Action, tool result, is asked to continue
5. **Repeats:** until LLM says `Thought: I now know the answer, Final Answer: ...`

#### Example Output

```
Thought: I need to find information about Plan M
Action: search_plans_kb
Action Input: Plan M details
Observation: [Retrieved from KB]
Thought: I have enough information
Final Answer: Plan M includes 15GB data...
```

#### Retriever Variants Shown

- **Plain similarity:** always returns k results (may be near-duplicates)
- **MMR (Maximal Marginal Relevance):** diverse results, not just closest matches
- **Score threshold:** drops low-confidence matches
- **Metadata filtering:** scopes search to one KB category

---

### 2️⃣ `mobile_operator_langgraph.py` — Stateful Graph + Checkpointing

**Best for:** Real-world applications with persistent user sessions and turn-by-turn interactions.

#### Key Concepts

| Concept | Implementation |
|---------|---------|
| **Graph** | `StateGraph(AgentState)` — nodes are functions, edges define flow |
| **State** | `AgentState` TypedDict: holds phone_number, question, last_answer, answer, retry_count |
| **Nodes** | `answer_node` (generates response), `clarify_node` (enriches question for retry) |
| **Conditional Routing** | `route_after_answer()` decides: loop back for clarification, or end |
| **Cycles/Loops** | `graph.add_edge("clarify", "answer")` — within a single `.invoke()` call |
| **Checkpointing** | `MemorySaver` + `thread_id` — persists state **across separate `.invoke()` calls** |
| **Prompt Variants** | Two prompts in one node, chosen based on whether `last_answer` exists |

#### Code Structure

```python
# Section 1: State object (carries conversation + working memory)
class AgentState(TypedDict):
    phone_number: str
    question: str
    last_answer: Optional[str]  # from previous turn
    answer: Optional[str]        # this turn's result
    retry_count: int

# Section 2: Prompts (different strategies for turn 1 vs follow-up)
plan_prompt = ChatPromptTemplate.from_template("...")      # turn 1
followup_prompt = ChatPromptTemplate.from_template("...")  # turn 2+

# Section 3: Nodes (functions that mutate state)
def answer_node(state: AgentState) -> AgentState:
    # Picks prompt based on last_answer
    # Runs LLM
    # Returns updated state
    
def clarify_node(state: AgentState) -> AgentState:
    # Enriches question with account data
    # Increments retry_count
    # Returns updated state

# Section 4: Router (conditional logic)
def route_after_answer(state: AgentState) -> str:
    # Inspects answer for uncertainty phrases
    # Returns "clarify" or "end"

# Section 5: Build graph
graph = StateGraph(AgentState)
graph.add_node("answer", answer_node)
graph.add_node("clarify", clarify_node)
graph.add_edge(START, "answer")
graph.add_conditional_edges("answer", route_after_answer, {"clarify": "clarify", "end": END})
graph.add_edge("clarify", "answer")  # CYCLE

# Section 6: Checkpointer (bridges invoke() calls)
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Section 7: Turn-by-turn invocation
turn1 = app.invoke(
    {"phone_number": "0641234567", "question": "What plan am I on?", "retry_count": 0},
    config={"configurable": {"thread_id": "0641234567"}}
)
turn2 = app.invoke(
    {"phone_number": "0641234567", "question": "Does it cover roaming?", "retry_count": 0},
    config={"configurable": {"thread_id": "0641234567"}}
)
# turn2's answer_node sees turn1's last_answer automatically (from checkpointer)
```

#### Execution Flow

```
Turn 1: app.invoke({"question": "What plan am I on?", ...})
  START → answer_node (plan_prompt, no context)
  → route_after_answer (is answer uncertain?)
  → ["clarify" or "end"]
  → checkpointer persists state

Turn 2: app.invoke({"question": "Does it cover roaming?", ...})
  checkpointer RELOADS turn1's state
  START → answer_node (followup_prompt, last_answer from turn1)
  → route_after_answer (is answer uncertain?)
  → ["clarify" or "end"]
  → checkpointer persists state
  → answer resolves "it" using last_answer
```

#### Clarification Loop (within one turn)

If the LLM's answer contains phrases like *"I don't know"*, and `retry_count < MAX_RETRIES`:

```
Turn 3: app.invoke({"question": "Does it support 5G in Japan?", ...})
  answer_node → answer is uncertain
  → route_after_answer returns "clarify"
  → clarify_node adds plan info: "...my plan is Plan M"
  → graph.add_edge("clarify", "answer")  # LOOP BACK
  → answer_node runs again, with enriched question
  → route_after_answer checks again
  → (repeat until confident or MAX_RETRIES)
```

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

### Not Covered

- ❌ **Streaming** (.stream, .astream)
- ❌ **Async** (.ainvoke)
- ❌ **Human-in-the-Loop** (interrupts)
- ❌ **Parallel Execution** (fan-out, Send)
- ❌ **Subgraphs** (nested graphs)
- ❌ **ToolNode** (integrated tool execution)
- ❌ **Time Travel** (state history replay)

---

## 📖 Learning Path

1. **Start with LangChain** (`mobile_operator_langchain.py`):
   - Understand ReAct agent reasoning
   - See how tools are called autonomously
   - Learn RAG + vector store concepts
   - Grasp LCEL pipe syntax

2. **Move to LangGraph** (`mobile_operator_langgraph.py`):
   - Understand graph-based state machines
   - See how state flows between nodes
   - Learn conditional routing + cycles
   - Grasp checkpointing for persistent sessions

3. **Compare the two**:
   - Both solve "help a customer" but with different tradeoffs
   - LangChain prioritizes agent autonomy; LangGraph prioritizes state control
   - In production: often use both (agent inside a LangGraph node)

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

## 🚀 Next Steps

1. **Add real tools:** Replace FAKE_ACCOUNTS with actual database calls
2. **Integrate into web app:** Wrap LangGraph in FastAPI/Flask
3. **Add vector search:** Ingest real customer policies, not hardcoded strings
4. **Stream responses:** Use `.stream()` for real-time feedback
5. **Add human approval:** Use LangGraph interrupts for sensitive operations
6. **Monitor & trace:** Use LangSmith to debug agent behavior

---

## 📄 License

MIT

---

## References

- [LangChain Docs](https://python.langchain.com)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph)
- [OpenAI API](https://platform.openai.com/docs)
- [ReAct Paper](https://arxiv.org/abs/2210.03629) (Reasoning + Acting)
- [RAG Survey](https://arxiv.org/abs/2312.10997)
