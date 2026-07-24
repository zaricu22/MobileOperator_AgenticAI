"""
Mobile Operator support AGENT (v2) — TURN-BY-TURN version using a
checkpointer, matching how a real system actually gets called: ONE
app.invoke() PER USER MESSAGE, not all questions passed in a single call.

LangGraph concepts now implemented in this file (previously missing):
  - START constant       -> graph.add_edge(START, "answer"), replacing the
                            older set_entry_point() style
  - Conditional routing  -> add_conditional_edges("answer", route_after_answer, ...)
                            picks "clarify" or "end" based on state, not a fixed wire
  - Cycles/loops         -> graph.add_edge("clarify", "answer") points BACKWARD to
                            an already-visited node, bounded by retry_count
  - Checkpointing        -> MemorySaver + thread_id bridges state ACROSS separate
                            app.invoke() calls, so turn 2 can resolve "it" without
                            the caller re-supplying turn 1's answer

  - Graph state: the checkpointer reloads a turn's persisted state before the
    next turn's node runs, which is what lets "does it cover roaming?" resolve
    "it" WITHOUT the caller re-supplying the previous answer
  - LLM: the same node calls the model whether it's the first turn or a
    follow-up — the prompt just changes shape depending on what's in state
  - Prompt Engineering: two prompt variants inside one node, chosen based on
    whether a previous answer exists in state

Core LangGraph concept this whole file is built on:
  In LangGraph, nodes = actions (functions that do something) — verbs, not
  nouns. Edges = what runs next. The actual "content" (the question, the
  account info, the answer) isn't the node itself — it's the state, a data
  object that gets handed from node to node, and each node's job is to read
  some of that state and write more of it.

pip install langchain langchain-openai langgraph
export OPENAI_API_KEY=...
"""

from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MAX_RETRIES = 2
UNCERTAIN_PHRASES = ("i don't know", "not sure", "unclear", "cannot determine")

# //====================  FAKE DATA  ======
# ---- 1. Account data answers will be grounded in ----
FAKE_ACCOUNTS = {
    "0641234567": {"plan": "Plan M", "data_left_gb": 3.2, "roaming_included": "EU only, 10GB cap"},
}
# //=======================================

# //====================  STATE OBJECT  ===
# ---- 2. Graph state — carries the conversation + working memory between nodes ----
class AgentState(TypedDict):
    phone_number: str
    question: str
    last_answer: Optional[str]
    answer: Optional[str]
    retry_count: int
# //=======================================

# //====================  PROMPTS  ========
# ---- 3. Prompts — one for the first question in a conversation, one for a follow-up ----
plan_prompt = ChatPromptTemplate.from_template(
    """You are a mobile operator support agent. Be concise.

Account info: {account}

Question: {question}
Answer:"""
)

followup_prompt = ChatPromptTemplate.from_template(
    """You are continuing the same support conversation. Here is what you
already told the customer:

"{previous_answer}"

Now answer their follow-up question, using that previous answer as context
to resolve anything ambiguous (like "it" or "that"). If you genuinely cannot
answer from the info given, say "I'm not sure" rather than guessing:

Follow-up question: {question}
Answer:"""
)
# //=======================================

# //====================  NODES  ==========
# ---- 4. Node: answer THIS TURN's question — picks a prompt based on what's in state ----
# Purpose: whether this is turn 1 or turn 5 of a conversation, the SAME node runs.
# What changes is only which prompt gets used:
#   - if state.get("last_answer") is None (previous answer doesn't exist), that's
#     the very first turn (nothing to reload yet) — we use plan_prompt
#   - on every next turn we use followup_prompt, with last_answer reloaded
#     automatically by the checkpointer, not passed in by the caller
def answer_node(state: AgentState) -> AgentState:
    account = FAKE_ACCOUNTS.get(state["phone_number"], {})  # <- Context-Augmented Generation
    last_answer = state.get("last_answer")

    if last_answer:
        chain = followup_prompt | llm | StrOutputParser()
        answer = chain.invoke({"previous_answer": last_answer, "question": state["question"]})  # <- Context Engineering
    else:
        chain = plan_prompt | llm | StrOutputParser()
        answer = chain.invoke({"account": account, "question": state["question"]})

    print(f"answer_node -> answer: {answer!r}")
    # both fields updated together: "answer" is this turn's result, and promoting it
    # into "last_answer" NOW is what makes it available as "last_answer" on the NEXT
    # separate invoke() call, once the checkpointer persists this state
    return {**state, "answer": answer, "last_answer": answer}


# ---- 5. Node: CLARIFY — the node a CYCLE loops back through, within ONE turn ----
# Purpose: just change state before control goes back to answer_node —
# bounded by retry_count, this loop happens entirely inside a single
# invoke() call — it's unrelated to the cross-turn memory the checkpointer provides (between invoke calls).
def clarify_node(state: AgentState) -> AgentState:
    account = FAKE_ACCOUNTS.get(state["phone_number"], {})  # <- Context-Augmented Generation
    enriched_question = (
        f"{state['question']} (For context, my plan is {account.get('plan', 'unknown')}.)"  # <- Context Engineering
    )
    print(f"clarify_node -> retry {state['retry_count'] + 1}/{MAX_RETRIES}, enriching question")
    return {**state, "question": enriched_question, "retry_count": state["retry_count"] + 1}
# //=======================================

# //====================  ROUTER  =========
# ---- 6. Router — CONDITIONAL ROUTING: loop back for clarification, or end this turn ----
def route_after_answer(state: AgentState) -> str:  # <- LRM (reasoning)
    answer_lower = (state.get("answer") or "").lower()
    is_uncertain = any(phrase in answer_lower for phrase in UNCERTAIN_PHRASES)
    if is_uncertain and state["retry_count"] < MAX_RETRIES:
        return "clarify"  # route:"clarify" — not the "clarify" NODE name, just this function's return value
    return "end"  # route:"end"
# //=======================================

# //====================  GRAPH  ==========
# ---- 7. Build the graph ----
graph = StateGraph(AgentState)
graph.add_node("answer", answer_node)
graph.add_node("clarify", clarify_node)

# Runtime order:
#   1. START -> "answer" — the graph begins, answer_node runs
#   2. After answer_node finishes, route_after_answer(state) is called (this is
#      what add_conditional_edges wires up) — it inspects state["answer"] and
#      returns either route:"clarify" or route:"end"
#   3. If it returned route:"clarify" -> clarify_node runs next
#   4. route:"clarify" -> route:"answer" — after clarify_node finishes, control goes back
#      to answer_node with updated state["question"] (this is the actual loop)
#   5. answer_node runs again, then step 2 repeats — route_after_answer is
#      checked again
#   6. This keeps cycling until route_after_answer returns route:"end" (or
#      retry_count hits MAX_RETRIES, which forces route:"end" regardless of how
#      uncertain the answer still sounds)
graph.add_edge(START, "answer")
graph.add_conditional_edges(
    "answer",
    route_after_answer,
    {"clarify": "clarify", "end": END},
)
graph.add_edge("clarify", "answer")  # CYCLE: loops back within this turn

# ---- 8. Checkpointer — bridges state ACROSS separate invoke() calls ----
# Purpose: without this, every app.invoke() would start from a blank AgentState,
# and "does it cover roaming?" would have no "last_answer" to resolve "it" against.
# MemorySaver keeps each thread_id's most recent state in memory between calls.
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
# //=======================================

# //====================  MAIN  ===========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "0641234567"}}

    # --- Turn 1: a brand-new conversation for this thread_id ---
    # retry_count=0 is passed explicitly because it's TURN-LOCAL — if we didn't
    # reset it, a later turn would inherit whatever retry_count the previous
    # turn ended on. last_answer is NOT passed — there isn't one yet.
    turn1 = app.invoke(
        {"phone_number": "0641234567", "question": "What plan am I on?", "retry_count": 0},
        config=config,
    )
    print("Turn 1:", turn1["answer"])

    # --- Turn 2: a SEPARATE invoke() call, same thread_id ---
    # Notice: we do NOT pass last_answer here. The checkpointer reloads turn 1's
    # persisted state (including last_answer) BEFORE this input is merged in, so
    # answer_node sees it automatically. retry_count=0 is passed again because
    # it's turn-local and must not carry over from turn 1.
    turn2 = app.invoke(
        {"phone_number": "0641234567", "question": "Does it cover roaming?", "retry_count": 0},
        config=config,
    )
    print("Turn 2:", turn2["answer"])  # resolves "it" via last_answer from turn 1

    # --- Turn 3: deliberately unanswerable from the account data on hand ---
    # This is the CLARIFY / CONDITIONAL ROUTING use-case actually firing: nothing
    # in FAKE_ACCOUNTS says anything about 5G or Japan, so followup_prompt should
    # produce an uncertain answer -> route_after_answer returns "clarify" ->
    # clarify_node enriches the question and loops back to answer_node -> repeats
    # until either a confident answer comes back or MAX_RETRIES is hit
    # (inside route_after_answer).
    turn3 = app.invoke(
        {"phone_number": "0641234567", "question": "Does it support 5G roaming in Japan?", "retry_count": 0},
        config=config,
    )
    print("Turn 3:", turn3["answer"])
    print("Turn 3 retries used:", turn3["retry_count"])  # > 0 if clarify actually looped
# //=======================================

# Full concept list mapped to this file:

# - LLM                 -> ChatOpenAI does the generation inside answer_node, every turn
# - Prompt Engineering  -> plan_prompt grounds the first turn in account data;
#                          followup_prompt resolves "it"/"that" using last_answer, and
#                          is told to say "I'm not sure" rather than guess (so
#                          route_after_answer has something to detect)
# - Vector Database     -> not demonstrated — no embeddings or similarity search here
# - RAG                 -> not demonstrated — same reason, no retrieval step at all
# - LangChain           -> prompt | llm | StrOutputParser() pipe syntax inside answer_node
# - LangGraph           -> START entry point, add_conditional_edges for routing, a
#                          genuine CYCLE (clarify -> answer back-edge, bounded by
#                          retry_count), and a MemorySaver checkpointer — this last
#                          part is the main addition: state now survives ACROSS
#                          separate invoke() calls, not just between nodes in one call.
#                          Still NOT used: ToolNode, streaming (.stream/.astream),
#                          async (.ainvoke), human-in-the-loop interrupts, parallel
#                          fan-out/Send, subgraphs, state history/time travel, the
#                          long-term Store, per-node RetryPolicy, recursion_limit config
# - MCP                 -> not demonstrated — no tool calls, every node is pure LLM generation
# - LRM (reasoning)     -> route_after_answer + the clarify loop is a (small) form of
#                          the model's own answer driving another attempt, though the
#                          retry DECISION itself is still plain Python, not the LLM
# - Context Engineering -> answer_node's prompt is built around exactly ONE piece of
#                          cross-turn context (last_answer), not the full conversation
#                          history; clarify_node adds exactly one more fact (plan name)
# - Context-Augmented   -> the account lookup (FAKE_ACCOUNTS) is curated, structured
#   Generation             context fed into the prompt, not retrieved by search
# - Memory              -> THIS is the concept this version demonstrates most directly:
#                          MemorySaver + thread_id is what lets turn 2 resolve "it"
#                          without turn 1's answer being re-passed by the caller
