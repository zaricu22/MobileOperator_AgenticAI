"""
Mobile Operator support example — LangChain ONLY (no LangGraph), covering:

  Basic (same as earlier examples):
    - LLM, Prompt Engineering, Vector Database, RAG, LangChain (LCEL pipes)

  Advanced (new in this file):
    - JsonOutputParser        -> force the LLM to return validated structured JSON
    - @tool + llm.bind_tools  -> let the model choose a tool, WITHOUT a full agent loop
    - create_react_agent /
      AgentExecutor           -> full agent loop: LLM picks a tool, sees the result,
                                  decides whether to call another tool or answer
    - ConversationBufferMemory-> keeps chat history across multiple agent calls
    - VectorStoreRetriever /
      PyPDFLoader             -> real-world document ingestion path for the KB (a PDF
                                  handbook, not hardcoded strings), plus retriever
                                  variants beyond plain similarity_search: MMR
                                  (diversity), score threshold (drop low-confidence
                                  matches), and metadata filtering (scope to a category)

pip install langchain langchain-groq langchain-huggingface langchain-community faiss-cpu pypdf pydantic
export GROQ_API_KEY=...
python -m llm.langchain_agent
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.tools import tool
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import create_react_agent, AgentExecutor

# Groq instead of ChatOpenAI: same LangChain chat-model interface, but Groq has
# a free tier (no billing setup needed) and serves this open-weight model fast.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# //============  SECTION: KB_RETRIEVER ===================
# ---- 1. Knowledge base ingestion — VectorStoreRetriever ----
# Purpose: hand-written kb_chunks strings, embedded and indexed, with retriever
# variants beyond plain similarity_search demonstrated on top of them.
# kb_metadata: each chunk tagged with a category, so retrieval can later be scoped
# to one section of the handbook instead of searching everything.
kb_chunks = [
    "Prepaid Plan S: 5GB data, 100 min, 100 SMS, 990 RSD/month.",
    "Prepaid Plan M: 15GB data, unlimited calls/SMS, 1490 RSD/month.",
    "Prepaid Plan M is our most popular plan, great value for heavy users.",  # near-duplicate of the line above, on purpose
    "Roaming in EU is included in Plan M, capped at 10GB.",
    "Roaming outside the EU is billed per MB, see the international rate card.",
]
kb_metadata = [
    {"category": "plans"},
    {"category": "plans"},
    {"category": "plans"},
    {"category": "roaming"},
    {"category": "roaming"},
]
# Local embedding model instead of OpenAIEmbeddings: Groq (this file's LLM
# provider) has no embeddings endpoint, and this keeps the whole script free/
# API-key-free for the vector store too. Runs on-device via sentence-transformers
# (no network call per embed) - weights are pulled from the HF Hub once on first
# run and cached under ~/.cache/huggingface, so every run after that is offline.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(kb_chunks, embeddings, metadatas=kb_metadata)

# VectorStoreRetriever: the standard LangChain wrapper around a vector store's
# search, so it can be used anywhere a "Runnable" retriever is expected — but it's
# NOT limited to plain similarity_search. Three different configurations below,
# each solving a different problem plain top-k similarity has:

# (a) plain similarity — what every earlier example used. Always returns exactly
#     k results, even if some are near-duplicates or barely relevant.
retriever_plain = vector_db.as_retriever(search_kwargs={"k": 2})

# (b) MMR (Maximal Marginal Relevance) — returns DIVERSE results instead of just the
#     top-k closest. Without this, both "Plan M" chunks above (near-duplicates) could
#     fill the k=2 slots, crowding out the roaming chunk that's actually relevant to
#     a different part of the question. `fetch_k` widens the candidate pool MMR
#     picks the diverse subset from; `lambda_mult` trades relevance vs diversity
#     (1.0 = pure relevance, 0.0 = pure diversity).
retriever_diverse = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.5},
)

# (c) score threshold — DROPS low-confidence matches instead of always returning k
#     results. Plain similarity_search(k=2) always hands back 2 chunks even if
#     neither one is actually relevant to the question; this only returns chunks
#     above the given similarity score, so an off-topic question can legitimately
#     return zero results instead of forcing a bad match.
retriever_confident = vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.8},
)

# (d) metadata filtering — scopes the search to one category instead of the whole
#     KB, so a roaming question can't accidentally match on a plans chunk just
#     because the wording happens to be similar.
retriever_roaming_only = vector_db.as_retriever(
    search_kwargs={"k": 2, "filter": {"category": "roaming"}}
)

# search_plans_kb (the tool below) uses the diverse retriever by default, since an
# agent-facing tool benefits most from not getting crowded out by duplicate chunks.
kb_retriever = retriever_diverse
# //=======================================================

# //============  SECTION: PDF_RETRIEVER ==================
PDF_PATH = "plans_handbook.pdf"

# ---- 1. PyPDFLoader — reads the PDF page by page ----
# Purpose: each page becomes its own Document object with metadata already
# attached (source file, page number) — this is what hardcoded strings can't
# give you for free.
loader = PyPDFLoader(PDF_PATH)
raw_docs = loader.load()

print(f"Loaded {len(raw_docs)} pages")
print("Page 1 metadata:", raw_docs[0].metadata)
print("Page 1 preview:", raw_docs[0].page_content[:80].replace("\n", " "), "...")

# ---- 2. RecursiveCharacterTextSplitter — breaks each page into smaller chunks ----
# Purpose: a whole page can be too large/unfocused to embed as one vector. The
# splitter cuts each page into ~300-character pieces, but split_documents()
# CARRIES OVER each parent page's metadata onto every chunk it produces — so a
# chunk from the middle of page 2 still knows it came from page 2.
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(raw_docs)

print(f"\nSplit into {len(chunks)} chunks")
for c in chunks:
    print(f"  page={c.metadata['page']}  text={c.page_content[:60]!r}...")

# ---- 3. Embed + store — same FAISS step as every other example ----
# Purpose: nothing changes here vs the string-based examples; from_documents()
# just uses each chunk's .page_content for the embedding and keeps .metadata
# attached to the stored vector.
pdf_vector_db = FAISS.from_documents(chunks, embeddings)
pdf_retriever = pdf_vector_db.as_retriever(search_kwargs={"k": 2})
# //=======================================================

# //============  SECTION: TOOLS ==========================
# ---- 2. Tools — @tool turns a plain function into something an LLM/agent can call ----
# Purpose: each tool is a bounded capability (MCP-style) the model can invoke by name
# and structured arguments, instead of us hardcoding which function runs when.
FAKE_ACCOUNTS = {
    "0641234567": {"plan": "Plan M", "data_left_gb": 3.2, "minutes_left": "unlimited"},
}
FAKE_OUTAGES = {
    "novi sad": "No known outages.",
    "belgrade": "Partial 4G outage in New Belgrade, ETA fix: 3 hours.",
}


@tool
def get_account_usage(phone_number: str) -> str:
    """Look up remaining data/minutes and current plan for a phone number."""
    acc = FAKE_ACCOUNTS.get(phone_number)
    return str(acc) if acc else "Account not found."


@tool
def check_network_outage(city: str) -> str:
    """Check current network outage status for a city."""
    return FAKE_OUTAGES.get(city.lower(), "No data for that location.")


@tool
def search_plans_kb(query: str) -> str:
    """Search the plans/roaming knowledge base for relevant policy info."""
    docs = kb_retriever.invoke(query)  # <- RAG
    return "\n".join(d.page_content for d in docs) or "No matching info found."


tools = [get_account_usage, check_network_outage, search_plans_kb]
# //=======================================================

# //============  SECTION: JSON_OUTPUT_PARSER =============
# ---- 3. JsonOutputParser — force structured output for a classification step ----
# Purpose: instead of parsing free text ("this looks like a billing question..."),
# make the LLM return a JSON object matching a schema we control, so downstream
# code can use it like normal data (state["intent"], state["urgency"]) with no
# string-parsing guesswork.
class QueryInfo(BaseModel):
    intent: str = Field(description="one of: plans, roaming, usage, outage")
    urgency: str = Field(description="one of: low, medium, high")


json_parser = JsonOutputParser(pydantic_object=QueryInfo)

extract_prompt = PromptTemplate(
    template=(
        "Extract structured info from the customer's question.\n"
        "{format_instructions}\n"  # <- Context-Augmented Generation
        "Question: {question}\n"
    ),
    input_variables=["question"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},  # <- Context-Augmented Generation
)

# LCEL chain: prompt -> LLM -> parser, same pipe pattern as earlier examples,
# just with JsonOutputParser instead of StrOutputParser at the end.
# syntax: PromptTemplate | ChatGroq | args
extract_chain = extract_prompt | llm | json_parser
# //=======================================================

# //============  SECTION: MEMORY =========================
# ---- 5. ConversationBufferMemory — remembers chat history across separate calls ----
# Purpose: without this, every agent_executor.invoke() call below would start from
# zero context. The memory object accumulates the conversation so a follow-up
# question ("does it cover roaming?") can be understood in light of earlier turns.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# //=======================================================

# //============  SECTION: BIND_TOOLS =====================
# ---- 4. @tool + llm.bind_tools — manual, single-step tool calling (no agent loop) ----
# Purpose: the simplest way to let an LLM "choose" a tool. bind_tools tells the model
# what tools exist and their argument schemas; the model's response contains
# `tool_calls` describing what IT wants to call. WE still have to execute the tool
# and feed the result back — there's no automatic loop like AgentExecutor has below.
llm_with_tools = llm.bind_tools(tools)


def manual_tool_call_demo(question: str) -> str:
    response = llm_with_tools.invoke(question)
    if not response.tool_calls:
        return response.content  # model answered directly, no tool needed

    call = response.tool_calls[0]
    tool_by_name = {t.name: t for t in tools}
    result = tool_by_name[call["name"]].invoke(call["args"])
    print(f"manual_tool_call_demo -> model chose tool '{call['name']}' with args {call['args']}")
    return result
# //=======================================================

# //============  SECTION: REACT_AGENT ====================
# ---- 6. create_react_agent + AgentExecutor — the full autonomous agent loop ----
# Purpose: unlike step 4's ONE tool call, a ReAct agent can call a tool, look at the
# result, decide to call ANOTHER tool, and keep going until it decides it has enough
# to answer — the "Thought -> Action -> Observation" loop repeats as many times as
# the model decides it needs to.
#
# Where each placeholder below gets filled in from:
#   {tools}, {tool_names}  -> from the `tools` list passed to create_react_agent()
#                             below, at agent-creation time  # <- Context Engineering
#   {input}                -> from the dict passed to agent_executor.invoke()
#                             (e.g. invoke({"input": "..."})) - the only one
#                             YOU provide directly, per call  # <- Context Engineering
#   {chat_history}          -> pulled from `memory` automatically by AgentExecutor,
#                             because memory_key="chat_history" matches this name
#                             # <- Context Engineering
#   {agent_scratchpad}      -> generated internally by AgentExecutor's loop: it
#                             formats the Thought/Action/Observation steps taken
#                             SO FAR in this invoke() call and re-injects them here
#                             before each next call to the LLM (empty on the first
#                             pass, grows with each tool call after that)
#                             # <- Context Engineering (both lines above)
react_prompt = PromptTemplate.from_template(
    """Answer the customer's question as best you can. You have access to these tools:

{tools}

Use this format:

Question: the input question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the customer

Previous conversation:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, react_prompt)  # <- LRM (reasoning): plans, acts, observes, re-plans
agent_executor = AgentExecutor(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)


def build_agent_executor() -> AgentExecutor:
    """Fresh AgentExecutor with its OWN ConversationBufferMemory, instead of the
    shared `memory` global above. Callers that serve multiple customers at once
    (e.g. web/app.py, keying one of these per phone number) need isolated chat
    histories - reusing `agent_executor` would leak one customer's conversation
    into another's `chat_history`."""
    session_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return AgentExecutor(
        agent=agent, tools=tools, memory=session_memory, verbose=True, handle_parsing_errors=True
    )
# //=======================================================

# //============  SECTION: MAIN ===========================
if __name__ == "__main__":
    # //--------------------------------------------------------
    # --- JsonOutputParser in isolation ---
    info = extract_chain.invoke({"question": "URGENT: is there an outage in Belgrade?"})
    print("Structured extraction:", info)

    # //--------------------------------------------------------
    # --- bind_tools in isolation (one tool call, no loop) ---
    tool_result = manual_tool_call_demo("How much data do I have left? My number is 0641234567.")
    print(tool_result)
    # feed the tool's raw result back through JsonOutputParser to force it into
    # the QueryInfo schema, since the tool's return value is plain text, not JSON
    print(extract_chain.invoke({"question": tool_result}))

    # //--------------------------------------------------------
    # --- full agent loop, with memory carried across both calls ---
    r1 = agent_executor.invoke({"input": "What's in Plan M?"})
    print(r1["output"])
    # feed the agent's free-text final answer back through JsonOutputParser to
    # force it into the QueryInfo schema, since the ReAct loop itself only ever
    # produces plain text, not JSON
    print(extract_chain.invoke({"question": r1["output"]}))

    r2 = agent_executor.invoke({"input": "Does it cover roaming in the EU?"})
    print(r2["output"])  # resolves "it" using chat_history from memory

    # //--------------------------------------------------------
    # ----  PDF-Retrieval in isolation - query and show WHERE answer came from ----
    question = "Does my plan cover roaming in the EU?"
    results = pdf_retriever.invoke(question)  # <- RAG
    # Purpose: this is the payoff of keeping page metadata — the answer can now cite
    # a page number, not just return anonymous text like the hardcoded-string
    # version had to.
    print(f"\nQuestion: {question}")
    for r in results:
        print(f"  [page {r.metadata['page'] + 1}] {r.page_content[:100]!r}...")
# //=======================================================

# Full concept list mapped to this file:

# - LLM                  -> ChatGroq powers every step: classification, tool
#                           selection, and the agent's reasoning/answers
# - Prompt Engineering    -> extract_prompt forces a JSON schema; react_prompt forces
#                           the strict Thought/Action/Observation format
# - Vector Database       -> two separate FAISS indexes: vector_db from kb_chunks
#                           (KB_RETRIEVER section) and pdf_vector_db from the real
#                           ingested PDF (PDF_RETRIEVER section)
# - RAG                   -> search_plans_kb wraps kb_retriever (MMR) as a tool the
#                           AGENT decides to call; retriever_confident and
#                           retriever_roaming_only show two more retrieval strategies
#                           available for the same underlying vector_db; pdf_retriever
#                           shows the same mechanism over a real ingested PDF
# - LangChain             -> LCEL pipes (prompt | llm | parser) used throughout
# - LangGraph             -> NOT used in this file — AgentExecutor's built-in loop
#                           replaces the graph/routing role LangGraph played earlier
# - MCP                   -> the 3 @tool functions stand in for real backend/MCP calls
# - LRM (reasoning)       -> the ReAct loop (steps 6) IS multi-step reasoning: the
#                           model plans, acts, observes, and re-plans across turns
# - Context Engineering   -> react_prompt only exposes chat_history + tool
#                           descriptions, not the whole KB — the agent pulls in more
#                           context only by calling search_plans_kb itself
# - Context-Augmented     -> extract_chain's format_instructions are injected,
#   Generation               structured context telling the model exactly how to
#                           shape its JSON output
# - Memory                -> ConversationBufferMemory persists chat_history across
#                           both agent_executor.invoke() calls in __main__
