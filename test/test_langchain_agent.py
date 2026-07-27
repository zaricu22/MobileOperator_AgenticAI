"""
Direct tests against llm/langchain_agent.py's tools and retrievers - the
parts of this file that DON'T touch the LLM at all, so they're deterministic
and need no Groq call or fake model.

Not covered here (would need a real Groq call, or a refactor to inject a
fake LLM): extract_chain (JsonOutputParser step), manual_tool_call_demo /
llm_with_tools (bind_tools step), and agent_executor / the full ReAct loop.
agent_executor, extract_chain, and llm_with_tools all bind the real ChatGroq
at IMPORT TIME - unlike llm/langgraph_agent.py's nodes, which look up `llm`
fresh on every call, there's no module attribute here a test could
monkeypatch after the fact (see the CAUTION comment in test_api.py).

Also not covered: test_api.py's backend="langchain" tests already exercise
api/main.py's/web/app.py's session dispatch, but with a FakeExecutor
standing in for the whole agent - these tests are the other half, covering
the tools and retrievers those fakes never actually touch.

pip install pytest
pytest test/test_langchain_agent.py
"""
import llm.langchain_agent as m


def test_get_account_usage_known_number():
    result = m.get_account_usage.invoke({"phone_number": "0641234567"})
    assert result == "{'plan': 'Plan M', 'data_left_gb': 3.2, 'minutes_left': 'unlimited'}"


def test_get_account_usage_unknown_number():
    result = m.get_account_usage.invoke({"phone_number": "0000000000"})
    assert result == "Account not found."


def test_check_network_outage_known_city_is_case_insensitive():
    result = m.check_network_outage.invoke({"city": "BELGRADE"})
    assert result == "Partial 4G outage in New Belgrade, ETA fix: 3 hours."


def test_check_network_outage_unknown_city():
    result = m.check_network_outage.invoke({"city": "Nis"})
    assert result == "No data for that location."


def test_search_plans_kb_tool_wraps_kb_retriever():
    # search_plans_kb is what the ReAct agent actually calls - it should
    # return SOMETHING for an on-topic query, not the empty-result fallback.
    result = m.search_plans_kb.invoke({"query": "roaming"})
    assert result != "No matching info found."
    assert "Roaming" in result


def test_retriever_plain_always_returns_exactly_k_results():
    # Plain similarity_search always returns k results regardless of
    # relevance - even a nonsense query still fills both slots.
    docs = m.retriever_plain.invoke("asdkjaslkdjaslkdj")
    assert len(docs) == 2


def test_retriever_roaming_only_filters_by_metadata():
    # The filter is a HARD constraint, not a relevance suggestion - even a
    # query about something else entirely still only returns roaming chunks.
    docs = m.retriever_roaming_only.invoke("what plans do you offer")
    assert len(docs) == 2
    assert all(d.metadata["category"] == "roaming" for d in docs)


def test_retriever_diverse_avoids_near_duplicate_chunks():
    # kb_chunks has two near-duplicate "Plan M" chunks on purpose (see the
    # comment next to kb_chunks in langchain_agent.py) - MMR with k=2 should
    # not return both for a plans-focused query, unlike plain similarity
    # search, which easily would fill both slots with near-duplicates.
    docs = m.retriever_diverse.invoke("Tell me about Plan M")
    contents = {d.page_content for d in docs}
    both_plan_m_duplicates = {
        "Prepaid Plan M: 15GB data, unlimited calls/SMS, 1490 RSD/month.",
        "Prepaid Plan M is our most popular plan, great value for heavy users.",
    }
    assert not both_plan_m_duplicates.issubset(contents)


def test_retriever_confident_score_threshold_never_passes_with_faiss_l2_distance():
    # KNOWN QUIRK, not a design choice: FAISS's default distance metric is
    # L2, whose "relevance scores" aren't bounded to [0, 1] the way
    # similarity_search_with_relevance_scores() expects (LangChain warns
    # about this at call time: "Relevance scores must be between 0 and 1").
    # Because of that mismatch, score_threshold=0.8 filters out EVERY
    # result, even clearly on-topic ones - this retriever doesn't behave as
    # advertised in this file's comments without also configuring a
    # normalized distance_strategy on the FAISS index. Documented here so a
    # future fix to that doesn't silently break this test.
    on_topic = m.retriever_confident.invoke("roaming")
    off_topic = m.retriever_confident.invoke("what time does the pizza shop close")
    assert on_topic == []
    assert off_topic == []
