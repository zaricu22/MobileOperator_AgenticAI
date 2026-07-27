import os
import sys

# Repo root is where the llm/, api/, web/ packages live - make sure it's on
# sys.path so `import llm.langgraph_agent` / `from api.main import api`
# resolve from test/ regardless of how pytest is invoked (e.g. from a
# different cwd).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# llm/langgraph_agent.py instantiates ChatGroq at import time. The
# constructor doesn't make a network call, but it does expect *something* in
# GROQ_API_KEY — set a dummy value so tests never depend on a real key
# (the LLM itself is monkeypatched out in the tests).
os.environ.setdefault("GROQ_API_KEY", "gsk-test-not-a-real-key")
