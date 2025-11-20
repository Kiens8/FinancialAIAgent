from rag.rag_engine import RAGEngine
from models.llm_client import LLMClient
import json

class DecisionAgent:
    def __init__(self):
        self.rag = RAGEngine()
        self.llm = LLMClient()

    def decide(self, market_state):
        context = "\n".join(self.rag.query("trading rules"))
        prompt = f"""
You are a trading decision agent.
Context:
{context}

Market State:
{json.dumps(market_state, indent=2)}

Decide:
- Enter / Exit / Avoid
- Confidence level
- % capital to allocate
Output JSON only.
"""
        return self.llm.ask(prompt)
