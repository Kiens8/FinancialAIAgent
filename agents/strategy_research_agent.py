from rag.rag_engine import RAGEngine
from models.llm_client import LLMClient

class StrategyResearchAgent:
    def __init__(self):
        self.rag = RAGEngine()
        self.llm = LLMClient()

    def research(self, question):
        context = "\n".join(self.rag.query(question))
        prompt = f"""
You are a financial strategy research assistant.
Context:
{context}

Task: Generate improved or new trading rules.
Output format: JSON rules array.
"""
        return self.llm.ask(prompt)
