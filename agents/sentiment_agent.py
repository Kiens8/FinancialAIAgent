import json
from models.llm_client import LLMClient

class SentimentAgent:
    def __init__(self):
        self.llm = LLMClient()

    def analyze(self, comments):
        text = "\n".join(comments)
        prompt = f"Analyze sentiment (Positive/Negative/Mixed + summary):\n{text}"
        return self.llm.ask(prompt)
