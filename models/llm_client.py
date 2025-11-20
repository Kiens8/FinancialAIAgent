import requests

class LLMClient:
    def __init__(self, model="deepseek-r1:7b"):
        self.model = model

    def ask(self, prompt):
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt}
        )
        try:
            data = res.json()
            if 'response' in data:
                return data['response']
            if 'result' in data:
                return data['result']
            return data.get('choices', [{}])[0].get('message', {}).get('content', '')
        except Exception:
            return res.text
