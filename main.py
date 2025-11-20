from agents.strategy_research_agent import StrategyResearchAgent
from agents.decision_agent import DecisionAgent
from utils.file import load_text

if __name__ == '__main__':
    research_agent = StrategyResearchAgent()
    decision_agent = DecisionAgent()

    # Load rules
    rules = load_text("config/strategy_rules.txt")
    if rules:
        research_agent.rag.add_document(rules)

    # Example query
    print("Generating strategy update...")
    updated = research_agent.research("How to improve momentum strategy?")
    print(updated)

    # Make decision
    decision = decision_agent.decide({"price": 120, "volume": 50000})
    print(decision)
