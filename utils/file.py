from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def load_text(path):
    p = BASE / path
    if not p.exists():
        return ''
    return p.read_text(encoding='utf-8')
