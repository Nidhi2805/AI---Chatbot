"""Live terminal demo. Run: python interactive_demo.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import AIFirewallPipeline
from src.display import show_query_result

pipe = AIFirewallPipeline(user="demo")
print("Type a question (or 'quit'):")
while True:
    try:
        q = input("> ").strip()
    except EOFError:
        break
    if q.lower() in {"quit", "exit"}:
        break
    if q:
        show_query_result(pipe, q)
