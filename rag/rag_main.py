import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Make GNN importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GNN'))

from rag.agent import build_agent

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AutoNet-GNN  ·  Agentic RAG Interface            ║
║        Ask anything about your network topology         ║
╚══════════════════════════════════════════════════════════╝
Commands:
  • "scan"          → trigger fresh GNN scan
  • "anomalies"     → list all flagged nodes
  • "explain 0"     → explain why node 0 was flagged
  • "fix 0"         → get remediation plan for node 0
  • "neighbors 5"   → show node 5's connections
  • "exit"          → quit
"""

def main():
    console.print(BANNER, style="bold cyan")
    
    # Choose backend
    backend = Prompt.ask(
        "LLM Backend", choices=["ollama", "openai"], default="ollama"
    )
    
    if backend == "ollama":
        model = Prompt.ask("Ollama model", default="llama3.2")
    else:
        model = Prompt.ask("OpenAI model", default="gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = Prompt.ask("OpenAI API Key", password=True)
    
    rebuild = Prompt.ask("Rebuild knowledge base?", choices=["y", "n"], default="n") == "y"
    
    console.print("\n[bold green]Building agent...[/]")
    agent = build_agent(llm_backend=backend, model=model, rebuild_kb=rebuild)
    console.print("[bold green]✔ Agent ready. Start chatting!\n[/]")
    
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/]")
            
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye.[/]")
                break
            
            if not user_input.strip():
                continue
            
            console.print("\n[bold yellow]Agent:[/]")
            result = agent.invoke({"input": user_input})
            console.print(Panel(result["output"], border_style="green"))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/]")


if __name__ == "__main__":
    main()
