import os
import sys
import re
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

# Make the project root importable (so `rag` package resolves correctly)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.agent import build_agent

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AutoNet-GNN  ·  Agentic RAG Interface            ║
║     Ask anything — the agent decides what to do         ║
╚══════════════════════════════════════════════════════════╝
"""

# Quick-action shortcuts → expanded natural-language prompts fed to the LLM
SHORTCUTS = {
    "scan":        "Run a fresh GNN scan on the network and report all anomalies found.",
    "anomalies":   "Query all anomaly detection results and list every flagged node with its score.",
    "root cause":  "Find the root cause of the most critical anomalies: query the anomaly scores, "
                   "explain the top flagged node, search the knowledge base for matching patterns, "
                   "and summarise what is most likely causing the issue.",
    "status":      "Give me an overall network health status: query all anomaly scores and provide "
                   "a concise summary of network condition.",
    "exit":        None,
    "quit":        None,
    "q":           None,
}

# Patterns for commands with a node argument: "explain 5", "fix 3", "neighbors 7"
NODE_PATTERNS = [
    (r"^explain\s+(\d+)$",   "Explain in detail why node {n} was flagged as anomalous, "
                              "including feature analysis and security assessment."),
    (r"^fix\s+(\d+)$",       "Generate a full remediation action plan for node {n}."),
    (r"^neighbors\s+(\d+)$", "Show the 1-hop neighbor context of node {n} and their anomaly status."),
]


def print_help():
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Shortcut", style="cyan")
    table.add_column("What the agent does")
    table.add_row("scan",            "Trigger a fresh GNN scan")
    table.add_row("anomalies",       "List all flagged nodes")
    table.add_row("root cause",      "Find & explain the most likely root cause")
    table.add_row("status",          "Overall network health summary")
    table.add_row("explain <id>",    "Deep-dive on a specific node (e.g. explain 5)")
    table.add_row("fix <id>",        "Remediation plan for a node (e.g. fix 3)")
    table.add_row("neighbors <id>",  "Show 1-hop neighbors (e.g. neighbors 7)")
    table.add_row("exit / q",        "Quit")
    table.add_row("[italic]anything else[/]", "[italic]Free-form question — agent decides the tools[/]")
    console.print(table)
    console.print("[dim]Tip: you can also type naturally, e.g. 'why is node 12 behaving oddly?'[/]\n")


def resolve_prompt(user_input: str) -> str | None:
    """
    Map shortcut commands or patterns to expanded prompts.
    Returns None for exit commands, or the (possibly expanded) prompt string.
    """
    lower = user_input.strip().lower()

    # Check plain shortcuts first
    if lower in SHORTCUTS:
        return SHORTCUTS[lower]  # None for exit commands

    # Check node-argument patterns
    for pattern, template in NODE_PATTERNS:
        m = re.match(pattern, lower)
        if m:
            return template.format(n=m.group(1))

    # Free-form — pass straight through
    return user_input.strip()


def main():
    console.print(BANNER, style="bold cyan")

    # Choose backend — just press Enter for ollama (default)
    backend_raw = Prompt.ask(
        "LLM Backend [bold](ollama / openai)[/] — press Enter for ollama",
        default="ollama"
    ).strip().lower()
    backend = "openai" if backend_raw == "openai" else "ollama"

    if backend == "ollama":
        model = Prompt.ask("Ollama model", default="llama3.2")
    else:
        model = Prompt.ask("OpenAI model", default="gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = Prompt.ask("OpenAI API Key", password=True)

    rebuild_raw = Prompt.ask(
        "Rebuild knowledge base? [bold](y / n)[/] — press Enter for no",
        default="n"
    ).strip().lower()
    rebuild = rebuild_raw == "y"

    console.print("\n[bold green]Building agent...[/]")
    agent = build_agent(llm_backend=backend, model=model, rebuild_kb=rebuild)
    console.print("[bold green]✔ Agent ready![/]\n")

    print_help()

    # All turns share a thread_id so memory is maintained across the session
    config = {"configurable": {"thread_id": "autonet-session"}}

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/]")

            if not user_input.strip():
                continue

            if user_input.strip().lower() in ("help", "?", "h"):
                print_help()
                continue

            resolved = resolve_prompt(user_input)

            # Exit commands
            if resolved is None:
                console.print("[yellow]Goodbye.[/]")
                break

            console.print("\n[bold yellow]Agent thinking...[/]")
            result = agent.invoke(
                {"messages": [("human", resolved)]},
                config=config
            )
            output = result["messages"][-1].content
            console.print(Panel(output, border_style="green", title="Agent", title_align="left"))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/]")


if __name__ == "__main__":
    main()
