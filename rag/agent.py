import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from rag.tools import (
    query_anomalies,
    explain_node,
    get_neighbors,
    generate_remediation,
    run_gnn_scan,
)
from rag.knowledge_base import build_knowledge_base, load_knowledge_base, CHROMA_DIR
from rag.prompts import SYSTEM_PROMPT


def build_agent(llm_backend: str = "ollama", model: str = "llama3.2",
                rebuild_kb: bool = False):
    """
    Build the AutoNet-GNN Agentic RAG executor using LangGraph.

    Args:
        llm_backend : "ollama" (free/local) or "openai"
        model       : model name (e.g. "llama3.2", "gpt-4o-mini")
        rebuild_kb  : rebuild the knowledge base from scratch
    """
    # 1. LLM
    if llm_backend == "ollama":
        llm = ChatOllama(model=model, temperature=0.1)
    else:
        llm = ChatOpenAI(model=model, temperature=0.1,
                         api_key=os.environ.get("OPENAI_API_KEY"))

    # 2. Vector store (dual poisoning knowledge + GNN reports)
    if rebuild_kb or not os.path.exists(CHROMA_DIR):
        vectorstore = build_knowledge_base()
    else:
        vectorstore = load_knowledge_base()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_network_knowledge",
        description=(
            "Search the AutoNet-GNN dual poisoning defense knowledge base. "
            "Contains: node anomaly reports, feature/structure/dual poisoning "
            "detection strategies, defense mechanisms, remediation playbooks, "
            "and real-world attack scenarios. Use this for context before "
            "explaining or remediating anomalies."
        )
    )

    # 3. All tools
    tools = [
        query_anomalies,
        explain_node,
        get_neighbors,
        generate_remediation,
        run_gnn_scan,
        retriever_tool,
    ]

    # 4. Memory
    memory = MemorySaver()

    # 5. System prompt
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    # 6. LangGraph ReAct agent
    agent = create_react_agent(
        llm,
        tools,
        prompt=system_msg,
        checkpointer=memory,
    )

    return agent
