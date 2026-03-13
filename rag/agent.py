import os
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
from rag.knowledge_base import build_knowledge_base, load_knowledge_base
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

    # 2. Vector store + retriever tool
    if rebuild_kb or not os.path.exists("./chroma_db"):
        vectorstore = build_knowledge_base()
    else:
        vectorstore = load_knowledge_base()

    retriever      = vectorstore.as_retriever(search_kwargs={"k": 4})
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_network_knowledge",
        description=(
            "Search the AutoNet-GNN knowledge base for past anomaly reports, "
            "node history, security patterns, and remediation guidance. "
            "Use this to get context before explaining or remediating an anomaly."
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

    # 4. In-memory conversation history (per thread_id)
    memory = MemorySaver()

    # 5. System prompt
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    # 6. LangGraph react agent (replaces AgentExecutor)
    agent = create_react_agent(
        llm,
        tools,
        prompt=system_msg,
        checkpointer=memory,
    )

    return agent
