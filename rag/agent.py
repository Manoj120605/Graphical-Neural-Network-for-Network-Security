import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools.retriever import create_retriever_tool

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
                rebuild_kb: bool = False) -> AgentExecutor:
    """
    Build the AutoNet-GNN Agentic RAG executor.
    
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
    
    # 4. Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    # 5. Memory (last 10 turns)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )
    
    # 6. Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )
