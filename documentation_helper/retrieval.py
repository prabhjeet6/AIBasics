from langchain.tools import tool
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from config import settings

from agentutils import get_llm
from ragutils import get_embeddings_model, get_vector_store


embeddings = get_embeddings_model()
vectorstore = get_vector_store(settings.langchain_documentation_index_name)
model = get_llm()


@tool
def retrieve_context(query: str):
    """
    Retrieve relevant documentation to help answer user queries about LangChain.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retrieved_docs = retriever.invoke(query)

    serialized = "\n\n".join(
        (
            f"Source:{doc.metadata.get('source', 'unknown')}\n\n"
            f"Context:{doc.page_content}"
        )
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_agent_for_retrieval(query: str) -> dict[str, list[Any] | Any] | None:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
          - answer: The generated answer
          - context: List of retrieved documents
    """

    system_prompt = (
        " You are a helpful AI assistant that answers questions about Langchain documentation. "
        " You have access to a tool that retrieves relevant documentation. "
        " Use the tool to find relevant information before answering questions. "
        " Always cite the sources you use in your answers "
        " If you can not find the answer in the retrieved documentation, say so. "
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})

    # response is a dict, response["messages"] gives value corresponding to it
    # messages is a list, response["messages"][-1] returns last element of the list
    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

        return {
            "answer": answer,
            "context": context_docs,
        }
    return None


if __name__ == "__main__":
    result = run_agent_for_retrieval(query="What are Deep agents?")
    print(result)
    print("Retrieval Ends")
