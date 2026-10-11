from typing import Any,Dict
# TODO
from graph.state import GraphState
from langchain_pinecone import PineconeVectorStore

from ragutils import get_embeddings_model

retriever = PineconeVectorStore(
    index_name="rag-pinecone", # TODO correct/new index
    embedding=get_embeddings_model(),
).as_retriever(
    search_kwargs={"k": 4}
)

def retrieve(state: GraphState)->Dict[str,Any]:
    print("Retrieving")
    question=state["question"]
    documents=retriever.invoke(question)
    return {"documents":documents,"question":question}
