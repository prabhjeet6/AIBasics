from typing import Any,Dict
# TODO
from graph.state import GraphState
from langchain_pinecone import PineconeVectorStore

from ragutils import get_embeddings_model

from  ./../ingestion import retriever

def retrieve(state: GraphState)->Dict[str,Any]:
    print("Retrieving")
    question=state["question"]
    documents=retriever.invoke(question)
    return {"documents":documents,"question":question}
