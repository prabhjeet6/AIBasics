from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import settings

# connect to vector store providing embedding model according to the configuration provided in the
# index created in pinecone as inputs
def get_embeddings_model()->HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def get_vector_store()->VectorStore:
    embeddings=get_embeddings_model()
    return PineconeVectorStore(index_name=settings.index_name, embedding=embeddings)