from langchain_core.document_loaders import BaseLoader
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter, TextSplitter
from langchain_unstructured import UnstructuredLoader

from config import settings

# Embedding: It is a technique useful in NLP to create a vector space such that distance b/w
# vectors in the space have a certain meaning.

# A vector is a sequence of numbers which can represent complex objects like words, images or
# audio files etc. in a continuous high dimensional space called embedding.

# Embedding Model: It can be thought of as a black box which receives objects which are represented
# by text, processes them and outputs array of numbers which represents those objects in vector
# space .

# Embedding Model finds vector for the query and compares that with the stored vectors based on
# the chosen metric

# vector db saves those vectors/embeddings

# connect to vector store providing embedding model according to the configuration provided in the
# index created in pinecone as inputs
def get_embeddings_model()->HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def get_vector_store()->VectorStore:
    embeddings=get_embeddings_model()
    return PineconeVectorStore(index_name=settings.index_name, embedding=embeddings)


# With a chunk overlap of zero, the context might be lost, langchain's
# RecursiveCharacterTextSplitter tries to split at ending of sentence or paragraph and keeps
# overlap of 10 to 20 percent

def get_text_splitter()->TextSplitter:
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

def get_loader()->BaseLoader:
    return UnstructuredLoader ("./document.txt", chunking_strategy="basic", max_characters=1000000)