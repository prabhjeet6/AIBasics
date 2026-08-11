from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

from config import settings


if __name__ == "__main__":
    print("Processing Ingestion Pipeline:")
    document_loader=UnstructuredLoader ("./document.txt", chunking_strategy="basic", max_characters=1000000)
    document=document_loader.load()

    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts=text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=settings.index_name,
        pinecone_api_key=settings.pinecone_api_key
    )
    print("Ingestion Pipeline processing complete.")