from langchain_pinecone import PineconeVectorStore

from config import settings
from ragutils import get_embeddings_model,get_text_splitter,get_loader

if __name__ == "__main__":
    print("Processing Ingestion Pipeline:")

    document_loader=get_loader()
    document=document_loader.load()

    text_splitter=get_text_splitter()
    texts=text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks")

    embeddings = get_embeddings_model()

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=settings.index_name,
        pinecone_api_key=settings.pinecone_api_key
    )
    print("Ingestion Pipeline processing complete.")