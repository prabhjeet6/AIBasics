from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_unstructured import UnstructuredLoader

from config import settings
from ragutils import get_embeddings_model

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs=[
    UnstructuredLoader(web_url=url,chunking_strategy="basic",max_characters=1000000)
    .load()
    for url in urls
]

# it is a list comprehension that flattens a nested list.
# List flattening is needed because each call to UnstructuredLoader.load() returns a list of documents, so your list
# comprehension creates a nested list of documents. But split_documents() expects a flat list of Document
# objects.

docs_list=[
    item for sublist in docs for item in sublist
]

text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,chunk_overlap=0
)

doc_splits=text_splitter.split_documents(docs_list)

embeddings = get_embeddings_model()

# ingest
PineconeVectorStore.from_documents(
        doc_splits,
        embeddings,
        index_name=settings.index_name,
        pinecone_api_key=settings.pinecone_api_key
)

# retrieve
retriever = PineconeVectorStore(
    index_name=settings.agentic_rag_index_name,
    embedding=get_embeddings_model(),
).as_retriever(
    search_kwargs={"k": 4}
)

