from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_unstructured import UnstructuredLoader

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
docs_list=[
    item for sublist in docs for item in sublist
]

text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,chunked_overlap=0
)

doc_splits=text_splitter.split_documents(docs_list)

