import asyncio
import os
import ssl
from typing import Any,Dict,List

import certifi

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import (
    TavilyCrawl,
    TavilyExtract,
    TavilyMap,
    tavily_extract,
    tavily_crawl,
)
from openrouter import embeddings
from scipy._external.cobyqa import settings

from config import Settings
from ragutils import get_embeddings_model, get_vector_store
from retrieval import vectorstore

#from logger import (Colors,log_error,log_header,log_info_log_success,log_warning)

# Configure SSL context to use certifi certificates
ssl_context=ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

#TODO: pass correct index name
embeddings=get_embeddings_model()
vectorstore=get_vector_store(settings.index_name)

tavily_extract=TavilyExtract(max_depth=5,max_breadth=20,max_pages=1000)
tavily_map=TavilyMap()
tavily_crawl=TavilyCrawl()

