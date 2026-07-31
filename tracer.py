import os   

from config import settings
from langfuse.langchain import CallbackHandler
from langfuse import get_client

os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_base_url)

def get_tracer()->CallbackHandler:
 return CallbackHandler()

def flush_tracer():
 get_client().flush()
    