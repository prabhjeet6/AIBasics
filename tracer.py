
from langfuse.langchain import CallbackHandler
from langfuse import get_client



def get_tracer()->CallbackHandler:
 return CallbackHandler()

def flush_tracer():
 get_client().flush()
    