from langchain_core.prompts import PromptTemplate
from llm import get_llm,get_ollama_llm
from langfuse.langchain import CallbackHandler
from langfuse import get_client
import os
from config import settings

def main():

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_base_url)

    langfuse_handler = CallbackHandler()

    #llm = get_llm()
    
    llm =get_ollama_llm()

    prompt_message="Tell me a startup idea that can help me half a million dollars quickly and tell me which llm(exact name) am i talking to, and is it free"
    
    input_prompt_template=PromptTemplate(
        template=prompt_message,
        input_variables=[]
        )

    chain=input_prompt_template | llm # langchain expression language first part(promptTemplate to be fed to second part(llm))
    response = chain.invoke({}, config={"callbacks": [langfuse_handler]}  )

    print(response.content)

    langfuse = get_client()
    langfuse.flush()

main()