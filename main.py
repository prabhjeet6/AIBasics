from langchain_core.prompts import PromptTemplate
from llm import get_llm,get_ollama_llm
from tracer import get_tracer,flush_tracer

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

import os
from config import settings

os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)

def main():

    tracing_handler = get_tracer()

    llm = get_llm()
    
    #llm =get_ollama_llm() # ollama does not support Tools

    tools=[TavilySearch()]

    agent=create_agent(model=llm,tools=tools)

    #prompt_message="Tell me a startup idea that can help me half a million dollars quickly and tell me which llm(exact name) am i talking to, and is it free"
    #input_prompt_template=PromptTemplate(
    #    template=prompt_message,
    #    input_variables=[]
    #    )

    #chain=input_prompt_template | llm # langchain expression language first part(promptTemplate to be fed to second part(llm))
    #response = chain.invoke({}, config={"callbacks": [tracing_handler]}  )

    response=agent.invoke({
        "messages":HumanMessage("Please find best job openings for AI Engineering with langchain and Java Backend background with salary >= 40 lpa")
    },
    config={
        "callbacks": [tracing_handler]
    })
    print(response)

    flush_tracer()

main()