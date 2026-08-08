#from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from agentutils import get_llm,get_ollama_llm, get_agent
from tracer import get_tracer,flush_tracer




def main():

    tracing_handler = get_tracer()
    
    agent=get_agent()

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