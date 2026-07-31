from langchain_core.prompts import PromptTemplate
from llm import get_llm,get_ollama_llm


from tracer import get_tracer,flush_tracer



def main():

    tracing_handler = get_tracer()

    #llm = get_llm()
    
    llm =get_ollama_llm()

    prompt_message="Tell me a startup idea that can help me half a million dollars quickly and tell me which llm(exact name) am i talking to, and is it free"
    
    input_prompt_template=PromptTemplate(
        template=prompt_message,
        input_variables=[]
        )

    chain=input_prompt_template | llm # langchain expression language first part(promptTemplate to be fed to second part(llm))
    response = chain.invoke({}, config={"callbacks": [tracing_handler]}  )

    print(response.content)

    flush_tracer()

main()