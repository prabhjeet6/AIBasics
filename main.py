from langchain_core.prompts import PromptTemplate
from llm import get_llm

def main():

    llm = get_llm()

    response = llm.invoke(
        "Who won the Cricket World Cup in 2011? Answer in one sentence."
    )

    print(response.content)



main()