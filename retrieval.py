
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore


from agentutils import get_llm
from ragutils import get_embeddings_model,get_vector_store

# get the embedding model
embeddings = get_embeddings_model()

# get the frontier llm
llm=get_llm()

vectorstore =get_vector_store()

# retrieve top three results from the vector store
retriever=vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template=ChatPromptTemplate.from_template(""" 
Answer the  question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:

""")

def format_docs(docs):
    """
    Format retrieved documents into a single string.
    """

    return "\n\n".join(doc.page_content for doc in docs)

# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================

def create_retrieval_chain_with_lcel():
    """

    Create a retrieval chain with LCEL(langchain Expression Language)
    Returns a chain that can be invoked with {"question":"..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator(|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with langchain's  type system
    - Better debugging: langchain provides better observability tools

    """

    retrieval_chain=(
        RunnablePassthrough.assign(
            context=itemgetter("question")
                    |retriever|format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents,formats then and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone

    """

    # Step 1: Retrieve relevant documents
    docs=retriever.invoke(query)

    # Step 2: format documents into context string
    context=format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages=prompt_template.format_messages(context=context,question=query)

    # Step 4: Invoke llm with the formatted messages
    response=llm.invoke(messages)

    return response.content

if __name__=="__main__":

    print("Processing Retrieval Chain:")
    query = "what is Pinecone in machine learning?"

    print("RAG implementation without LCEL")
    result_without_lcel=retrieval_chain_without_lcel(query)
    print("Answer:\n")
    print(result_without_lcel)

    print("RAG implementation with LCEL")
    chain_with_lcel=create_retrieval_chain_with_lcel()
    result_with_lcel=chain_with_lcel.invoke({"question":query})
    print("Answer:\n")
    print(result_with_lcel)