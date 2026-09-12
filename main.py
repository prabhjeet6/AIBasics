#from langchain_core.prompts import PromptTemplate
from typing import List, Any, Dict

from langchain_core.messages import HumanMessage

from agentutils import get_llm,get_ollama_llm, get_agent
from documentation_helper.retrieval import run_agent_for_retrieval
from tracer import get_tracer,flush_tracer
import streamlit as st


# Execution without Streamlit and frontend
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

# Commented implementation without Streamlit
#main()


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


# Implementation with Streamlit for frontend Support, runs on Streamlit server hosted locally on http://localhost:8501/.
# Command to run:
# uv run streamlit run main.py

st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = run_agent_for_retrieval(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)


