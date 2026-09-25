from typing import TypedDict,Annotated,Literal

from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph import END,StateGraph
from langgraph.graph.message   import add_messages

from chains import generate_chain,reflection_chain

class MessageGraph (TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

REFLECT="reflect"
GENERATE="generate"

# A LangGraph node is just a python function with a state as an argument
def generation_node(state:MessageGraph):
    return {"messages":[generate_chain.invoke({"messages":state["messages"]})]}

def reflection_node(state:MessageGraph):
    response=reflection_chain.invoke({"messages": state["messages"]})
    return {"messages":[HumanMessage(content=response.content)]}

builder=StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE,generation_node)
builder.add_node(REFLECT,reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state:MessageGraph) -> Literal["reflect", "__end__"]:
    if(len(state["messages"])>6):
        return END
    return REFLECT
# As it is conditional edge, if it fails, it goes to END, otherwise, goes to REFLECT
# , after critiquing goes to GENERATE, which goes to should_continue, forming a feedback loop
builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph=builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
#   print(response)
    for i, message in enumerate(response["messages"], 1):
        print("\n" + "=" * 80)
        print(f"MESSAGE {i}: {type(message).__name__}")
        print("=" * 80)
        print(message.content)