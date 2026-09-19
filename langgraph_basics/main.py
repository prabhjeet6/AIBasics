from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState,StateGraph,END
from nodes import run_agent_reasoning,tool_node

AGENT_REASON="agent_reason"
ACT="act"
LAST=-1

# if AI message suggests a tool call, continue, else stop
def should_continue(state: MessagesState)->str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT
# create a langGraph with MessagesState
flow = StateGraph(MessagesState)

# Add reasoning node
flow.add_node(AGENT_REASON,run_agent_reasoning)

# starting point
flow.set_entry_point(AGENT_REASON)

# ACT node
flow.add_node(ACT,tool_node)

# After agent_reason finishes, call should_continue() to decide where to go.
# if the response is END, end, otherwise act
flow.add_conditional_edges(AGENT_REASON,should_continue,{END:END,ACT:ACT})

# This edge below creates a loop for langGraph, which differentiates it from langChain

# agent_reason
#       ↓
#     act
#       ↓
# agent_reason
#       ↓
#     act?
#     / \
#   yes  no
#   ↓     ↓
#  act    END
flow.add_edge(ACT,AGENT_REASON)

# langGraph compiles the graph definition into an executable app
app=flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__=="__main__":
    print("ReAct LangGraph with Function Calling")
    response=app.invoke({"messages":[HumanMessage(content="What is the temperature in Lucknow?List it and then triple it")]})
    print(response["messages"][LAST].content)





