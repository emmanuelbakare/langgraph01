from langchain_core.agents import AgentAction,AgentFinish
from langgraph.graph import StateGraph, END

from nodes import reason_node, act_node
from react_state import AgentState
from dotenv import load_dotenv

load_dotenv()

REASON_NODE="reason_node"
ACT_NODE="act_node"

def should_continue(state:AgentState)->str:
    if isinstance(state['agent_outcome'], AgentFinish):
        return "stop"
    return "action" 

flow = StateGraph(AgentState)

flow.add_node(REASON_NODE, reason_node)
flow.add_node(ACT_NODE, act_node)

flow.set_entry_point(REASON_NODE)
flow.add_conditional_edges(
    REASON_NODE,
    should_continue,
    {
        "stop": END,
        "action": ACT_NODE

    }
)
flow.add_edge(ACT_NODE,REASON_NODE)

app = flow.compile()

result= app.invoke({
    "input": "How many days ago was the latest SpaceX launch",
    "agent_outcome": None,
    "intermediate_steps": []
}) 

print(result)
print()
print(result['agent_outcome'].return_values["output"], "final result")



