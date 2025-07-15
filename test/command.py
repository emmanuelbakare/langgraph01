from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import Command

class StateAgent(TypedDict):
    text: str 

def node_a(state:StateAgent):
    print("Inside Node A")

    return Command(
        goto="node_b",
        update={
            "text": state['text'] + "A"
        }
    )

def node_b(state:StateAgent):
    print("Inside Node B")

    return Command(
        goto="node_c",
        update={
            "text" :state['text'] + "B"
        }

    )

def node_c(state:StateAgent):
    print("Inside Node C")

    return Command(
        goto=END,
        update={
            "text" :state['text'] + "C"
        }

    )

graph = StateGraph(StateAgent)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.set_entry_point("node_a")

app = graph.compile()
result= app.invoke({"text":""})
print(result['text'])
