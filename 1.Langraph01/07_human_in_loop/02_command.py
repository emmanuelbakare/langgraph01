from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from typing import TypedDict

class AgentState(TypedDict):
    text: str 

def node_a(state:AgentState):
    print("In node A")
    return Command(
        goto="node_b",
        update={
            "text" : state['text'] + "a"
        }
    )

def node_b(state:AgentState):
    print("In Node B")
    return Command(
        goto="node_c",
        update={
            "text": state['text'] + "b"
        }
    )

def node_c(state:AgentState):
    print("In Node C")

    return Command(
        goto=END,
        update={
            "text":state['text'] + "c"
        }
    )

graph = StateGraph(AgentState)
graph.set_entry_point("node_a")
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)

app = graph.compile()

result = app.invoke({
    "text":"HELLO"
})
print(result)

