from langgraph.graph import StateGraph, END, START
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class AgentState(TypedDict):
    text: str 

memory = MemorySaver()

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
    user_interrupt = interrupt("Which node do you want to go to (c or d)? ") 
    print("Review Choice: ", user_interrupt)
    if(user_interrupt.lower()=="c"):
        return Command(
            goto="node_c",
            update={
                "text": state['text'] + "b"
            }
        )
    elif user_interrupt.lower()=="d":
        return Command(
            goto="node_d",
            update={
                "text": state['text'] + "b"
            }
        )
    else:
        return Command(
            goto="node_a",
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

def node_d(state:AgentState):
    print("In Node D")

    return Command(
        goto=END,
        update={
            "text":state['text'] + "d"
        }
    )

config = {"configurable":{
    "thread_id": 1
}}

graph = StateGraph(AgentState)
graph.set_entry_point("node_a")
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d) 

app = graph.compile(checkpointer=memory)

initial_input= {"text": "HELLO "}
first_result = app.invoke(initial_input, config, stream_mode="updates" )
first_result

print(app.get_state(config).next)

