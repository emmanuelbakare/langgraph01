from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from dotenv import load_dotenv






load_dotenv()
# llm = ChatGroq(model="llama3-8b-8192")   # 8b model  - faster model
llm = ChatGroq(model="llama3-70b-8192")  # 70b model  x8 better but slower

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state:BasicChatState)->BasicChatState:
    return {
        "messages": [llm.invoke(state["messages"])]  #
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")

graph.add_edge("chatbot", END)

# add memory
memory= MemorySaver()

app = graph.compile(checkpointer=memory)

#tie the memory with a thread ID. A unique id that shows the conversation history as it builds
config = {"configurable":{
    "thread_id": 1
}}

while True:
    user_input = input("User: ")

    if user_input in ["exit","end"]:
        break 
    # remember to pass the configuration, thread id to the invoke method
    result = app.invoke({
        "messages":[HumanMessage(content=user_input)]
    }, config=config)
    # print(result)
    print("AI:",result["messages"][-1].content)

