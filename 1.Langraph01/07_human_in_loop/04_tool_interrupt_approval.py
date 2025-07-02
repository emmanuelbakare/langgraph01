from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()


#Create Tavily tools and Initialize LLM and LLM with tools
memory =MemorySaver()

#Create Tools
search_tool  = TavilySearchResults(max_results=2)
tools = [search_tool]


# llm = ChatGroq(model="llama3-8b-8192")   # 8b model  - faster model
llm = ChatGroq(model="llama3-70b-8192")  # 70b model  x8 better but slower
llm_with_tools= llm.bind_tools(tools=tools)


## Create the nodes
## create the type schema
class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

#create chatbot node
def chatbot(state:BasicChatState)->BasicChatState:
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]  #
    }

#create conditional edge (this edge decides movement from chatbot node to either tools node or end)
def tools_router(state:BasicChatState):
    last_message = state["messages"][-1]

    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tools"
    return "stop"

#create a tool node
tool_node =ToolNode(tools=tools)

#Create The Graphs. on compile, add the checkpointer and the interrupt_before parameter

#create graph
graph = StateGraph(BasicChatState)

graph.set_entry_point("chatbot")
graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.add_conditional_edges(
    "chatbot",
    tools_router,
{
    "tools":"tool_node",
    "stop": END
})
graph.add_edge("tool_node", "chatbot")

app = graph.compile(checkpointer=memory, interrupt_before=["tool_node"])


#### Take user Input and generate the result

configure = {"configurable":
    {"thread_id":1}
}
user_input = input("Question: ")
events = app.stream(
    {"messages": [HumanMessage(content=user_input )]},
    config=configure,
    stream_mode="values"
)


for event in events:
    event["messages"][-1].pretty_print()


#show next node
snapshot = app.get_state(config=configure)
snapshot.next


# give the llm your input. this time we just give NONE so that it procceeds to run the Tavily tool and return the expected result

events = app.stream(None, configure, stream_mode="values")

for event in events:
    print(event["messages"][-1].pretty_print())

