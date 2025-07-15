from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools=tools)

def model(state:AgentState):
    return{
        "messages":[llm_with_tools.invoke(state["messages"])],
    }

def tools_router(state: AgentState):
    last_message = state["messages"][-1]

    if(hasattr(last_message,"tool_calls") and len(last_message.tool_calls) > 0):
        return "tools"
    else:
        return "stops"


#graph

tool_node = ToolNode(tools=tools)
graph = StateGraph(AgentState)

graph.add_node("model", model)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("model")

graph.add_conditional_edges("model", tools_router, {"tools":"tool_node","stops":END})
graph.add_edge("tool_node","model")

app = graph.compile()


##### using values
input = {
    "messages":["What's the current weather in Lagos"]
}

events = app.stream(input=input, stream_mode="values")

for event in events:
    print(event)

###### using updates
# input = {
#     "messages":["What's the current weather in Lagos"]
# }

# events = app.stream(input=input, stream_mode="updates")

# for event in events:
#     print(event)

