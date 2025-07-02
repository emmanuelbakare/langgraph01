from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv



#Create Tools
search_tool  = TavilySearchResults(max_results=2)
tools = [search_tool]

load_dotenv()
# llm = ChatGroq(model="llama3-8b-8192")   # 8b model  - faster model
llm = ChatGroq(model="llama3-70b-8192")  # 70b model  x8 better but slower
llm_with_tools= llm.bind_tools(tools=tools)



class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

#create chatbot node
def chatbot(state:BasicChatState)->BasicChatState:
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]  #
    }

#create conditional node
def tools_router(state:BasicChatState):
    last_message = state["messages"][-1]

    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    return END

#create a tool node
tool_node =ToolNode(tools=tools)


#create graph
graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot",tools_router)
graph.add_edge("chatbot", END)

app = graph.compile()

while True:
    user_input = input("User: ")

    if user_input in ["exit","end"]:
        break 

    result = app.invoke({
        "messages":[HumanMessage(content=user_input)]
    })
    # print(result)
    print("AI:",result["messages"][-1].content)

