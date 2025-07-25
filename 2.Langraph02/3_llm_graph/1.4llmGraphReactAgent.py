from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain.agents import tool, initialize_agent
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()




@tool
def add(a:int, b:int):
    """This is an Addition function that adds two numbers together"""
    return a + b 

tools = [add]

#invoke llm and bind tools to it.
# model = ChatOpenAI(model="gpt-4o").bind_tools(tools) # for open AI
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20").bind_tools(tools)


class AgentState(TypedDict):
    #this property uses add_meessage to concatenate messages passed to 'messages' instead of overwriting them
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content="You are my AI Assistant. Please answer my query to the best of your ability")

    response = model.invoke([system_prompt] + state['messages'])
    return {"messages":[response]}

def should_continue(state:AgentState):
    messages = state['messages']
    # check is there is a tool to call on the last message
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'continue'
    else:
        return 'end'

AGENT="Our Agent"
TOOL = "Tools"
graph= StateGraph(AgentState)

graph.add_node(AGENT, agent_call)
tool_node = ToolNode(tools=tools)
graph.add_node(TOOL, tool_node)

graph.add_edge(START, AGENT)
graph.add_conditional_edges(
    AGENT,
    should_continue,
    {
        "continue": TOOL,
        "end": END
    }
)

graph.add_edge(TOOL, AGENT)

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]

        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages":[("user", "Add 3 + 4. Add 20 and 7 and write me a joke")]}
print_stream(app.stream(inputs, stream_mode="values"))
