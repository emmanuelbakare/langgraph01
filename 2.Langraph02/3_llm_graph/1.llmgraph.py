from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    message: List[HumanMessage]

llm =  ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')
def process_node(state:AgentState) -> AgentState:
    """ Get an LLM message"""

    result = llm.invoke(state['message'])
    print(result.content)
    return result

graph = StateGraph(AgentState)
PROCESS="Process"
graph.add_node(PROCESS, process_node)
graph.add_edge(START, PROCESS)
graph.add_edge(PROCESS, END)

app = graph.compile()
user_input =input("Enter: ")
while user_input!="exit":
    context= AgentState(message=[HumanMessage(content=user_input)])
    result = app.invoke(context)
    user_input =input("Enter: ")





 