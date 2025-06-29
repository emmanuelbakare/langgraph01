#store the message  in an external file so that next time you log in, the conversation can continue
# add memory to your application

from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm =  ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

def process_node(state:AgentState) -> AgentState:
    """ Get an LLM message"""

    result = llm.invoke(state['message'])
    # Append the ai response to the existing human message.
    state['message'].append(AIMessage(content=result.content))
    
    print("\nResult: ",result.content)
    return result

# CREATE GRAPH
PROCESS= "Process"
graph = StateGraph(AgentState)
graph.add_node(PROCESS, process_node)
graph.add_edge(START, PROCESS)
graph.add_edge(PROCESS, END)

app = graph.compile()

user_input=input("Enter: ")
conversation = []
while user_input != "exit!":
    conversation.append(HumanMessage(content=user_input))
    
    result = app.invoke({"message":conversation})
    conversation  = result['messages']
    user_input=input("Enter: ")

# this is the part where the message is stored inside a logging.txt file
print("Commence Writting Conversation to Log File")
with open('logging.txt', 'w') as file:
    file.write("Your Conversation Log \n")

    for message in conversation:
        if isinstance(message,HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")

    file.write("End of Conversation")
print("Conversation Saved to logging.txt file")

