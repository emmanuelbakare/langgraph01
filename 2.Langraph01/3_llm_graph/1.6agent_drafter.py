"""
Task:
Our company is not working efficiently! we spend way too much time drafting documents and this need to be fixed.

For the company, you need to create an AI Agentic System that can **speed up drafting documents, emails, etc**. The AI Agentic System should have **Human-AI Collaboration** meaning the Human should be able to provide continours feedback and the AI agent should stop when the Human is happy with the draft. The system should also **be fas and be able to save the drafts*.
"""

from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain.agents import tool, initialize_agent
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

#this is the global variable to store document
document_content = ""



@tool
def update(content:str)->str:
    """Updates the document with the provided content"""

    global document_content
    document_content = content
    return f"Document have been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename:str) ->str:
    """
    Save the current document to a text file and finish the process

    Args:
        filename: name of the text file
    """
    global document_content

    if not filename.endswith(".txt"): 
        filename = f"{filename}.text"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"Document have been saved to: {filename}")
        return f"Document have been saved successfully to: '{filename}'"
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update, save]
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20").bind_tools(tools)
# model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent_node(state:AgentState)->AgentState:
    system_message= SystemMessage(content=f"""
You are a Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

- If the user want to update and modify content, use the 'update' tool with the complete updated content.
- If the user wants to save and finish, you need to use the 'save' tool.
- Make sure to always show the current document state after modifications

the current document content is: {document_content}
""")
    
    if not state['messages']:
        user_input="I'm ready to help you update a document, what would you like to create? "
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document ")
        print("\nUSER:", user_input)
        user_message=HumanMessage(content=user_input)

    all_messages =  [system_message] + list(state['messages']) + [user_message]

    response = model.invoke(all_messages)

    print("\n AI: ", response.content)
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\n USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages":list(state['messages']) + [user_message, response]}

def should_continue(state:AgentState)->AgentState:
    """ Determine if we should end or continue the conversation"""

    messages= state['messages']

    if not messages:
        return "continue"
    
    #This looks for the most recent tools message...
    for message in reversed(messages):
        #... and check if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and
        "saved" in message.content.lower() and
        "document" in message.content.lower()):
            return "end" #goes to the end edge which lead to the endpoint
        
    return "continue"

def print_message(messages):

    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print("\nTOOL RESULT:", message.content)

 
graph  =StateGraph(AgentState)
AGENT = "agent"
TOOL = "tools"

graph.add_node(AGENT,agent_node)
graph.add_node(TOOL, ToolNode(tools))

graph.add_edge(START,AGENT)
graph.add_edge(AGENT,TOOL)

graph.add_conditional_edges(
    TOOL,
    should_continue,
    {
        "continue": AGENT,
        "end": END,

    }
)

app= graph.compile()

# this code allow to run the program and generate output
def run_document_agent():
    print("\n ======DRAFTER=====")

    state = {"messages":[]}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")

if __name__=="__main__":
    run_document_agent()











