from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage, FunctionMessage
from langchain.agents import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# get document retriever function 
def get_retriever(file_path):
    #get embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    #get file and folder path
    script_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(script_dir, file_path)  #file path
    persist_directory = script_dir  #folder path

    collection_name = "resume_checker" #collection name
    chroma_db_path = os.path.join(persist_directory, "chroma.sqlite3") #path to chroma.db

    if not os.path.exists(pdf_path): # if pdf file doesnt exist
        raise FileNotFoundError(f'PDF file not found in {pdf_path}')

    #if chroma.db already exist, just load it
    #if it does exist create a new one
    if os.path.exists(chroma_db_path):
        print("Loading existing Chroma DB...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
    else:
        print("Creating new Chroma DB...")

        #load the pdf
        pdf_loader = PyPDFLoader(pdf_path)
        try:
            pages = pdf_loader.load()
            print(f"Document contains {len(pages)} pages")
        except Exception as e:
            print("Error Loading PDF", e)
            raise

        #split the pdf
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        pages_split = text_splitter.split_documents(pages) # pass the loaded pages to the splitte config to created splitted docs

        #create the chroma.db (vector database)
        try:
            vectorstore = Chroma.from_documents(
                documents=pages_split,
                embedding=embeddings,
                persist_directory=persist_directory, #optional parameter if not provided, data will be in-memory only
                collection_name=collection_name  # optional  - use to give name to embeddings in the vector database.
            )
            # vectorestore could also only have document and bedding parameter
            # vectorstore = Chroma.from_documents(documents=pages_split, embedding=embeddings) 

            print("Chroma DB created and persisted.")
        except Exception as e:
            print(f"Error Setting Chroma DB: {str(e)}")
            raise

    #user the database to gerate a retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    return retriever


retriever = get_retriever("resume.pdf")

@tool
def retrieval_tool(query:str)->str:
    """
    This tool searches record from a resume and returns the result
    """
    print('retrieving answer')
    docs = retriever.invoke(query)

    if not docs:
      return "I found no releveant information in the Document"  
    
    results =[]

    for i, doc in enumerate(docs,1):
        results.append(f"Document {i}:\n {doc.page_content}")
    
    return "\n\n".join(results)

tools = [retrieval_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

system_prompt="""
You are intelligent AI assistant who answers questions about resume uploaded in your konwledge base
Use the retriever tool available to answer questions about the resume. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answer
"""

def call_llm(state: AgentState) -> AgentState:
    """ Function to call the LLM with the current state."""
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    message = llm.invoke(messages)
    print(state)
    return {"messages": [message]}  # LangGraph appends this to state['messages']


def should_continue(state:AgentState):
    """ Check if the last message contains tool calls"""
    result = state['messages'][-1]

    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0



tools_dict = {our_tool.name: our_tool for our_tool in tools} # creating a dictionary from our list of tools


#LLM Agent

 




def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    
    for t in tool_calls:
        tool_name = t['name']
        query = t['args'].get('query', '')
        if tool_name not in tools_dict:
            content = "Error: tool not found"
        else:
            content = tools_dict[tool_name].invoke(query)
        tm = ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(content))
        results.append(tm)

    # Return full message list: this lets LangGraph append correctly
    return {"messages": results}



graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent",take_action)

graph.set_entry_point("llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True:"retriever_agent", False: END}
)

graph.add_edge("retriever_agent","llm")

rag_agent = graph.compile()

def running_agent():
    print("\n*** RAG AGENT***")

    while True:
        user_input = input("\n What is your question: ")
        if user_input.lower() in ['exit','quit']:
            break

        messages = [HumanMessage(content=user_input)] # converts back to Human type

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

running_agent()
