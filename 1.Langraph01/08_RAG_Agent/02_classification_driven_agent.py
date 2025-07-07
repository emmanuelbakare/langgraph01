# Clasification Driven Agent

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

#Function to Retrieve a bible text from api and combine it as one text chunk
import requests
import re

## get bible passage from https://bible-api and return a string version of the bible

def bible_text(passage: str) -> str:
    """
    Fetches a Bible passage from Bible-API and returns it in a formatted string:
    Example: "Luke 15:1-2 1. Text 2. Text"
    """
    url = f"https://bible-api.com/{passage.replace(' ', '%20')}?translation=kjv"
    response = requests.get(url)
    
    if response.status_code != 200:
        return f"Error fetching passage: {response.status_code}"

    data = response.json()
    reference = data.get("reference", "")
    verses = data.get("verses", [])
    
    merged_verses = " ".join(
        [f'{v["verse"]}. {v["text"].strip()}' for v in verses]
    )
    final_text = f"{reference} {merged_verses}"
    cleaned_text =re.sub(r'\s+', ' ', final_text).strip()

    return cleaned_text




#Retrieve the text using the bible_text function and store it as a list of Documents
docs =[
Document(page_content=bible_text("Luke 15"), metadata={"source":"Luke 15"}),
Document(page_content=bible_text("Psalm 1"), metadata={"source":"Psalm 1"}),
Document(page_content=bible_text("Psalm 124"), metadata={"source":"Psalm 124"}),
Document(page_content=bible_text("Psalm 2"), metadata={"source":"Psalm 2"}),

] 



# create an embedding function, create an embedding database db, using the function and the document as parameters
embedding_function = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embedding_function)
# Create a retriever configuration using the embedded db, then retrieve a question to get a set of similar result
# it is from this 3 results (k=3) that you will now query (invoke) the final answer



retriever =db.as_retriever(search_type="mmr", search_kwargs={"k":3})

question= "what happened when the boy spent all he had"
try:
    result = retriever.invoke(question)
    # print(result)  # uncomment this if you want to see the response- this is a list of document excerpts with similarity to the question answer
except Exception as e:
    print(e)

### generate a prompt template that will be used to finally query the  result (list of similar answers)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")  # get the llm


template = """Answer the question based on the following context: {context} 
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)  # create the promptTemplate

# function to convert the generated docs into one big chunk of text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#this creates the chain, retreives the context and question using the previous doc response and the previous answer
# this chain will use llm to only retreive result from the previous docs (context) (which is now merged into a big chunk of text with format_docs)
rag_chain = prompt | llm


# We start Building this section where We will maintain 3 different state
# List of Messages
# List of Document
# on_topic string that tells if the query is on topic- i.e. the answer can be found in the context

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.schema import Document
from langgraph.graph import add_messages, StateGraph, END, START

class AgentState(TypedDict):
    messages: list[BaseMessage]
    documents: list[Document]
    on_topic: str



#Define all the Nodes needed and the structure the database output will look like

from pydantic import BaseModel, Field

class GradeQuestion(BaseModel):
    """Boolean Value to check weather a question is relatable to the biblical issues """
    score: str = Field(description="Question is related to bible. If yes -> 'Yes, if no ->'No' ")

def question_classifier(state:AgentState):
    question = state["messages"][-1]
    system = """You are a classifier that help determine if a user question is relant to bible related story, history or parable.
    If the question is related to bible respond with 'Yes' if it is not respond with 'No'
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User Question: {question}")
        ]
    )

    structured_llm= llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt |  structured_llm
    result = grader_llm.invoke({"question":question})
    state["on_topic"] = result.score

    return state


def on_topic_router(state:AgentState):
    on_topic = state['on_topic']
    if on_topic.lower()=="yes":
        return "on_topic"
    else:
        return "off_topic"


def retrieve(state:AgentState):
    question = state["messages"][-1].content
    documents = retriever.invoke(question)
    state["documents"]=documents
    return state

def generate_answer(state:AgentState):
    question = state["messages"][-1].content
    documents = state["documents"]
    generation = rag_chain.invoke({"context":documents, "question":question})
    state["messages"].append(generation)

def off_topic_response(state:AgentState):
    state["messages"].append(AIMessage(content = "I am sorry I cannot answer this question"))
    return state
    
    

# Create the graph workflow

workflow = StateGraph(AgentState)

workflow.add_node("topic_decision",question_classifier)
workflow.add_node("off_topic_response",off_topic_response)
workflow.add_node("retrieve",retrieve)
workflow.add_node("generate_answer",generate_answer)

workflow.add_conditional_edges(
    "topic_decision",
    on_topic_router,
    {
        "on_topic": "retrieve",
        "off_topic": "off_topic_response"
    }
)

workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("off_topic_response", END)

workflow.set_entry_point("topic_decision")

graph = workflow.compile()



#execute a question
result = graph.invoke(
    input = {
        "messages":[HumanMessage(content="Why do you think killing the fatted calf made the brother angry")]
    }
)
print(result["messages"][-1].content)