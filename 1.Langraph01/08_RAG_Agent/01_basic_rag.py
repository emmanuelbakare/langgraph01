# Basic implementation of Rag

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

### Function to Retrieve a bible text from api and combine it as one text chunk
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


### Retrieve the text using the bible_text function and store it as a list of Documents
docs =[
Document(page_content=bible_text("Luke 15"), metadata={"source":"Luke 15"}),
Document(page_content=bible_text("Psalm 1"), metadata={"source":"Psalm 1"}),
Document(page_content=bible_text("Psalm 124"), metadata={"source":"Psalm 124"}),
Document(page_content=bible_text("Psalm 2"), metadata={"source":"Psalm 2"}),

] 


### creat an embedding function, create an embedding database db, using the function and the document as parameters 
embedding_function = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embedding_function)

### Create a retriever configuration using the embedded db, then retrieve a question to get a set of similar result
#- it is from this 3 results (k=3) that you will now query (invoke) the final answer

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
qa_chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x)),
        "question": lambda x: x
    }
    # {
    #     "context": lambda x: format_docs(result),
    #     "question": lambda x: x
    # }
    | prompt
    | llm
    | StrOutputParser()

)

# while True:
#   final_answer=qa_chain.invoke(question)
final_answer = qa_chain.invoke(question)
print(final_answer)
