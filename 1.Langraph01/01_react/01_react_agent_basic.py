from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

result = llm.invoke("Give me a fact about cats")

print(result.content)