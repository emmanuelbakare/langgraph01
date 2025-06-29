from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

search_tool = TavilySearchResults(search_depth="basic")

tools = [search_tool]

agent = initialize_agent(
    llm = llm,
    tools=tools,
    agent="zero-shot-react-description",
    verbose = True

)

# agent.invoke("Give me 5 hot news from Nigeria today")
agent.invoke("Israel and Iran are at war. As at today, what are the casualties from both side (death, injured...)")


