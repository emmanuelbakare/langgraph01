# use user defined tools  called get_sysyem_time to retrieve the current time



from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
from tools import get_system_time

load_dotenv()




llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

search_tool = TavilySearchResults(search_depth="basic")

tools = [search_tool, get_system_time]

agent = initialize_agent(
    llm = llm,
    tools=tools,
    agent="zero-shot-react-description",
    verbose = True

)

# agent.invoke("Give me 5 hot news from Nigeria today")
agent.invoke("When was spaceX's last launch and how many days ago was that from this instant")


