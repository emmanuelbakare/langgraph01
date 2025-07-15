# use user defined tools  called get_sysyem_time to retrieve the current time



from dotenv import load_dotenv

from langchain.agents import create_agent
# from langchain_community.tools import TavilySearchResults
from langchain_tavily import TavilySearch
from tools import get_system_time

load_dotenv()





search_tool = TavilySearch(max_results=3)

tools = [search_tool, get_system_time]

agent = create_agent(
    # model="gpt-4o-mini",
    model="gpt-4o-mini",
    tools=tools,
    system_prompt="",

)

# agent.invoke("Give me 5 hot news from Nigeria today")
prompt = {
    "messages":[
        ("user","When was spaceX's last launch and how many days ago was that from this instant")
    ]
}
response = agent.invoke(prompt)

print(response["messages"][-1].content)

