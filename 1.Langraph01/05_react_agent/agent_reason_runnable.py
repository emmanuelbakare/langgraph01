from langchain_community.tools import TavilySearchResults
from langchain.agents import tool, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

import datetime

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format:str = "%Y-%m-%d %H:%M:%S"):
    """ Return the current date and time in the specified format"""
    current_time = datetime.datetime.now()
    formatted_time  =current_time.strftime(format)
    return formatted_time

llm = ChatOpenAI(model="gpt-4o")
tools = [search_tool,get_system_time]
react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
