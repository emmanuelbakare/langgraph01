# February 2026 implementation of agent
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

search_tool = TavilySearch(max_results=3)

tools = [search_tool]
SYSTEM_PROMPT ="""give your answer in a paragraph or as a list. add the date and time of the occurrence of this news"""
agent = create_agent(
    model = "gpt-4o-mini",
    tools=tools,
    system_prompt=SYSTEM_PROMPT

)

msg1 ="Israel and Iran are at war. As at today, what are the casualties from both side (death, injured...)"
msg2 ="Give me 5 hot news from Nigeria today"
prompt = {
    "messages":[
        ("user",msg2)
    ]
}
# prompt =ChatPromptTemplate.from_messages("Israel and Iran are at war. As at today, what are the casualties from both side (death, injured...)")
# agent.invoke("Give me 5 hot news from Nigeria today")
response= agent.invoke(prompt)
final_response = response["messages"][-1].content
print(f"=======PROMPT========\n{prompt}\n\n ====RESPONSE====\n {final_response}")


