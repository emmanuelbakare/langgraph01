from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()




class AgentState(TypedDict):
    messages: List[Dict[str,str]]

input_data =  {
    "messages":[
        {
            "role":"user",
             "content": "add 2 , 5 , 10 , 44 together", 
        }
    ]
}


def add_node(state:AgentState)->AgentState:
    """add numbers together"""
    user_input = state["messages"][-1]['content']
    if "add" in user_input.lower():
        nums = [ int(num) for num in user_input.split() if num.isdigit()]
        total = sum(nums)
        print(nums)
        response = { "role":"assistant","content":str(total)}
    else:
        response = { "role":"assistant","content":"I dont understand your message"}

    return  {"messages": state['messages'] + [response]}


result = add_node(input_data)
print("RESULT \n",result)




