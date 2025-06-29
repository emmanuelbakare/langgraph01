from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

MAX_ITERATION = 2

graph = MessageGraph()

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)


graph.set_entry_point("draft") # from start node, add the draft node
graph.add_edge("draft", "execute_tools") # from draft node, add the execute tool node
graph.add_edge("execute_tools", "revisor")# from execute node add revisor node

def event_loop(state: List[BaseMessage])->str:
    # check the number of tools are in the state message.
    #this is to capture how many times the tool is called
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits

    #if the counted tool call is greater than the maximum allowed then end the program
    # if not, call the tool again.
    if num_iterations > MAX_ITERATION:
        return END
    return "execute_tools"

# from revisor node add tools node and let tools node decide either to goto END node or loop back to revisor node
graph.add_conditional_edges("revisor", event_loop) 

app = graph.compile()

response =app.invoke("Write about how small businesses can leverage AI to grow")

# answers = response[-1].tool_calls[0]["args"]["answer"]
answers = response.tool_calls[0]["args"]["answer"]
print(answers)

print("RESPONSE:\n", response)


#RESPONSE ERROR
"""

BadRequestError: Error code: 400 - 
{'error': {'message': "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: call_yy8PY5qnMOOJunpKfdiifC72", 'type': 'invalid_request_error', 'param': 'messages.[3].role', 'code': None}}

"""
