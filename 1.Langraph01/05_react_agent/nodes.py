from dotenv import load_dotenv

from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState

load_dotenv()

def reason_node(state:AgentState)->AgentState:
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


def act_node(state:AgentState)->AgentState:
    agent_action = state['agent_outcome']

    # extract tool name and input from action
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    #find matching tool function
    tool_function = None
    for tool in tools:
        if tool.name ==tool_name:
            tool_function = tool 
            break

    #execute the tool with the input
    if tool_function:
        if isinstance(tool_function, dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = f"Tool {tool_name} not found"
    
    return {"intermediate_steps":[(agent_action, str(output))]}