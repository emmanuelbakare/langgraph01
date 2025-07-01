from typing import TypedDict, Union, Annotated
from langchain_core.agents import AgentAction, AgentFinish
import operator 

class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentAction, None]
    intermediate_steps: Annotated[list[tuple[AgentAction,str]], operator.add]

