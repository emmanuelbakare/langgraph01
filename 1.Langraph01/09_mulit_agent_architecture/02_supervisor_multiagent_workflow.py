from typing import Annotated, Sequence, List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

load_dotenv()

llm=ChatOpenAI(model="gpt-4o")
tavily_search = TavilySearchResults(max_results=2)
repl = PythonREPLTool()


class Supervisor(BaseModel):
    next: Literal["enhancer","researcher","coder"] = Field(
        description="""Determines which specialist to activate next in the workflow sequence: 
            'enhancer' when user input requires clarification, expansion, or refinement.
            'researcher' when additional facts, context, or data collection is necesarry
            'coder' when implementation, computation, or technical problem-solving is required
        """
    )
    reason: str = Field(description="""Detailed justification for the routing decision, explaining the rationale 
                    behind selecting the particular specialist and how this advances the task towards completion""")

def supervisor_node(state:MessagesState) -> Command[Literal["enhancer","researcher", "coder"]]:
    system_prompt=""" 
        You are a workflow supervisor managing a team of three specilaised agetns: Prompt Enhancer, Researcher, and Coder.
        Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state
        and needs of the tasks. Provide a clear, concise rationale for each decision to ensure transparency in your
        decision-making process.

        **Team Members**:
        1. **Prompt Enhancer**: Always consider this agent first. They clarify ambiguous requests, improve poorly defined queries,
        and esnire the task is well-structured before deeper processing begins.
        2. **Researcher**: Specilizes in information gathering, fact-finding, and collecting relevant data needed to address the 
        user's request.
        3. **Coder**: Focuses on technical implementation, calculation, data analysis, algorithm development, and coding solutions.

        **Your Responsibilities**:
        1. Analyze each user request and agent response for completeness, accuracy, and relevance.
        2. Route the task to the most approriate agent at each decision point.
        3. Maintain workflow momentum by avoiding redundant agent assignments.
        4. Continue the prcess util the users request is fully and satisfactorily resolved.

        Your objective is to create an efficient workflow that leverages each agent's strength while minimizing 
        unncessary steps, ultimately delivering complete and accurate solutions to user requests.
    """

    messages = [
        {
            "role":"system",
            "content":system_prompt
        }] + state["messages"]

    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason 

    print(f"--- Workflow Transition: Supervisor - {goto.upper()} ----")
    return Command(
        update={
            "messages":[
                HumanMessage(content=reason, name="supervisor")
            ]
        },
        goto = goto,
    )
    

def enhancer_node(state:MessagesState)->Command[Literal["supervisor"]]:
    """
        Enhancer agent node that improves and clarifies user queries.
        Take the original user input and transforms it into a more precise,
        actionable request before passing it to the supervisor.
    """

    system_prompt = (
        "You are a Query Refinement Sepcialise with expertise in transforming vaue requiest into precise instruction. Your responsibilities include:\n\n"
        "1. Analyzing the original query to identify key ntent and requirements\n"
        "2. Resolving any ambiquities without requesting additional user input\n"
        "3. Expanding underdeveloped aaspcts of the query with reasonable assumptions\n"
        "4. Restructuring the query for clarity and actionability\n"
        "5. Ensuring all technical terminology is properly defined in context\n"
        "Important: Never ask questions back to the user. Instead, make informed assumptions and create the most comprehensibe version of their request possible."
        
    )

    messages = [
        {
            "role":"system",
            "content":system_prompt
        }
    ]+ state["messages"]

    enhanced_query = llm.invoke(messages)
    print(f"---- Workflow Transition: Prompt Enhancer - Suervisor ----")

    return Command(
        update={
               "messages":[
                    HumanMessage(content=enhanced_query.content,
                          name="enhancer")
               ]
        },
        goto = "supervisor"
    )



def researcher_node(state:MessagesState)->Command[Literal["validator"]]:
    
    """
    Researcher agent node that gathers information using Tavily search.
    Takes the current task state, performs relevant research,
    and returns findings for validation.
    """

    research_agent = create_react_agent(
            llm,
            tools=[tavily_search],
            # state_modifier = """ 
            system_message = """ 
                You are an information specialist with expertise in comprehensive research.
                 your responsibilities include:\nn\
                 1. Identfiy key information needs based on the query context\n
                 2. Gathering relevant, accurate, and up-to-date informaiton from reliable sources\n
                 3. Organizing findings in a structured, easly digestible format\n
                 4. Citing sources when possible to establish credibility
                 5. Focusing exclusively on information gathering -avoid analysis or implementations\n\n
                 Provide thorough, factual responses without speculation where ifnrmation is unavailable
            """
        )
    result = research_agent.invoke(state)
    print(f"--- Workflow Transition: Research - Validator ----" )

    return Command(
        update={
            "messages":[
                HumanMessage(
                    content=result["messages"][-1].content,
                    name = "researcher"
                )
            ]
        },
        goto="validator"
    )
    

def coder_node(state:MessagesState)->Command[Literal["validator"]]:

    code_agent = create_react_agent(
        llm,
        tools =[repl],
        state_modifier="""
            You are a coder and analyst. Focus on mathematical calculations, analyzing, solving math questions,
            and executing code. Handle technical problem-solving and data tasks
        """
    )

    result = code_agent.invoke(state)
    print(f"---- Workflow Transition: Coder -- validator -----")

    return Command(
        update={
            "messages":[
                HumanMessage(content=result["messages"][-1].content,
                name="coder"
                )
            ]
        },
        goto="validator"
    )

# System prompt providing cleear instruction to the validator agent

system_prompt = """
    your task is to ensure reasonaly quality. 
    Specifically, you must:
    - Review the user's question (the first message int he workflow).
    - Review the answers (The last message in the workflow).
    If the answer addresses the core intent of the quesiton, even if not perfectly,
    signal to end the workflow with 'FINISH'
    - Only route back to the supervisor if the answer is completely off-topic, hamrful, or 
    fundamentally misunderstands the question.

    - Accept answer that are "good enough" rather than perfect
    - Prioritize workflow completion over perfect responses
    - Give benefit of doubt to borderline answers

    Routing Guidelines:
    1. 'supervisor' Agent: ONLY for responses that are completely incorrect or off-topic.
    2. Respond with 'FINISH' in all other cases to end the workflow.
"""


class Validator(BaseModel):
    next: Literal["supervisor","FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor to continue or 'FINISH' to terminate"
    )
    reason: str = Field(
        description="The reason for the decision"
    )

def validator_node(state:MessagesState)->Command[Literal["supervisor", "__end__"]]:
    user_question = state["messages"][0].content
    agent_asnwer = state["messages"][-1].content
    
    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":user_question},
        {"role":"assitant", "content":agent_answer},
    ]
    
    response = llm.with_structured_output(Validator).invoke(messages)
    
    goto = response.next
    reason = response.reason
    
    if goto == "FINISH" or goto ==END:
        goto = END
        print("--- Transitioning to END ---")
    else:
        print(f"--- Workflow Transition: Validator -> Supervisor ---")
    
    return Command(
        update={
            "messages":[
                HumanMessage(content = reason, name= "validator")
            ]
        },
        goto=goto
    )
            

graph = StateGraph(MessagesState)

graph.add_node("supervisor",supervisor_node)
graph.add_node("enhancer",enhancer_node)
graph.add_node("researcher",researcher_node)
graph.add_node("coder",coder_node)
graph.add_node("validator",validator_node)

graph.add_edge(START,"supervisor")
app = graph.compile()




import pprint

inputs = {
    "messages":[
        ("user","Weather in Channai")
    ]
}

for event in app.stream(inputs):
    for key, value in event.items():
        if value is None:
            continue
        last_message = value.get("messages",[])[-1] if "messages" in value else None
        if last_message:
            pprint.pprint(f"Output from node {key}: ")
            pprint.pprint(last_message, indent=2, width=80, depth=None)
            print()