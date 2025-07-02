from langgraph.graph import add_messages, StateGraph, END, START
from langgraph.types import Command, interrupt
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
import uuid
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192") 


class State(TypedDict):
    linkedin_topic: str
    generated_post: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]


def model(state:State):
    """ Here' we're using the LLM to generate a LinkedIn post with human feedback incorporated"""

    print("[model] Genering Content")
    linkedin_topic = state["linkedin_topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else ["No Feedback yet"]

    # "Here, we define the prompt"
    
    prompt= f"""
LinkedIn Topic : {linkedin_topic}
Human Feeback: {feedback[-1] if feedback else "no feedback yet"} 
Generate a structured and well-written LinkedIn post based on the given topic.

Consider previous human feedback t refine the response
"""

    response = llm.invoke([
        SystemMessage(content="you are an expert LinkedIn content writer"),
        HumanMessage(content=prompt)
    ])

    generated_linkedin_post= response.content

    print(f"[model_node] Generated post: \n {generated_linkedin_post }")

    return {
        "generated_post":[AIMessage(content=generated_linkedin_post)],
        "human_feedback": feedback
    }


def human_node(state:State):
    """ Human Intervention node - loops back to model unless input is done"""

    print("\n [human_node] awaiting human feedback...")

    generated_post = state['generated_post']

    #Interrupt to get user feeback

    user_feedback = interrupt(
        {
            "generated_post": generated_post,
            "message":"Provide feedback or type 'done' to finish    "
        }
    )

    print(f"[human_node] Retrieve human feeback: {user_feedback}")

    #if user types "done", transition to END node 
    if user_feedback.lower()=="done":
        return Command(
            update={"human_feedback":state["human_feedback"]},
            goto = "end_node"
        )
    
    #otherwise, feedback and return to model for re-generation
    return Command(
        update={"human_feedback":state['human_feedback'] + [user_feedback]},
        goto ="model"
    )

def end_node(state:State):
    """Final node"""

    print("\n [end_node] process finished")
    print("Final Generated Post:", state["generated_post"][-1])
    print("Final Human Feedback",state["human_feedback"])
    return {
        "generated_post": state["generated_post"],
        "human_feedback": state["human_feedback"]
    }

#building the Graph

graph = StateGraph(State)

graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)

# Defin the flow

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")
graph.add_edge("end_node", END)

#Enable Interrupt mechanism
checkpointer=MemorySaver()
app = graph.compile(checkpointer=checkpointer)

thread_config = {"configurable":{
    "thread_id": uuid.uuid4()
}}

linkedin_topic = input("Enter your LinkedIn topic: ")

initial_state = {
    "linkedin_topic": linkedin_topic,
    "generated_post": [],
    "human_feedback": []
}

for chunk in app.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():
        # if we reach an interrupt, continously ask for human feedback

        if(node_id=="__interrupt__"):
            while True:
                user_feedback = input("Provide feedback (or type 'done' when finished): ")

                # resume the graph executioin with the user's feedback
                app.invoke(
                    Command(resume=user_feedback),
                    config=thread_config
                )

                # Exit loop if user says done
                if user_feedback.lower() == "done":
                    break



