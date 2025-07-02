from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage 
from IPython.display import display, Markdown
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(model="llama3-70b-8192") 

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


GENERATE_POST = "generate_post"
POST = "post"
GET_REVIEW_DECISION = "get_review_decision"
COLLECT_FEEDBACK = "collect_feedback"

system_message=SystemMessage(content="Output only the response. Do not add a prefix or suffix description to the output, output your response in markdown format")
# system_message="Output only the response. Do not add a prefix or suffix description to the output"
def generate_post(state:AgentState)->AgentState:
    return {
        "messages":[system_message, llm.invoke(state['messages'])]
    }

def get_review_decision(state:AgentState):
    post_content = state['messages'][-1].content

    print(f"\n Current LinkedIn Post\n{post_content}\n")

    decision = input("Post to LinkedIn (yes/no): ")

    if(decision.lower()=="yes"):
        return "post"
    return "feedback"

def post(state:AgentState):
    final_post = state['messages'][-1].content

    display(Markdown("## Final Output"))
    display(Markdown(f"{final_post}"))
    # print(f"\nFinal LinkedIn Post\n {final_post}")
    # print("\n Post have been approved and is now live on LinkedIn")



def collect_feedback(state:AgentState)->AgentState:
    feedback = input("How can I improve this post? ")
    return {
        "messages":[HumanMessage(content=feedback)]
    }

# Create Graph

graph = StateGraph(AgentState)

graph.add_node(GENERATE_POST, generate_post)
# graph.add_node(GET_REVIEW_DECISION, get_review_decision)
graph.add_node(COLLECT_FEEDBACK, collect_feedback)
graph.add_node(POST, post)

graph.set_entry_point(GENERATE_POST)
graph.add_conditional_edges(GENERATE_POST, 
                            get_review_decision,
                           {
                               "post":POST,
                               "feedback": COLLECT_FEEDBACK,
                           })
graph.add_edge(POST, END)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

app = graph.compile() 

user_input = "Write a LinkedIn post abouton AI Agent taking over content creation"
response = app.invoke({
    "messages":[HumanMessage(content=user_input)]
})

# print(response)

#display a markdown of the final output
# from IPython.display import display, Markdown
# display(Markdown("## Final Output"))
# display(Markdown(response['messages'][-1].content))
    
     

