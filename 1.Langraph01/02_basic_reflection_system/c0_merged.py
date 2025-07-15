from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START,  MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
# from c00_chain import generation_chain, reflection_chain
# from c001_chain import generation_chain, reflection_chain

load_dotenv()

#==================Prompt Generation



load_dotenv()

generation_prompt = ChatPromptTemplate.from_messages(
   [
       (
        "system", "You are a twitter techie influencer assitance tasked with writing excellent twitter post"
         "Generate the best twitter post possible for the user's request."
        "If the user provides citiques, respond with a revised version of your previous attempts"
       ),
       MessagesPlaceholder(variable_name="messages")
   ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
   [
       (
        "system",  "You are a viral twitter influcer grading a tweet. Generate critique and recommendation for the user's tweet"
        "Always provide detailed recommendations, including request for length, virality, style, etc",
       ),
       MessagesPlaceholder(variable_name="messages")
   ]
)

llm =  ChatGroq(model='llama-3.3-70b-versatile')

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm 







#=================Graph Generation
graph = StateGraph(MessagesState)

GENERATE="generate"
REFLECT = "reflect"

def generation_node(state):
    return generation_chain.invoke(
        {"messages":state}
    )

def reflection_node(state):
    response = reflection_chain.invoke({
        "messages":state
    })
    return [HumanMessage(content = response.content)]

graph.add_node(GENERATE, generation_node)
graph.add_node(REFLECT, reflection_node)

# graph.set_entry_point(GENERATE)
graph.add_edge(START,GENERATE)

# conditional statement to test where next to go
def should_continue(state):
    if(len(state) > 2):
        return "end" 
    return "reflect"

graph.add_conditional_edges(GENERATE, should_continue)

graph.add_edge(REFLECT, 
               GENERATE,
               {
                   "end": END,
                   "reflect": REFLECT
               })

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()
print("="*80)

response =app.invoke()