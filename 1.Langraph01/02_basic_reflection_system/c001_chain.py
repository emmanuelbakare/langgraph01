from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

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