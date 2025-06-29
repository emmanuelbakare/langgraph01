from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime 
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from schema import AnswerQuestion
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage

load_dotenv()
#Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         """ You are expert AI Researcher
         Current Time: {time}
         
         1. {first_instruction}
         2. Reflect and critique your answer. Be severe to maximize improvement.
         3. After the reflectiom, **List 1-3 search queires seperately** for researching improvements.
         Do not include them inside the reflection
         """),
         MessagesPlaceholder(variable_name="messages"),
         ("system",
          "Answer the user's question above using thre required format.")
    ]
).partial(  # .partials is used to prepopulate prompt template before invoking it on LLM
    time=lambda: datetime.datetime.now().isoformat()
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250word answer"
)
tools=[AnswerQuestion]
# llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')
llm = ChatOpenAI(model="gpt-4o")
pydantic_parser = PydanticToolsParser(tools=tools)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=tools,tool_choice='AnswerQuestion') | pydantic_parser

response = first_responder_chain.invoke({
    "messages":[HumanMessage(content="write me a blog post on how small business can leverage AI to grow")]
})

print(response)