from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from schema import AnswerQuestion, ReviseAnswer
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
llm = ChatOpenAI(model="gpt-4o")
# pydantic_parser = PydanticToolsParser(tools=tools)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=tools,tool_choice='AnswerQuestion')

pydantic_validator= PydanticToolsParser(tools=tools)

#you can pass it through the pydantic_validator if you want to do validation on your response.
# it will however not respond with the appropriate format as an AIMessage
# first_responder_parsed  = first_responder_prompt_template | llm.bind_tools(tools=tools,tool_choice='AnswerQuestion') |pydantic_validator



#revisor section - This section is activate after the initial prompt (responder) as generated its response
# response of the responder will be the input for the revisor. i.e. The revisor revised the original (responder) response
revise_instructions = """
Revise your  previous answer using the new information
- You should use the revious critique to add important information to your answer.
- You MUST inlude numerical citations in your revised answer to ensure it can ver verified
- add a "References" section to the botton of your anwser (which does not count towards the word limit). 
in form of:
    - [1] https://example.com
    - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more th
250 words.
"""

# add the actor template and the revisor template together
revisor_chain = actor_prompt_template.partial(
    first_instruction = revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice='ReviseAnswer')


response = first_responder_chain.invoke({
    "messages":[HumanMessage(content="write me a blog post on how small business can leverage AI to grow")]
})

 