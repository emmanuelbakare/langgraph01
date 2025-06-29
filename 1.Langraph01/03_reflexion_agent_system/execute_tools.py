import json
from typing import List, Dict, Any 
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_community.tools import TavilySearchResults
# create the Tavily Search tool
tavily_tools = TavilySearchResults(max_results=3)



def execute_tools(state:List[BaseMessage]) ->List[BaseMessage]:
    #from the list of message, get the last message (which is usally an AIMessage tool message)
    last_ai_message = state[-1] 

    #extract tool calls from the AI Message
    # start by checking if the last ai message is a tool call, if not return []
    if not hasattr(last_ai_message,"tool_calls") or not last_ai_message.tool_calls:
        return []
    
    tool_messages = [] #this will store all the tools
    # if last_ai_message is a tool then loop through all the available tools
    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]: # is there a tool called AnswerQuestion or ReviseAnswer
            call_id = tool_call["id"] #if you found any of then the get their id
            search_queries = tool_call["args"].get("search_queries", []) # also get the search_queries list data inside args dict

            #loop through the list in search_queries and store the result in query_results
            #each search_queries item contain a query string. This will be passed to tavily to generate respond
            query_results = {}
            for query in search_queries:
                result = tavily_tools.invoke(query) # this uses tavily tool to generate a list of 3 response for each query in search_queries
                query_results[query]= result # store the response in the query_results dictionary

            #create a tool with the generated result
            tool_messages.append(
                ToolMessage(
                    content = json.dumps(query_results),
                    tool_call_id = call_id,
                )
            )
    return state + tool_messages